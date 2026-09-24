#!/bin/bash
# Pull the pinned Maverick UD-Q4_K_XL shards onto local disk, checked.
#
#   tools/gguf_pull.sh <dest_dir> [--floor MBPS] [--no-hash]
#   tools/gguf_pull.sh --probe-only [--floor MBPS] <url>
#
# 1. Probe: eight parallel 256 MiB range requests against the first shard.
#    Every request must exit 0, answer 206 and deliver exactly its range,
#    within a timeout; the rate is the bytes that ARRIVED over the wall time.
#    Below --floor (default 100 MB/s, about a 40-minute pull) it fails.
# 2. Download the five shards at the pinned revision with the hub client
#    (hf_transfer when installed), then check every size against
#    profiler/data/maverick-ud-q4_k_xl-shards.json and, unless
#    --no-hash, every sha256 against the LFS object ids recorded there.
# Needs: curl, python3 with huggingface_hub; HF_TOKEN optional (the repo is
# public).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PINS="$HERE/profiler/data/maverick-ud-q4_k_xl-shards.json"
FLOOR=100; HASH=1; PROBE_ONLY=; DEST=; URL=
while [ $# -gt 0 ]; do
  case $1 in
    --floor) FLOOR=$2; shift 2 ;;
    --no-hash) HASH=0; shift ;;
    --probe-only) PROBE_ONLY=1; shift ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) if [ -n "$PROBE_ONLY" ]; then URL=$1; else DEST=$1; fi; shift ;;
  esac
done

probe() {   # probe <url> <floor_MBps>; prints the rate; fails on any bad range
  local url=$1 floor=$2 par=8 chunk=$((256 * 1024 * 1024)) d i lo hi
  local -a auth=()
  [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
  d=$(mktemp -d)
  local t0 t1
  t0=$(date +%s.%N)
  for i in $(seq 0 $((par - 1))); do
    lo=$((i * chunk)); hi=$((lo + chunk - 1))
    ( curl -sSL --fail --max-time 300 "${auth[@]}" -r "${lo}-${hi}" \
          -o "$d/$i.bin" -w '%{http_code} %{size_download}' "$url" \
          > "$d/$i.meta" 2> "$d/$i.err"; echo $? > "$d/$i.rc" ) &
  done
  wait
  t1=$(date +%s.%N)
  local total=0 bad=0 rc code size ondisk
  for i in $(seq 0 $((par - 1))); do
    rc=$(cat "$d/$i.rc" 2>/dev/null || echo 99)
    # curl -w writes no trailing newline, so `read` returns 1 at EOF while
    # still assigning: never treat that return as a failure.
    code=none; size=0
    read -r code size < "$d/$i.meta" 2>/dev/null || true
    [ -n "$code" ] || code=none; [ -n "$size" ] || size=0
    ondisk=$(stat -c %s "$d/$i.bin" 2>/dev/null || echo 0)
    if [ "$rc" != 0 ] || [ "$code" != 206 ] || [ "$size" != "$chunk" ] || [ "$ondisk" != "$chunk" ]; then
      bad=$((bad + 1))
      echo "probe: range $i FAILED (curl rc=$rc http=$code bytes=$size on disk=$ondisk) $(head -c 200 "$d/$i.err" 2>/dev/null)" >&2
    fi
    total=$((total + ondisk))
  done
  rm -rf "$d"
  local mbps
  mbps=$(python3 -c "import sys; b,t0,t1=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]); print(f'{b/max(t1-t0,1e-6)/1e6:.1f}')" "$total" "$t0" "$t1")
  echo "probe: $total bytes over $par streams in $(python3 -c "import sys; print(f'{float(sys.argv[2])-float(sys.argv[1]):.1f}')" "$t0" "$t1")s = ${mbps} MB/s aggregate (floor ${floor})"
  [ "$bad" -eq 0 ] || { echo "probe: $bad of $par range requests failed" >&2; return 1; }
  python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "$mbps" "$floor" \
    || { echo "probe: ${mbps} MB/s is below the ${floor} MB/s floor" >&2; return 1; }
}

if [ -n "$PROBE_ONLY" ]; then
  [ -n "$URL" ] || { echo "usage: $0 --probe-only [--floor MBPS] <url>" >&2; exit 2; }
  probe "$URL" "$FLOOR"; exit $?
fi
[ -n "$DEST" ] || { echo "usage: $0 <dest_dir> [--floor MBPS] [--no-hash]" >&2; exit 2; }
[ -f "$PINS" ] || { echo "missing $PINS" >&2; exit 2; }
REPO=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['repo'])" "$PINS")
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$PINS")
FIRST=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['files'][0]['path'])" "$PINS")
echo "== pull: $REPO @ $REV -> $DEST"
mkdir -p "$DEST"
probe "https://huggingface.co/$REPO/resolve/$REV/$FIRST" "$FLOOR"
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
python3 - "$PINS" "$DEST" <<'PY'
import json, os, sys, time
from huggingface_hub import hf_hub_download
pins, dest = json.load(open(sys.argv[1])), sys.argv[2]
try:
    import hf_transfer  # noqa: F401
except ImportError:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    print("hf_transfer not installed: single-stream pull (pip install hf_transfer for the fast path)", flush=True)
t0 = time.time(); got = 0
for f in pins["files"]:
    p = hf_hub_download(pins["repo"], f["path"], revision=pins["revision"],
                        local_dir=dest, token=os.environ.get("HF_TOKEN"))
    size = os.path.getsize(p); got += size
    print(f"got {p} {size/1e9:.2f} GB", flush=True)
    if size != f["size"]:
        sys.exit(f"SIZE MISMATCH {f['path']}: {size} != pinned {f['size']}")
dt = time.time() - t0
print(f"pulled {got/1e9:.2f} GB in {dt/60:.1f} min = {got/dt/1e6:.0f} MB/s; every size matches the pins", flush=True)
PY
if [ "$HASH" = 1 ]; then
  echo "== sha256 against the pinned LFS object ids"
  python3 - "$PINS" "$DEST" <<'PY'
import json, sys, hashlib, os, time
pins, dest = json.load(open(sys.argv[1])), sys.argv[2]
for f in pins["files"]:
    p = os.path.join(dest, f["path"]); h = hashlib.sha256(); t0 = time.time()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 << 20), b""):
            h.update(chunk)
    ok = h.hexdigest() == f["sha256"]
    print(f"{'OK ' if ok else 'BAD'} {f['path']} ({time.time()-t0:.0f}s)", flush=True)
    if not ok:
        sys.exit(f"HASH MISMATCH {f['path']}")
print("every shard hashes to its pin")
PY
fi
echo "== pull complete: $DEST"
