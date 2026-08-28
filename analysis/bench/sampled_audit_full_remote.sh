#!/usr/bin/env bash
# Clean-box Vast chain for the real 48-layer Maverick sampled audit:
# download the five GGUF shards, enroll the resident model, then time exactly
# one 2,596-claim sampled-audit engine pass. Download and enrollment are outside
# the reported audit wall time.
set -u
cd "${VERINF_ROOT:-/workspace/VerInf}"
PY="uv run --project $PWD python3"
HOST="$(hostname)"
OUT="analysis/bench/remote_results/$HOST"
MODEL_DIR="${MODEL_DIR:-/workspace/gguf}"
REPO="unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF"
QUANT="UD-Q4_K_XL"
BASE="Llama-4-Maverick-17B-128E-Instruct-${QUANT}"
SHARDS=5
PROMPT_N=442
CONT_N=558
MIN_MBPS="${MIN_MBPS:-120}"
# Same fixed tokens/model as the accepted 2026-08-06 real run. Reusing its
# public serving statement removes an otherwise unnecessary pre-audit forward.
PUBLIC_SZ="${PUBLIC_SZ:-32865524}"
mkdir -p "$OUT" "$MODEL_DIR"
export LIGERO_T_QUERIES=54
# Public model artifacts only. Do not forward a workstation credential into
# a rented instance; both the probe and aria2 use anonymous access.
unset HF_TOKEN
step() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }
fail() {
  echo "SAMPLED-AUDIT-ABORT: $*"
  printf '{"accepted":false,"aborted":"%s"}\n' "$*" > "$OUT/campaign_results.json"
  exit 1
}

step "0. parallel network probe"
URL="https://huggingface.co/$REPO/resolve/main/$QUANT/${BASE}-00001-of-0000${SHARDS}.gguf"
# Keep the probe at 2 GiB, but use 32 ranges to approximate aria2's 80-stream
# x16/j5 downloader instead of measuring the CDN's per-connection cap.
PAR=32
CHUNK_MIB=64
CHUNK=$((CHUNK_MIB * 1024 * 1024))
T0=$(date +%s)
for i in $(seq 0 $((PAR - 1))); do
  LO=$((i * CHUNK)); HI=$((LO + CHUNK - 1))
  curl -sL -r "${LO}-${HI}" -o /dev/null "$URL" &
done
wait
DT=$(( $(date +%s) - T0 )); [ "$DT" -gt 0 ] || DT=1
MBPS=$(( PAR * CHUNK_MIB / DT ))
[ "$MBPS" -gt 0 ] || fail "network probe transferred below 1 MB/s"
echo "probe: ${MBPS} MB/s aggregate; projected pull $((243000 / MBPS / 60)) min"
[ "$MBPS" -ge "$MIN_MBPS" ] || fail "network ${MBPS} MB/s below ${MIN_MBPS} MB/s"

step "1. download five Maverick GGUF shards (existing aria2 x16/j5 path)"
# Reuse the downloader proven by optrun_mav.sh: 16 ranges for each of five
# shards. A single hf_hub_download stream measured only 49 MB/s on this host.
apt-get install -y -q aria2 >/dev/null 2>&1 || fail "aria2 install failed"
GGUF_DIR="$MODEL_DIR/$QUANT"
mkdir -p "$GGUF_DIR"
: > /workspace/sampled-audit-download.txt
for n in 00001 00002 00003 00004 00005; do
  file="${BASE}-${n}-of-0000${SHARDS}.gguf"
  printf '%s\n  out=%s\n  dir=%s\n' \
    "https://huggingface.co/$REPO/resolve/main/$QUANT/$file" \
    "$file" "$GGUF_DIR" >> /workspace/sampled-audit-download.txt
done
dl_t0=$SECONDS
aria2c -i /workspace/sampled-audit-download.txt -x16 -s16 -j5 -k 25M \
  --file-allocation=none --console-log-level=warn --summary-interval=15 \
  --auto-file-renaming=false --allow-overwrite=true --max-tries=8 \
  --retry-wait=5 || fail "aria2 download failed"
echo "aria2 download took $((SECONDS - dl_t0))s"
$PY - "$GGUF_DIR" "$BASE" "$SHARDS" <<'PY' || fail "shard size validation failed"
import os, sys
d, base, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
expected = [49807659968, 49830263808, 49832905856, 49508645760, 33155199296]
for i, want in enumerate(expected, 1):
    path = os.path.join(d, f"{base}-{i:05d}-of-{n:05d}.gguf")
    got = os.path.getsize(path)
    if got != want:
        raise RuntimeError(f"shard {i}: {got} bytes, expected {want}")
    print("validated", path, got, flush=True)
PY
GGUF="$MODEL_DIR/$QUANT/${BASE}-00001-of-0000${SHARDS}.gguf"
test -s "$GGUF" || fail "first GGUF shard missing after download"
du -sh "$MODEL_DIR" | tee "$OUT/model_size.txt"

step "2. recreate the fixed 442+558 token statement"
$PY - "$OUT/tokens.json" "$PROMPT_N" "$CONT_N" <<'PY'
import json
import random
import sys

out, p, c = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = random.Random(20260806)
json.dump({"prompt": [rng.randrange(202048) for _ in range(p)],
           "continuation": [rng.randrange(202048) for _ in range(c)]}, open(out, "w"))
PY

step "3. one-time resident-model enrollment (outside audit timing)"
$PY demo/demo_maverick_full.py --from-gguf "$GGUF" --tokens "$OUT/tokens.json" \
  --layers 48 --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
  --enroll-weights "$MODEL_DIR/maverick.wcommit" \
  2>&1 | tee "$OUT/enroll.log" || fail "model enrollment failed"
ROOT=$(grep -oE 'root=[0-9a-f]{64}' "$OUT/enroll.log" | tail -1 | cut -d= -f2)
[ -n "$ROOT" ] || fail "enrollment produced no root"
echo "$ROOT" > "$OUT/weight_root.txt"

step "4. verifier entropy"
umask 077
SECRET=/workspace/sampled-audit-verifier.secret
head -c 32 /dev/urandom > "$SECRET"

step "5. timed real 2,596-claim sampled audit"
TOKENS_JSON="$OUT/tokens.json" \
WEIGHT_COMMITMENT="$MODEL_DIR/maverick.wcommit" \
EXPECTED_WEIGHT_ROOT="$ROOT" \
PUBLIC_SZ="$PUBLIC_SZ" \
VERIFIER_SECRET_FILE="$SECRET" \
SAMPLED_AUDIT_OUT="$OUT" \
MODEL_DIR="$MODEL_DIR" \
bash analysis/bench/sampled_audit_vast.sh || fail "sampled audit rejected or exceeded cap"

step "6. publish campaign result"
$PY - "$OUT" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
audit = json.loads((out / "stage_times.json").read_text())
result = {"kind": "real-maverick-sampled-audit", **audit}
(out / "campaign_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: result[k] for k in
                  ("accepted", "claims", "selected", "fraction", "wall_s", "c0_root")},
                 indent=2))
PY
echo DONE
