#!/usr/bin/env bash
# WC-LCRL-STC over the REAL full Maverick GGUF, one rental:
#   0. parallel network probe (same as s5b — single-stream lies about HF);
#   1. download all 5 UD-Q4_K_XL shards (~243 GB, hf_transfer);
#   2. SHAKEDOWN: wc_maverick on 1 layer / 2 experts of the real file —
#      exercises the fused kquant GPU path before hours are spent;
#   3. full run: all layers, all experts, lm-head included;
#   4. campaign_results.json last, so the poller tears down only when done.
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
MIN_MBPS="${MIN_MBPS:-40}"
mkdir -p "$OUT" "$MODEL_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
step() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }
fail() { echo "WCMAV-ABORT: $*"
  tail -n 200 /workspace/suite.log > "$OUT/abort_tail.txt" 2>/dev/null || true
  echo "{\"aborted\":\"$*\"}" > "$OUT/campaign_results.json"; exit 1; }

step "0. network probe"
URL="https://huggingface.co/$REPO/resolve/main/$QUANT/${BASE}-00001-of-0000${SHARDS}.gguf"
AUTH=(); [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
PAR=8; CHUNK=$((256 * 1024 * 1024)); T0=$(date +%s)
for i in $(seq 0 $((PAR - 1))); do
  LO=$((i * CHUNK)); HI=$((LO + CHUNK - 1))
  curl -sL "${AUTH[@]}" -r "${LO}-${HI}" -o /dev/null "$URL" &
done
wait
DT=$(( ($(date +%s) - T0) > 0 ? ($(date +%s) - T0) : 1 ))
MBPS=$(( PAR * 256 / DT ))
echo "probe: ${MBPS} MB/s aggregate; projected ~243 GB pull: $(( 243000 / MBPS / 60 )) min"
[ "$MBPS" -lt "$MIN_MBPS" ] && fail "network ${MBPS} MB/s below ${MIN_MBPS} MB/s"

step "1. download $SHARDS shards"
$PY - "$MODEL_DIR" "$REPO" "$QUANT" "$BASE" "$SHARDS" <<'PY' || fail "download failed"
import sys, os
from huggingface_hub import hf_hub_download
d, repo, quant, base, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
import shutil
for i in range(1, n + 1):
    f = f"{quant}/{base}-{i:05d}-of-{n:05d}.gguf"
    p = hf_hub_download(repo, f, local_dir=d, token=os.environ.get("HF_TOKEN"))
    # the xet download path keeps a chunk cache the size of the shard —
    # doubling disk use killed the 400GB box at shard 4; drop it per shard
    shutil.rmtree(os.path.join(d, ".cache"), ignore_errors=True)
    print("got", p, os.path.getsize(p) / 1e9, "GB", flush=True)
PY
GGUF="$MODEL_DIR/$QUANT/${BASE}-00001-of-0000${SHARDS}.gguf"
du -sh "$MODEL_DIR" | tee "$OUT/model_size.txt"

step "2. shakedown: 1 layer, 2 experts (real Q4_K path)"
$PY analysis/bench/wc_maverick.py --gguf "$GGUF" --layers 1 --experts 2 \
    --json "$OUT/wcmav_shakedown.json" || fail "shakedown failed"
grep -q '"accept": true' "$OUT/wcmav_shakedown.json" || fail "shakedown REJECT"

step "3. FULL maverick: all layers, all experts, lm-head"
$PY analysis/bench/wc_maverick.py --gguf "$GGUF" --layers -1 --experts -1 \
    --lm-head --json "$OUT/wcmav_full.json" || fail "full run failed"

$PY - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
doc = {"kind": "wc_maverick",
       "shakedown": json.loads((out / "wcmav_shakedown.json").read_text()),
       "full": json.loads((out / "wcmav_full.json").read_text())}
(out / "campaign_results.json").write_text(json.dumps(doc, indent=2))
print("wrote", out / "campaign_results.json")
PY
echo DONE
