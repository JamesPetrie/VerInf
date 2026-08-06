#!/usr/bin/env bash
# S5b + S6 on the rented box, in ONE rental: fetch the real model, prove it,
# verify the proof. Every step writes its own artefact under
# analysis/bench/remote_results/<host>/ so a box that dies mid-way still tells
# us where it got to.
#
# Order is chosen so the cheap things fail first:
#   0. network probe   — 2 GB, abort if the full pull would take too long
#   1. download        — 5 GGUF shards
#   2. smoke           — 4 layers, tiny context: does the routed build run, and
#                        what does it peak at? Minutes, not hours.
#   3. witness-only    — the real geometry ONCE: gives Sz for the statement AND
#                        the model_load / semantic-sweep timings
#   4. enroll          — commit the weight block, print the root
#   5. admission       — kernel campaign + the measured model stages
#   6. prove           — the run itself
#   7. verify          — the independent Rust verifier, with policy
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
PROMPT_N="${PROMPT_N:-442}"
CONT_N="${CONT_N:-558}"
MIN_MBPS="${MIN_MBPS:-40}"          # abort the rental if the pull is slower
mkdir -p "$OUT" "$MODEL_DIR"
export LIGERO_T_QUERIES=54
export HF_HUB_ENABLE_HF_TRANSFER=1
step() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }
fail() { echo "S5B-ABORT: $*"; echo "{\"aborted\":\"$*\"}" > "$OUT/campaign_results.json"; exit 1; }

step "0. network probe"
# PARALLEL, because that is what the real download does. A single stream to
# HF caps around 20 MB/s regardless of the link, so a one-stream probe
# measures HF's per-connection limit and not this box — it rejected an
# 6715 Mbit/s A100 on the first attempt. hf_transfer opens many connections;
# the probe mirrors it with 8 ranges of 256 MB and reports the aggregate.
URL="https://huggingface.co/$REPO/resolve/main/$QUANT/${BASE}-00001-of-0000${SHARDS}.gguf"
AUTH=(); [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
PAR=8
CHUNK=$((256 * 1024 * 1024))
T0=$(date +%s)
for i in $(seq 0 $((PAR - 1))); do
  LO=$((i * CHUNK)); HI=$((LO + CHUNK - 1))
  curl -sL "${AUTH[@]}" -r "${LO}-${HI}" -o /dev/null "$URL" &
done
wait
T1=$(date +%s)
DT=$(( (T1 - T0) > 0 ? (T1 - T0) : 1 ))
MBPS=$(( PAR * 256 / DT ))
echo "probe: $((PAR * 256)) MB over $PAR streams in ${DT}s = ${MBPS} MB/s aggregate"
[ "$MBPS" -lt "$MIN_MBPS" ] && fail "network ${MBPS} MB/s below the ${MIN_MBPS} MB/s floor"
echo "projected pull of ~243 GB: $(( 243000 / MBPS / 60 )) min"

step "1. download $SHARDS shards"
$PY - "$MODEL_DIR" "$REPO" "$QUANT" "$BASE" "$SHARDS" <<'PY' || fail "download failed"
import sys, os
from huggingface_hub import hf_hub_download
d, repo, quant, base, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
for i in range(1, n + 1):
    f = f"{quant}/{base}-{i:05d}-of-{n:05d}.gguf"
    p = hf_hub_download(repo, f, local_dir=d, token=os.environ.get("HF_TOKEN"))
    print("got", p, os.path.getsize(p) / 1e9, "GB", flush=True)
PY
GGUF="$MODEL_DIR/$QUANT/${BASE}-00001-of-0000${SHARDS}.gguf"
du -sh "$MODEL_DIR" | tee "$OUT/model_size.txt"

step "2. tokens + smoke (4 layers)"
$PY - "$OUT/tokens.json" "$PROMPT_N" "$CONT_N" <<'PY'
import json, sys, random
out, p, c = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = random.Random(20260806)
json.dump({"prompt": [r.randrange(202048) for _ in range(p)],
           "continuation": [r.randrange(202048) for _ in range(c)]}, open(out, "w"))
print(f"tokens.json: {p}+{c} = {p+c}")
PY
$PY demo/demo_maverick_full.py --from-gguf "$GGUF" --tokens "$OUT/tokens.json" \
    --layers 4 --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
    --witness-only 2>&1 | tee "$OUT/smoke.log" | tail -20
grep -q "peakGPU" "$OUT/smoke.log" || fail "smoke did not finish (see smoke.log)"

step "3. witness-only at the real geometry (Sz + model stages)"
$PY demo/demo_maverick_full.py --from-gguf "$GGUF" --tokens "$OUT/tokens.json" \
    --layers 48 --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
    --witness-only 2>&1 | tee "$OUT/witness.log" | tail -20
SZ=$(grep -oE "Sz=[0-9]+" "$OUT/witness.log" | head -1 | cut -d= -f2)
[ -n "$SZ" ] || fail "no Sz in witness.log"
echo "$SZ" > "$OUT/public_sz.txt"
COLD=$(grep -oE "witness pass [0-9.]+s" "$OUT/witness.log" | head -1 | grep -oE "[0-9.]+")
[ -n "$COLD" ] || fail "no witness pass time in witness.log"
echo "public Sz = $SZ, cold sweep ${COLD}s"

step "4. enroll"
$PY demo/demo_maverick_full.py --from-gguf "$GGUF" --tokens "$OUT/tokens.json" \
    --layers 48 --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
    --enroll-weights "$MODEL_DIR/maverick.wcommit" 2>&1 | tee "$OUT/enroll.log" | tail -10
ROOT=$(grep -oE "root=[0-9a-f]{64}" "$OUT/enroll.log" | head -1 | cut -d= -f2)
[ -n "$ROOT" ] || fail "no enrolled root in enroll.log"
echo "$ROOT" > "$OUT/weight_root.txt"

step "5. admission report"
$PY analysis/bench/make_admission_report.py --from-gguf "$GGUF" \
    --tokens "$OUT/tokens.json" --layers 48 --experts 128 --d 5120 --d-ff 8192 \
    --vocab 202048 --weight-commitment "$MODEL_DIR/maverick.wcommit" \
    --public-sz "$SZ" --cold-sweep-s "$COLD" --egress-dir "$MODEL_DIR" \
    --out "$OUT/admission.json" 2>&1 | tee "$OUT/admission.log" | tail -25
grep -q "NOT ADMISSIBLE" "$OUT/admission.log" && fail "admission over cap — not starting the proof"

step "6. prove"
$PY demo/demo_maverick_full.py --from-gguf "$GGUF" --tokens "$OUT/tokens.json" \
    --layers 48 --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
    --weight-commitment "$MODEL_DIR/maverick.wcommit" --expected-weight-root "$ROOT" \
    --admission-report "$OUT/admission.json" --public-sz "$SZ" \
    --dump-proof "$MODEL_DIR/maverick-proof.json" 2>&1 | tee "$OUT/prove.log" | tail -30
STMT=$(grep -oE "statement_digest=[0-9a-f]{64}" "$OUT/prove.log" | head -1 | cut -d= -f2)
[ -n "$STMT" ] || fail "no statement_digest in prove.log"
echo "$STMT" > "$OUT/statement_digest.txt"
ls -l "$MODEL_DIR/maverick-proof.json" | tee -a "$OUT/prove.log"

step "7. verify"
./verifier/target/release/verify_proof "$MODEL_DIR/maverick-proof.json" \
    "$ROOT" "$STMT" 2>&1 | tee "$OUT/verify.log" | tail -20

$PY - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
v = (out / "verify.log").read_text() if (out / "verify.log").exists() else ""
json.dump({"accepted": "rust_verify: ACCEPT" in v,
           "sz": (out / "public_sz.txt").read_text().strip() if (out / "public_sz.txt").exists() else None,
           "root": (out / "weight_root.txt").read_text().strip() if (out / "weight_root.txt").exists() else None},
          open(out / "campaign_results.json", "w"))
print("campaign_results.json written")
PY
