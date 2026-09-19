#!/usr/bin/env bash
# WC-LCRL-STC FINAL RUN: the full 400B Maverick proof through the bridge.
# The s5b chain with the bridge switched on: no weight commitment, no online
# W fold — the streaming enrollment authenticates the model.
#   0. network probe   1. download 5 shards   2. smoke + BRIDGED SHAKEDOWN
#   3. witness-only (Sz + cold sweep)   4. admission report (--wc-bridge)
#   5. PROVE --wc-bridge (the timed run)   6. Rust verify   7. results
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
MIN_MBPS="${MIN_MBPS:-40}"
mkdir -p "$OUT" "$MODEL_DIR"
export LIGERO_T_QUERIES=54          # production geometry (admission target)
export HF_HUB_ENABLE_HF_TRANSFER=1
step() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }
fail() { echo "WCFINAL-ABORT: $*"
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
echo "probe: ${MBPS} MB/s; projected ~243 GB: $(( 243000 / MBPS / 60 )) min"
[ "$MBPS" -lt "$MIN_MBPS" ] && fail "network ${MBPS} MB/s below ${MIN_MBPS}"

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

step "2. tokens + bridged shakedown (2 layers, 4 experts, full chain + Rust)"
$PY - "$OUT/tokens.json" "$PROMPT_N" "$CONT_N" <<'PY'
import json, sys, random
out, p, c = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = random.Random(20260806)
json.dump({"prompt": [r.randrange(202048) for _ in range(p)],
           "continuation": [r.randrange(202048) for _ in range(c)]}, open(out, "w"))
print(f"tokens.json: {p}+{c} = {p+c}")
PY
$PY - "$GGUF" <<'PY' 2>&1 | tee "$OUT/shakedown.log" | tail -8 || fail "shakedown crashed"
import sys, time, random, torch
sys.path.insert(0, 'prover'); sys.path.insert(0, 'demo'); sys.path.insert(0, 'prover/tests')
import demo_maverick_full as drv
drv.WC_BRIDGE = True
from tape import Tape
import wc_bridge as wcb
from tests._rust_verify import rust_verify_tape
GGUF = sys.argv[1]
r = random.Random(0)
tape = Tape(drv.CFG, silu_config=drv.SILU_CFG, lazy=True)
torch.manual_seed(7)
logits, Sz, handles, sum_pos = drv.build_model(
    tape, GGUF, [r.randrange(202048) for _ in range(8)],
    [r.randrange(202048) for _ in range(8)],
    V=202048, d=5120, n_layers=2, E=4, d_ff=8192)
live = tape.run_engine_pass(free_intermediates=True, keep={Sz.var})
handles["reveal_pin"].public_rhs = int(live[Sz.var].cpu()[0])
del live; torch.cuda.empty_cache()
enr = wcb.lazy_enroll_tape(tape, b"wc-maverick-mask-v1",
                           f"maverick|{GGUF}|S=4096".encode(), wcb.WcParams())
print(f"shakedown enrollment root={enr.root.hex()[:16]}", flush=True)
t0 = time.time()
proof = tape.prove(weight_enrollment=enr)
print(f"shakedown prove {time.time()-t0:.1f}s", flush=True)
acc, msg = rust_verify_tape(tape, proof, seed=None)
print(f"shakedown ACCEPT={acc} {'' if acc else msg}", flush=True)
assert acc
PY
grep -q "shakedown ACCEPT=True" "$OUT/shakedown.log" || fail "bridged shakedown REJECT"

step "3. witness-only at the real geometry (Sz + cold sweep)"
$PY demo/demo_maverick_full.py --from-gguf "$GGUF" --tokens "$OUT/tokens.json" \
    --layers 48 --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
    --wc-bridge --witness-only 2>&1 | tee "$OUT/witness.log" | tail -12
SZ=$(grep -oE "Sz=[0-9]+" "$OUT/witness.log" | head -1 | cut -d= -f2)
[ -n "$SZ" ] || fail "no Sz in witness.log"
COLD=$(grep -oE "witness pass [0-9.]+s" "$OUT/witness.log" | head -1 | grep -oE "[0-9.]+")
[ -n "$COLD" ] || fail "no witness pass time"
echo "public Sz = $SZ, cold sweep ${COLD}s"; echo "$SZ" > "$OUT/public_sz.txt"

step "4. admission report (wc-bridge)"
$PY analysis/bench/make_admission_report.py --from-gguf "$GGUF" \
    --tokens "$OUT/tokens.json" --layers 48 --experts 128 --d 5120 --d-ff 8192 \
    --vocab 202048 --wc-bridge \
    --public-sz "$SZ" --cold-sweep-s "$COLD" --egress-dir "$MODEL_DIR" \
    --out "$OUT/admission.json" 2>&1 | tee "$OUT/admission.log" | tail -25
grep -q "NOT ADMISSIBLE" "$OUT/admission.log" && fail "admission over cap"

step "5. PROVE (wc-bridge) — the timed run"
$PY demo/demo_maverick_full.py --from-gguf "$GGUF" --tokens "$OUT/tokens.json" \
    --layers 48 --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
    --wc-bridge --admission-report "$OUT/admission.json" --public-sz "$SZ" \
    --dump-proof "$MODEL_DIR/maverick-wc-proof.json" 2>&1 | tee "$OUT/prove.log" | tail -30
ROOT=$(grep -oE "enrollment_root=[0-9a-f]{64}" "$OUT/prove.log" | head -1 | cut -d= -f2)
STMT=$(grep -oE "statement_digest=[0-9a-f]{64}" "$OUT/prove.log" | head -1 | cut -d= -f2)
TPROVE=$(grep -oE "prove returned \([0-9.]+s\)" "$OUT/prove.log" | grep -oE "[0-9.]+")
[ -n "$ROOT" ] || fail "no enrollment root in prove.log"
[ -n "$STMT" ] || fail "no statement_digest in prove.log"
echo "$ROOT" > "$OUT/enrollment_root.txt"; echo "$STMT" > "$OUT/statement_digest.txt"
echo "PROVE WALL TIME: ${TPROVE}s"

step "6. Rust verify (external root = enrollment root)"
./verifier/target/release/verify_proof "$MODEL_DIR/maverick-wc-proof.json" \
    "$ROOT" "$STMT" 2>&1 | tee "$OUT/verify.log" | tail -20

$PY - "$OUT" "$TPROVE" <<'PY'
import json, pathlib, sys
out, tprove = pathlib.Path(sys.argv[1]), float(sys.argv[2] or 0)
v = (out / "verify.log").read_text() if (out / "verify.log").exists() else ""
json.dump({"kind": "wc_final", "accepted": "rust_verify: ACCEPT" in v,
           "t_prove_s": tprove, "under_one_hour": bool(tprove and tprove < 3600),
           "sz": (out / "public_sz.txt").read_text().strip(),
           "enrollment_root": (out / "enrollment_root.txt").read_text().strip()},
          open(out / "campaign_results.json", "w"), indent=1)
print("campaign_results.json written")
PY
echo DONE
