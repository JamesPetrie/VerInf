#!/usr/bin/env bash
# FULL Maverick 400B run: download UD-Q4_K_XL GGUF (public, no token) then
# demo_maverick_full at OPTIMIZED rho=2 (N_LIG=32768) / T=17 + DISK-spill.
# MAV_LAYERS: 2 = de-risk (real weights, few layers), 48 = full 400B.
set -x
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
RESDIR=/workspace/VerInf/analysis/bench/remote_results/optrun
GGUF=/workspace/maverick-gguf/UD-Q4_K_XL
SPILL=/workspace/spill
mkdir -p "$RESDIR" "$GGUF" "$SPILL"
PYRUN="uv run --project /workspace/VerInf python3"
export VERINF_PYRUN="$PYRUN"
# GATE: validate machine vs offer before the (expensive) 400B download+prove.
export PF_DISK_DIR="$SPILL" PF_MIN_READ_GBPS="${MAV_MIN_READ_GBPS:-2.0}"
source analysis/bench/preflight_gate.sh
uv add gguf 2>&1 | tail -2          # gguf python reader (prover/loader.py needs it)
LAYERS=${MAV_LAYERS:-2}
BASE="https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF/resolve/main/UD-Q4_K_XL"
df -h /workspace

echo "########## DOWNLOAD UD-Q4_K_XL (5 shards ~250GB, aria2c -x16 -s16) ##########"
# Single TCP stream caps ~20-37 MB/s (long-haul window); aria2c splits each shard
# into 16 range-requests -> saturates the host uplink. resolve URL 302->xet CDN.
apt-get install -y aria2 >/dev/null 2>&1 || true
t_dl=$SECONDS
: > /workspace/dl.txt
for n in 00001 00002 00003 00004 00005; do
  f="Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL-${n}-of-00005.gguf"
  printf '%s\n  out=%s\n  dir=%s\n' "$BASE/$f" "$f" "$GGUF" >> /workspace/dl.txt
done
aria2c -i /workspace/dl.txt -x16 -s16 -j5 -k 25M --file-allocation=none \
  --console-log-level=warn --summary-interval=15 --auto-file-renaming=false \
  --allow-overwrite=true --max-tries=8 --retry-wait=5 || { echo "DOWNLOAD FAILED (aria2c)"; exit 3; }
cnt=$(find "$GGUF" -name '*.gguf' -size +1G | wc -l)
[ "$cnt" -eq 5 ] || { echo "DOWNLOAD FAILED (got $cnt/5 shards >1G)"; ls -la "$GGUF"; exit 3; }
echo "download took $((SECONDS-t_dl))s"; ls -la "$GGUF"; df -h /workspace

echo "########## PROVE demo_maverick_full layers=$LAYERS  rho=2 T=17 + disk-spill ##########"
LIGERO_N_LIG=32768 LIGERO_T_QUERIES=17 \
  LIGERO_WITNESS_SPILL_DISK=1 LIGERO_WITNESS_SPILL_DIR="$SPILL" LIGERO_STREAM_DBG=1 \
  $PYRUN demo/demo_maverick_full.py --from-gguf "$GGUF" \
    --layers "$LAYERS" --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
    --prompt-n 260 --cont-n 240 --dump-proof /workspace/mav_proof.json
PV=$?
echo "prove exit=$PV"

echo "########## Rust verify_proof (build then run) ##########"
VR=1
VBIN=/workspace/VerInf/verifier/target/release/verify_proof
if [ -s /workspace/mav_proof.json ]; then
  [ -x "$VBIN" ] || ( cd /workspace/VerInf/verifier && cargo build --release --bin verify_proof 2>&1 | tail -4 )
  "$VBIN" /workspace/mav_proof.json && VR=0 || VR=1
  echo "verify exit=$VR"
fi
printf '{"layers": %s, "prove_exit": %d, "verify_exit": %d}\n' "$LAYERS" "$PV" "$VR" > "$RESDIR/campaign_results.json"
ls -la /workspace/mav_proof.json 2>/dev/null
echo "########## MAV DONE (layers=$LAYERS prove=$PV verify=$VR) ##########"
