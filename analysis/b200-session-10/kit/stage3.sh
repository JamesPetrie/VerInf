# Session 10, stage 3 (after the range review): fidelity on the CPU, then the
# real bridged proofs with the measurements, then S=1000 fidelity if time.
set -u
source /workspace/session10.env
echo "STAGE3 START $(date -u +%H:%M:%S)"
summ() { grep -E 'RESULT|fidelity:|routing|PROVE|rust verify|negative|enrolled|bridge enrollment|STOPPED' $VERINF_LOGS/$1.log | sed 's/^\[demo_k2 [^]]*\] /    /'; }
for a in "s100 50 50 0" "s100-long 50 50 130072"; do set -- $a
  logrun fidelity-$1 python3 demo/demo_k2.py --mode fidelity --from-gguf "$VERINF_GGUF" \
    --prompt-n $2 --cont-n $3 --offset $4 --record $VERINF_REC/k2-fidelity-$1.json > /dev/null 2>&1
  echo "fidelity-$1 rc=$? $(date -u +%H:%M:%S)"; summ fidelity-$1
done
CACHES="--bridge --weight-cache --routed-cache --sweep-timing"
logrun prove-s100 python3 demo/demo_k2.py --mode prove --from-gguf "$VERINF_GGUF" $CACHES \
  --prompt-n 50 --cont-n 50 --negatives interleave,yarn-witness,yarn-statement,wrong-slice \
  --dump-proof $VERINF_PROOFS/k2-s100.bin --record $VERINF_REC/k2-prove-s100.json > /dev/null 2>&1
echo "prove-s100 rc=$? $(date -u +%H:%M:%S)"; summ prove-s100
logrun prove-s1000 python3 demo/demo_k2.py --mode prove --from-gguf "$VERINF_GGUF" $CACHES \
  --prompt-n 500 --cont-n 500 --dump-proof $VERINF_PROOFS/k2-s1000.bin \
  --record $VERINF_REC/k2-prove-s1000.json > /dev/null 2>&1
echo "prove-s1000 rc=$? $(date -u +%H:%M:%S)"; summ prove-s1000
logrun prove-s1000-long python3 demo/demo_k2.py --mode prove --from-gguf "$VERINF_GGUF" $CACHES \
  --prompt-n 500 --cont-n 500 --offset 130072 --record $VERINF_REC/k2-prove-s1000-long.json > /dev/null 2>&1
echo "prove-s1000-long rc=$? $(date -u +%H:%M:%S)"; summ prove-s1000-long
sha256sum $VERINF_PROOFS/*.bin > $VERINF_REC/session10-proofs.sha256 2>/dev/null
echo "STAGE3 PROOFS DONE $(date -u +%H:%M:%S)"
logrun fidelity-s1000 python3 demo/demo_k2.py --mode fidelity --from-gguf "$VERINF_GGUF" \
  --prompt-n 500 --cont-n 500 --record $VERINF_REC/k2-fidelity-s1000.json > /dev/null 2>&1
echo "fidelity-s1000 rc=$? $(date -u +%H:%M:%S)"; summ fidelity-s1000
echo "STAGE3 DONE $(date -u +%H:%M:%S)"
