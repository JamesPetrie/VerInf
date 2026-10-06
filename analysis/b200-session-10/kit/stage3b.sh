# Session 10, stage 3b (the budget cut): stage 3's bash was stopped once its
# S=1000 prove started, so the S=1000 long-position PROOF does not run; the
# long-position proof is made at S=100 instead (the S=1000 long positions are
# covered by their exact check). Waits for the S=1000 prove to exit.
set -u
source /workspace/session10.env
echo "STAGE3B START $(date -u +%H:%M:%S)"
while pgrep -f "demo_k2.py --mode prove .*--prompt-n 500" > /dev/null; do sleep 20; done
echo "prove-s1000 finished $(date -u +%H:%M:%S): $(grep -E 'RESULT' $VERINF_LOGS/prove-s1000.log | sed 's/^\[demo_k2 [^]]*\] //')"
summ() { grep -E 'RESULT|fidelity:|routing|PROVE|rust verify|negative|enrolled|bridge enrollment|STOPPED' $VERINF_LOGS/$1.log | sed 's/^\[demo_k2 [^]]*\] /    /'; }
summ prove-s1000
logrun prove-s100-long python3 demo/demo_k2.py --mode prove --from-gguf "$VERINF_GGUF" \
  --bridge --weight-cache --routed-cache --sweep-timing --prompt-n 50 --cont-n 50 --offset 130072 \
  --record $VERINF_REC/k2-prove-s100-long.json > /dev/null 2>&1
echo "prove-s100-long rc=$? $(date -u +%H:%M:%S)"; summ prove-s100-long
sha256sum $VERINF_PROOFS/*.bin > $VERINF_REC/session10-proofs.sha256 2>/dev/null
ls -la $VERINF_PROOFS >> $VERINF_REC/session10-proofs.sha256
echo "STAGE3B PROOFS DONE $(date -u +%H:%M:%S)"
logrun fidelity-s1000 python3 demo/demo_k2.py --mode fidelity --from-gguf "$VERINF_GGUF" \
  --prompt-n 500 --cont-n 500 --record $VERINF_REC/k2-fidelity-s1000.json > /dev/null 2>&1
echo "fidelity-s1000 rc=$? $(date -u +%H:%M:%S)"; summ fidelity-s1000
echo "STAGE3B DONE $(date -u +%H:%M:%S)"
