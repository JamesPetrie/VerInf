# Wait for stage 3's S=1000 prove to start, stop stage 3's bash (the prove's
# python and tee continue), launch stage 3b.
until pgrep -f "demo_k2.py --mode prove .*--prompt-n 500" > /dev/null; do sleep 10; done
sleep 5
echo "switch: S=1000 prove running $(date -u +%H:%M:%S); stopping stage3.sh: $(pgrep -f 'bash /workspace/stage3.sh' | tr '\n' ' ')"
pkill -f "bash /workspace/stage3.sh"
sleep 2
echo "switch: stage3.sh left: $(pgrep -f 'bash /workspace/stage3.sh' | wc -l); prove still running: $(pgrep -f 'demo_k2.py --mode prove .*--prompt-n 500' | wc -l)"
setsid nohup bash -c "cd /workspace/verinf && bash /workspace/stage3b.sh" > ~/stage3b.log 2>&1 < /dev/null &
echo "switch: stage3b launched $(date -u +%H:%M:%S)"
