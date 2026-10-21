# Shard 1 of Kimi K2 UD-Q4_K_XL at the pinned revision, size- and sha256-checked.
set -u
source /workspace/session10.env
t0=$(date +%s)
HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<'PY' || { echo "EXIT=1 (download)"; exit 1; }
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["K2_REPO"], os.environ["K2_SHARD1"], revision=os.environ["K2_REV"],
                    local_dir=os.environ["VERINF_GGUF_DIR"])
print("downloaded", p)
PY
t1=$(date +%s)
size=$(stat -c %s "$VERINF_GGUF")
echo "pull: $size bytes in $((t1 - t0)) s = $(( size / (t1 - t0 + 1) / 1000000 )) MB/s"
[ "$size" = "$K2_SHARD1_BYTES" ] || { echo "EXIT=1 (size $size != $K2_SHARD1_BYTES)"; exit 1; }
got=$(sha256sum "$VERINF_GGUF" | cut -d' ' -f1)
echo "sha256 $got ($(( $(date +%s) - t1 )) s)"
[ "$got" = "$K2_SHARD1_SHA256" ] || { echo "EXIT=1 (sha256 mismatch)"; exit 1; }
echo "EXIT=0"
