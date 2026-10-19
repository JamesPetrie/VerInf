"""The engine arm's Python side (analysis/k2-fidelity-study-design.md §3.3,
§5.1): run tools/llama_runner/verinf_runner on token ids and read back its
logits, captured layer tensors and manifest, adding what the C++ side does
not know (the token-ids hash, the model files' sizes, and, when asked, their
sha256). The loader-branch fingerprint is llama.cpp's own print_info lines.

No model knowledge here: which tensors to capture, which YaRN betas the
model's definition needs and how names map to ours are the adapters'."""
import hashlib
import json
import pathlib
import re
import subprocess

import numpy as np


def ids_sha256(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def file_sha256(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def run(binary, model, out, *, ids=None, mode="replay", n_prompt=None, n_gen=0,
        prompt_chunk=512, cont_chunk=1, pos_offset=0, capture=(), settings=None,
        hash_model=False, timeout=None):
    """Run the runner; returns load(out). `settings` maps runner flags without
    their dashes (n_batch, n_ubatch, threads, n_gpu_layers, kv_type,
    flash_attn, yarn_beta_fast, yarn_beta_slow) to values."""
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "--model", str(model), "--out", str(out), "--mode", mode]
    if mode != "info":
        (out / "input_ids.txt").write_text("\n".join(str(int(i)) for i in ids) + "\n")
        cmd += ["--tokens", str(out / "input_ids.txt"), "--prompt-chunk", str(prompt_chunk),
                "--cont-chunk", str(cont_chunk), "--pos-offset", str(pos_offset)]
        if n_prompt is not None:
            cmd += ["--n-prompt", str(n_prompt)]
        if mode == "generate":
            cmd += ["--n-gen", str(n_gen)]
        if capture:
            cmd += ["--capture", ",".join(capture)]
    for k, v in (settings or {}).items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    (out / "runner.stderr").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"verinf_runner exited {proc.returncode}: {proc.stderr[-2000:]}")
    res = load(out)
    m = res["manifest"]
    m["command"] = cmd
    if mode != "info":
        m["input_ids_sha256"] = ids_sha256(ids)
        m["sequence_ids_sha256"] = ids_sha256(res["tokens"])
    files = sorted(pathlib.Path(model).parent.glob(_shard_glob(pathlib.Path(model).name)))
    m["model_files"] = [dict(name=f.name, bytes=f.stat().st_size,
                             **({"sha256": file_sha256(f)} if hash_model else {}))
                        for f in files]
    (out / "manifest.json").write_text(json.dumps(m, indent=1))
    return res


def _shard_glob(name: str) -> str:
    """A split GGUF's sibling pattern (…-00001-of-00013.gguf), else the file."""
    m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", name)
    return f"{m.group(1)}-?????-of-{m.group(2)}.gguf" if m else name


def load(out):
    """Read a runner output directory: manifest, logits (n_scored, V), the
    full token sequence and each capture (positions, width)."""
    out = pathlib.Path(out)
    m = json.loads((out / "manifest.json").read_text())
    res = {"manifest": m, "fingerprint": fingerprint((out / "load.log").read_text())}
    if m["mode"] == "info":
        return res
    V = m["n_vocab"]
    res["logits"] = np.fromfile(out / "logits.f32", dtype="<f4").reshape(-1, V)
    res["tokens"] = [int(x) for x in (out / "tokens.txt").read_text().split()]
    res["captures"] = {
        name: np.fromfile(out / "capture" / f"{name}.f32", dtype="<f4").reshape(-1, c["width"])
        for name, c in m["captures"].items()}
    return res


# print_info lines that legitimately differ between a model and its truncation
TRUNCATION_KEYS = {"n_layer", "n_layer_all", "model type", "model params", "file size",
                   "file type", "general.name"}


def fingerprint(log: str) -> dict:
    """llama.cpp's print_info lines as {key: value}: the hyperparameters the
    loader actually used, which is what decides its branch."""
    fp = {}
    for line in log.splitlines():
        m = re.match(r"print_info: (.+?)\s+= (.*)$", line)
        if m:
            fp[m.group(1).strip()] = m.group(2).strip()
    return fp


def branch_differences(full: dict, truncated: dict, ignore=TRUNCATION_KEYS) -> dict:
    """Every print_info key whose value differs between the full model and its
    truncation, except those a truncation must change; empty means the same
    loading branch and hyperparameters."""
    keys = (set(full) | set(truncated)) - set(ignore)
    return {k: (full.get(k), truncated.get(k)) for k in sorted(keys)
            if full.get(k) != truncated.get(k)}
