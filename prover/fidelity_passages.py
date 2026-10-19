"""The frozen passages of analysis/k2-fidelity-study-design.md §6.4 and §6.6:
each user turn rendered with the pinned GGUF's chat template by llama.cpp
(tools/llama_runner/verinf_render) and tokenized as a deployment would, its
source hash verified, its length checked, and the token ids frozen.

    python prover/fidelity_passages.py --prompts analysis/k2-fidelity-prompts-2026-10-19.json \\
        --model K2-METADATA.gguf --render .../verinf_render --work DIR \\
        [--freeze analysis/k2-fidelity-passages-frozen.json] [--accept-lengths]

Without --freeze it only reports. A freeze never overwrites an existing record,
and refuses lengths outside the target unless --accept-lengths records that
the authors accepted them before any measurement. --verify FROZEN renders
again (on the pod, from the full, sha256-verified shard) and exits nonzero
unless every passage's ids equal the frozen ones."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

TARGET = (400, 500)              # prompt tokens (§6.4)
CONTINUATION = 500
TOTAL_CAP = 1000


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ids_sha256(ids) -> str:
    return sha256(np.asarray(ids, dtype="<i4").tobytes())


def render_all(prompts_path, model, render_bin, work):
    doc = json.loads(pathlib.Path(prompts_path).read_text())
    rows = []
    for p in doc["passages"]:
        msgs = p["messages"]
        if len(msgs) != 1 or msgs[0]["role"] != "user":
            raise ValueError(f"{p['id']}: one user message expected")
        content = msgs[0]["content"]
        if sha256(content.encode()) != p["user_content_sha256"]:
            raise ValueError(f"{p['id']}: user content does not match its recorded sha256")
        out = pathlib.Path(work) / p["id"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "user.txt").write_text(content)
        subprocess.run([str(render_bin), "--model", str(model), "--content", str(out / "user.txt"),
                        "--out", str(out)], check=True)
        rj = json.loads((out / "render.json").read_text())
        ids = [int(x) for x in (out / "ids.txt").read_text().split()]
        prompt = (out / "prompt.txt").read_bytes()
        rows.append(dict(id=p["id"], role=p["role"], user_content_sha256=p["user_content_sha256"],
                         prompt_sha256=sha256(prompt), n_prompt_tokens=len(ids),
                         ids_sha256=ids_sha256(ids), ids=ids, add_bos=rj["add_bos"],
                         content_has_special_tokens=rj["content_has_special_tokens"],
                         template_sha256=sha256(rj["template"].encode()),
                         llama_cpp_commit=rj["llama_cpp_commit"]))
    return doc, rows


def problems(rows):
    out = []
    for r in rows:
        n = r["n_prompt_tokens"]
        if not TARGET[0] <= n <= TARGET[1]:
            out.append(f"{r['id']}: {n} prompt tokens, outside the {TARGET[0]}-{TARGET[1]} target")
        if n + CONTINUATION > TOTAL_CAP:
            out.append(f"{r['id']}: {n} + {CONTINUATION} exceeds the {TOTAL_CAP}-token cap")
        if r["content_has_special_tokens"]:
            out.append(f"{r['id']}: the user text holds special-token text")
    if len({r["template_sha256"] for r in rows}) != 1:
        out.append("the passages were rendered with different templates")
    return out


def freeze(prompts_path, rows, dst, *, model_provenance: dict, accept_lengths=False):
    dst = pathlib.Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} exists: a freeze is never overwritten")
    found = problems(rows)
    hard = [p for p in found if "target" not in p]
    if hard or (found and not accept_lengths):
        raise ValueError("not frozen: " + "; ".join(found))
    rec = dict(frozen=True, prompts_file=str(prompts_path),
               prompts_sha256=sha256(pathlib.Path(prompts_path).read_bytes()),
               model_provenance=model_provenance, lengths_accepted_outside_target=bool(found),
               notes=found, passages=rows)
    dst.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def verify(rows, frozen_path):
    """[(id, frozen ids sha256, rendered ids sha256)] for every passage that
    differs from the frozen record, or is missing from either."""
    frozen = {p["id"]: p["ids_sha256"] for p in json.loads(pathlib.Path(frozen_path).read_text())["passages"]}
    got = {r["id"]: r["ids_sha256"] for r in rows}
    return [(i, frozen.get(i), got.get(i)) for i in sorted(set(frozen) | set(got))
            if frozen.get(i) != got.get(i)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", required=True, help="the pinned GGUF, or its metadata-only copy")
    ap.add_argument("--model-provenance", default="{}", help="JSON: where the model's bytes came from")
    ap.add_argument("--render", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--freeze")
    ap.add_argument("--accept-lengths", action="store_true")
    ap.add_argument("--verify", help="a frozen record the rendering must equal")
    a = ap.parse_args(argv)
    _, rows = render_all(a.prompts, a.model, a.render, a.work)
    for r in rows:
        print(f"{r['id']:26s} {r['role']:10s} {r['n_prompt_tokens']:4d} tokens  ids {r['ids_sha256'][:16]}")
    for p in problems(rows):
        print("NOTE", p)
    if a.verify:
        bad = verify(rows, a.verify)
        for i, want, got in bad:
            print(f"MISMATCH {i}: frozen {want} rendered {got}")
        print("VERIFY", "PASS" if not bad else "FAIL")
        if bad:
            return 1
    if a.freeze:
        freeze(a.prompts, rows, a.freeze, model_provenance=json.loads(a.model_provenance),
               accept_lengths=a.accept_lengths)
        print("frozen:", a.freeze)
    return 0


if __name__ == "__main__":
    sys.exit(main())
