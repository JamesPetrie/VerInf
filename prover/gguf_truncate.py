"""A model's first n blocks as a GGUF llama.cpp can load, for the engine arm
at two layers (analysis/k2-fidelity-study-design.md §5.2).

- Tensors are found by name across every shard present, never by assuming
  which shard holds what.
- The embedding, the final norm and the real output head are required:
  llama.cpp's DeepSeek loader falls back to the tied embedding when the head
  is absent, which would silently change an untied model.
- Every kept block must hold exactly its kind's tensors, from the
  architecture's list, so a block split across a missing shard is refused
  rather than loaded short.
- Quantized bytes are copied unchanged. Metadata is copied with the block
  count rewritten, per-layer arrays cut to n and the split keys dropped.
- llama.cpp's deepseek2 loader decides "lite" by layer count (26 or 27, or 48
  with a 128,256 vocabulary) and reads the query-LoRA rank only otherwise. A
  model without query-LoRA, such as Moonlight's 27 layers, therefore needs
  `q_lora_rank = 0` written once truncation takes it out of that branch.
  `deepseek2_branch_fix` writes it, and the branch check (llama_runner's
  fingerprints) confirms the truncation loads as the full model does."""
import pathlib
import re

import numpy as np

SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")
SKIP_KEYS = ("general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count")
GLOBALS_REQUIRED = ("token_embd.weight", "output_norm.weight", "output.weight")

# deepseek2 block tensors as llama.cpp loads them (src/models/deepseek2.cpp), split MLA form
DEEPSEEK2_ATTN = ("attn_norm.weight", "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
                  "attn_k_b.weight", "attn_v_b.weight", "attn_output.weight", "ffn_norm.weight")
DEEPSEEK2_Q_LORA = ("attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight")
DEEPSEEK2_Q_DIRECT = ("attn_q.weight",)
DEEPSEEK2_DENSE = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
DEEPSEEK2_MOE = ("ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.weight",
                 "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
                 "ffn_up_shexp.weight", "ffn_down_shexp.weight")


def shard_paths(path) -> list:
    """Every shard of a split GGUF present beside `path`, else [path]."""
    p = pathlib.Path(path)
    m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", p.name)
    if not m:
        return [p]
    return sorted(p.parent.glob(f"{m.group(1)}-?????-of-{m.group(2)}.gguf"))


def _readers(path):
    from gguf import GGUFReader
    return [GGUFReader(str(s)) for s in shard_paths(path)]


def metadata(path) -> dict:
    """The first shard's key-values, decoded."""
    r = _readers(path)[0]
    return {f.name: f.contents() for f in r.fields.values() if f.name not in SKIP_KEYS[1:]}


def deepseek2_block_names(il: int, *, dense_lead: int, q_lora: bool) -> set:
    names = DEEPSEEK2_ATTN + (DEEPSEEK2_Q_LORA if q_lora else DEEPSEEK2_Q_DIRECT)
    names += DEEPSEEK2_DENSE if il < dense_lead else DEEPSEEK2_MOE
    return {f"blk.{il}.{n}" for n in names}


def deepseek2_branch_fix(md: dict, tensors: dict) -> dict:
    """Keys to add so the truncation takes the full model's loading branch:
    q_lora_rank = 0 when the model has no query-LoRA and the key is absent."""
    arch = md["general.architecture"]
    has_q_lora = "blk.0.attn_q_a.weight" in tensors
    if not has_q_lora and f"{arch}.attention.q_lora_rank" not in md:
        return {f"{arch}.attention.q_lora_rank": 0}
    return {}


def truncate(src, dst, n_layers: int, *, block_names=None, branch_fix=deepseek2_branch_fix):
    """Write the first `n_layers` blocks of the model at `src` (one shard of a
    split GGUF is enough to name the set) to `dst`. Returns a record: the
    tensors kept and their sources, the keys rewritten, cut or added."""
    from gguf import GGUFValueType as VT
    from gguf import GGUFWriter
    readers = _readers(src)
    tensors = {}
    for r in readers:
        for t in r.tensors:
            if t.name in tensors:
                raise ValueError(f"{t.name} appears in two shards")
            tensors[t.name] = (t, pathlib.Path(r.data.filename).name if hasattr(r.data, "filename") else "")
    first = readers[0]
    md = {f.name: f for f in first.fields.values()}
    arch = md["general.architecture"].contents()
    n_full = int(md[f"{arch}.block_count"].contents())
    if not 0 < n_layers <= n_full:
        raise ValueError(f"{n_layers} layers of {n_full}")
    missing = [g for g in GLOBALS_REQUIRED if g not in tensors]
    if missing:
        raise ValueError(f"missing {missing}: the output head is required, never the embedding "
                         "in its place")
    if block_names is None:
        dense_lead = int(md[f"{arch}.leading_dense_block_count"].contents()) \
            if f"{arch}.leading_dense_block_count" in md else 0
        q_lora = "blk.0.attn_q_a.weight" in tensors
        block_names = lambda il: deepseek2_block_names(il, dense_lead=dense_lead, q_lora=q_lora)
    keep = list(GLOBALS_REQUIRED)
    for il in range(n_layers):
        want = block_names(il)
        have = {n for n in tensors if n.startswith(f"blk.{il}.")}
        if have != want:
            raise ValueError(f"block {il}: missing {sorted(want - have)}, unexpected "
                             f"{sorted(have - want)} (a shard absent, or another architecture)")
        keep += sorted(want)
    w = GGUFWriter(str(dst), arch)
    rec = {"source": [str(p) for p in shard_paths(src)], "n_layers": n_layers, "n_full": n_full,
           "dropped_keys": [], "cut_arrays": [], "added_keys": {}, "tensors": {}}
    for name, f in md.items():
        if name in SKIP_KEYS or name in SPLIT_KEYS:
            if name in SPLIT_KEYS:
                rec["dropped_keys"].append(name)
            continue
        vtype = f.types[0]
        val = f.contents()
        if name == f"{arch}.block_count":
            val = n_layers
        if vtype == VT.ARRAY:
            sub = f.types[-1]
            if name.startswith(arch + ".") and len(val) == n_full:
                val = val[:n_layers]
                rec["cut_arrays"].append(name)
            w.add_key_value(name, val, vtype, sub_type=sub)
        else:
            w.add_key_value(name, val, vtype)
    for k, v in branch_fix(
            {n: f.contents() for n, f in md.items() if n not in SKIP_KEYS[1:]}, tensors).items():
        w.add_key_value(k, v, VT.UINT32)
        rec["added_keys"][k] = v
    for name in keep:
        t, src_name = tensors[name]
        w.add_tensor(name, np.asarray(t.data), raw_shape=t.data.shape, raw_dtype=t.tensor_type)
        rec["tensors"][name] = dict(type=t.tensor_type.name, shape=[int(x) for x in t.shape],
                                    bytes=int(t.n_bytes), source=src_name)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return rec
