#!/usr/bin/env python3
"""Static check of the scripts that build proofs through the toy transformer
demo, against the current tree, without a card.

In July 2026 one keyword removed from `Tape.rmsnorm` broke every bench that
builds through `demo/demo_toy_transformer.py`, and nothing noticed for a month
because they only run on a GPU. This check runs on a CPU in about a second and
is part of tools/cpu_gates.sh. For each maintained file it reads the source,
never imports it, and reports:

  - an import that does not resolve, where internal modules are looked up on
    the sys.path the file itself builds (its directory, its sys.path.insert
    and append calls, and those of the internal modules it imports, in source
    order), and external ones against the standard library and EXTERNAL;
  - a name imported `from` an internal module that the module does not define;
  - an attribute read on an internal module alias that the module does not
    define;
  - a call to an internal function or class, module-qualified or imported by
    name, or to a `Tape` method on a tape, with an unknown keyword, too many
    positional arguments, or a required argument missing;
  - a LIGERO_* knob that nothing in prover/, demo/, verifier/, profiler/ or
    tools/ reads.

With no arguments it checks MAINTAINED and reports any analysis/bench script
that imports the toy demo but is not listed. With arguments it checks those
files only. Exit status: 0 clean, 1 findings, 2 usage (no files, a missing
file).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOY_DEMO = "demo_toy_transformer"

# The demo itself and every bench that builds through it, plus the routed-cache
# A/B the GPU gates run. A new bench that imports the demo must be added here
# (the default run reports one that is not).
MAINTAINED = [
    "demo/demo_toy_transformer.py",
    "analysis/bench/ab_coset_insitu.py",
    "analysis/bench/ab_gpu_softmax.py",
    "analysis/bench/ab_memgated_seq1024.py",
    "analysis/bench/ab_routed_cache.py",
    "analysis/bench/ab_witness_cache.py",
    "analysis/bench/ab_witness_spill.py",
    "analysis/bench/accept_toy_cache.py",
    "analysis/bench/accept_toy_spill.py",
    "analysis/bench/confirm_cap_seq1024.py",
    "analysis/bench/cost_calculator.py",
    "analysis/bench/gpu_softmax_ab.py",
    "analysis/bench/optrun_rho.py",
    "analysis/bench/phase_dump_seq1024.py",
    "analysis/bench/softmax_internal_profile.py",
    "analysis/bench/spill_ab.py",
    "analysis/bench/spill_costmodel.py",
    "analysis/bench/sweep_witness_cache.py",
    "analysis/bench/validate_coset_ntt.py",
    "analysis/bench/validate_disk_spill.py",
    "analysis/bench/validate_gpu_silu.py",
    "analysis/bench/validate_gpu_softmax.py",
    "analysis/bench/validate_witness_cache.py",
    "analysis/bench/validate_witness_spill.py",
    "analysis/bench/witness_recompute_probe.py",
    "analysis/bench/witness_type_probe.py",
]

EXTERNAL = {"numpy", "torch", "blake3", "gguf", "safetensors", "matplotlib",
            "pandas", "scipy", "psutil", "pytest", "transformers", "hf_transfer"}
KNOB_DIRS = ["prover", "demo", "verifier", "profiler", "tools"]


class ModuleInfo:
    """The module-level surface of one internal source file."""

    def __init__(self, path: Path):
        self.path = path
        self.tree = ast.parse(path.read_text(), filename=str(path))
        self.names: set[str] = set()
        self.defs: dict[str, ast.AST] = {}
        self.star = False
        for node in self.tree.body:
            self._collect(node, top=True)

    def _collect(self, node, top):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.names.add(node.name)
            self.defs.setdefault(node.name, node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        self.names.add(n.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name == "*":
                    self.star = True
                else:
                    self.names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    self._collect(child, top)
            for h in getattr(node, "handlers", []):
                for child in h.body:
                    self._collect(child, top)
            for child in getattr(node, "orelse", []) + getattr(node, "finalbody", []):
                self._collect(child, top)


_infos: dict[Path, ModuleInfo] = {}


def info(path: Path) -> ModuleInfo:
    if path not in _infos:
        _infos[path] = ModuleInfo(path)
    return _infos[path]


def _signature(node):
    """(positional names, keyword-only names, required positional, required
    keyword-only, *args?, **kwargs?) of a def, or of a class's __init__ or
    dataclass fields; None when it cannot be read statically."""
    if isinstance(node, ast.ClassDef):
        init = next((b for b in node.body if isinstance(b, ast.FunctionDef)
                     and b.name == "__init__"), None)
        if init is not None:
            sig = _signature(init)
            return None if sig is None else (sig[0][1:], sig[1], sig[2][1:], sig[3], sig[4], sig[5])
        is_dc = any((isinstance(d, ast.Name) and d.id == "dataclass")
                    or (isinstance(d, ast.Call) and getattr(d.func, "id", "") == "dataclass")
                    or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
                    for d in node.decorator_list)
        if not is_dc or node.bases:
            return None
        fields, required = [], []
        for b in node.body:
            if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name) \
                    and "ClassVar" not in ast.unparse(b.annotation):
                fields.append(b.target.id)
                if b.value is None:
                    required.append(b.target.id)
        return fields, [], required, [], False, False
    a = node.args
    pos = [x.arg for x in a.posonlyargs + a.args]
    req_pos = pos[:len(pos) - len(a.defaults)]
    kwonly = [x.arg for x in a.kwonlyargs]
    req_kw = [k.arg for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is None]
    return pos, kwonly, req_pos, req_kw, a.vararg is not None, a.kwarg is not None


def _check_call(call: ast.Call, node, what: str, method: bool) -> list[str]:
    sig = _signature(node)
    if sig is None:
        return []
    pos, kwonly, req_pos, req_kw, star, dstar = sig
    if method:
        pos, req_pos = pos[1:], req_pos[1:]
    out = []
    expand = any(isinstance(x, ast.Starred) for x in call.args) \
        or any(k.arg is None for k in call.keywords)
    kws = [k.arg for k in call.keywords if k.arg is not None]
    unknown = [k for k in kws if k not in pos and k not in kwonly]
    if unknown and not dstar:
        out.append(f"{what}: unknown keyword(s) {', '.join(unknown)}")
    npos = sum(not isinstance(x, ast.Starred) for x in call.args)
    if npos > len(pos) and not star:
        out.append(f"{what}: {npos} positional arguments, at most {len(pos)}")
    if not expand:
        missing = [p for p in req_pos[npos:] if p not in kws] + [k for k in req_kw if k not in kws]
        if missing:
            out.append(f"{what}: missing {', '.join(missing)}")
    return out


class FileCheck:
    def __init__(self, path: Path, knobs_read: set[str], sys_path=None, visited=None):
        self.path = path
        self.src = path.read_text()
        self.tree = ast.parse(self.src, filename=str(path))
        self.knobs_read = knobs_read
        self.findings: list[str] = []
        # sys.path is process-global: a checked script starts with its own
        # directory, and every internal module it imports edits the same list
        self.sys_path: list[Path] = sys_path if sys_path is not None else [path.parent]
        self.visited: set[Path] = visited if visited is not None else {path}
        self.mod_alias: dict[str, Path] = {}      # alias -> internal module file
        self.symbols: dict[str, tuple[Path, str]] = {}   # imported name -> (module, name)
        self.pathlib_alias: set[str] = set()
        self.path_class: set[str] = set()
        self.sys_alias: set[str] = set()
        self.os_alias: set[str] = set()
        self.env: dict[str, Path] = {}
        self.tape_vars: set[str] = {"tape"}
        self.tape_class: ast.ClassDef | None = None

    def report(self, node, msg):
        self.findings.append(f"{self.path.relative_to(ROOT) if self.path.is_relative_to(ROOT) else self.path}:"
                             f"{getattr(node, 'lineno', 0)}: {msg}")

    # -- statically evaluated paths (sys.path entries) --------------------
    def _path(self, e):
        if isinstance(e, ast.Name):
            if e.id == "__file__":
                return self.path
            return self.env.get(e.id)
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            return e.value
        if isinstance(e, ast.Call):
            f = e.func
            fname = ast.unparse(f)
            if (isinstance(f, ast.Name) and f.id in self.path_class) or \
                    (isinstance(f, ast.Attribute) and f.attr == "Path"
                     and isinstance(f.value, ast.Name) and f.value.id in self.pathlib_alias):
                v = self._path(e.args[0]) if e.args else None
                return Path(v) if v is not None else None
            if isinstance(f, ast.Name) and f.id == "str" and e.args:
                return self._path(e.args[0])
            if isinstance(f, ast.Attribute) and f.attr in ("resolve", "absolute") and not e.args:
                return self._path(f.value)
            if fname.split(".")[-2:] in (["path", "dirname"],) and e.args:
                v = self._path(e.args[0])
                return Path(v).parent if v is not None else None
            if fname.split(".")[-2:] in (["path", "abspath"], ["path", "realpath"]) and e.args:
                v = self._path(e.args[0])
                return Path(v) if v is not None else None
            if fname.split(".")[-2:] == ["path", "join"] and e.args:
                parts = [self._path(a) for a in e.args]
                if any(p is None for p in parts):
                    return None
                return Path(*map(str, parts))
            return None
        if isinstance(e, ast.Attribute) and e.attr == "parent":
            v = self._path(e.value)
            return Path(v).parent if v is not None else None
        if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Attribute) \
                and e.value.attr == "parents" and isinstance(e.slice, ast.Constant):
            v = self._path(e.value.value)
            return Path(v).parents[e.slice.value] if v is not None else None
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Div):
            left, right = self._path(e.left), self._path(e.right)
            return Path(left) / str(right) if left is not None and right is not None else None
        return None

    def _sys_path_call(self, call: ast.Call):
        f = call.func
        if not (isinstance(f, ast.Attribute) and f.attr in ("insert", "append")
                and isinstance(f.value, ast.Attribute) and f.value.attr == "path"
                and isinstance(f.value.value, ast.Name) and f.value.value.id in self.sys_alias):
            return False
        arg = call.args[-1] if call.args else None
        p = self._path(arg) if arg is not None else None
        if p is None:
            self.report(call, f"sys.path entry not statically resolvable: {ast.unparse(arg) if arg else '?'}")
            return True
        p = Path(p).resolve()
        if f.attr == "insert":
            self.sys_path.insert(0, p)
        else:
            self.sys_path.append(p)
        return True

    # -- imports ------------------------------------------------------------
    def _find(self, name: str):
        rel = name.replace(".", "/")
        for d in self.sys_path:
            for cand in (d / f"{rel}.py", d / rel / "__init__.py"):
                if cand.exists():
                    return cand.resolve()
        return None

    def _external(self, name: str):
        top = name.split(".")[0]
        return top in sys.stdlib_module_names or top in EXTERNAL

    def _inherit_paths(self, mod: Path):
        """Importing an internal module runs its sys.path edits and its own
        imports, once, on the same global list: replay them in source order."""
        if mod in self.visited:
            return
        self.visited.add(mod)
        FileCheck(mod, self.knobs_read, self.sys_path, self.visited)._walk_paths_only()

    def _import(self, node):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "sys":
                    self.sys_alias.add(a.asname or "sys")
                if a.name == "os":
                    self.os_alias.add(a.asname or "os")
                if a.name == "pathlib":
                    self.pathlib_alias.add(a.asname or "pathlib")
                m = self._find(a.name)
                if m is not None:
                    self.mod_alias[a.asname or a.name.split(".")[0]] = m
                    self._inherit_paths(m)
                elif not self._external(a.name):
                    self.report(node, f"import {a.name}: not found on the file's sys.path")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "pathlib":
                self.path_class.update(a.asname or a.name for a in node.names if a.name == "Path")
            m = self._find(node.module)
            if m is None:
                if not self._external(node.module):
                    self.report(node, f"from {node.module}: not found on the file's sys.path")
                return
            self._inherit_paths(m)
            mi = info(m)
            for a in node.names:
                if a.name == "*":
                    continue
                if a.name not in mi.names and not mi.star:
                    self.report(node, f"from {node.module} import {a.name}: not defined there")
                else:
                    self.symbols[a.asname or a.name] = (m, a.name)

    # -- walk -------------------------------------------------------------
    def _walk_paths_only(self):
        for node in self._ordered(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "sys":
                        self.sys_alias.add(a.asname or "sys")
                    if a.name == "pathlib":
                        self.pathlib_alias.add(a.asname or "pathlib")
                    m = self._find(a.name)
                    if m is not None:
                        self._inherit_paths(m)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module == "pathlib":
                    self.path_class.update(a.asname or a.name for a in node.names if a.name == "Path")
                m = self._find(node.module)
                if m is not None:
                    self._inherit_paths(m)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                p = self._path(node.value)
                if p is not None:
                    self.env[node.targets[0].id] = Path(p)
            elif isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr in ("insert", "append") \
                        and isinstance(f.value, ast.Attribute) and f.value.attr == "path" \
                        and isinstance(f.value.value, ast.Name) and f.value.value.id in self.sys_alias:
                    arg = node.args[-1] if node.args else None
                    p = self._path(arg) if arg is not None else None
                    if p is not None:
                        q = Path(p).resolve()
                        if f.attr == "insert":
                            self.sys_path.insert(0, q)
                        else:
                            self.sys_path.append(q)

    @staticmethod
    def _ordered(tree):
        """Every node, in source order."""
        out = []

        def rec(n):
            out.append(n)
            for c in ast.iter_child_nodes(n):
                rec(c)
        rec(tree)
        return out

    def run(self):
        for node in self._ordered(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._import(node)
            elif isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    p = self._path(node.value)
                    if p is not None:
                        self.env[node.targets[0].id] = Path(p)
                    v = node.value
                    if isinstance(v, ast.Call) and self._resolve_callee(v.func) is not None:
                        tgt, _, _ = self._resolve_callee(v.func)
                        if isinstance(tgt, ast.ClassDef) and tgt.name == "Tape":
                            self.tape_vars.add(node.targets[0].id)
            elif isinstance(node, ast.Call):
                if self._sys_path_call(node):
                    continue
                self._call(node)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) \
                    and isinstance(node.value, ast.Name) and node.value.id in self.mod_alias:
                mi = info(self.mod_alias[node.value.id])
                if node.attr not in mi.names and not mi.star:
                    self.report(node, f"{node.value.id}.{node.attr}: not defined in "
                                f"{mi.path.relative_to(ROOT) if mi.path.is_relative_to(ROOT) else mi.path}")
        for m in re.finditer(r"""["'](LIGERO_[A-Z0-9_]+)["']""", self.src):
            if m.group(1) not in self.knobs_read:
                line = self.src.count("\n", 0, m.start()) + 1
                self.findings.append(f"{self.path.relative_to(ROOT) if self.path.is_relative_to(ROOT) else self.path}:"
                                     f"{line}: knob {m.group(1)} is read nowhere in {', '.join(KNOB_DIRS)}")
        return self.findings

    def _resolve_callee(self, f):
        """(definition node, display name, is_method) for an internal callee."""
        if isinstance(f, ast.Name) and f.id in self.symbols:
            mod, name = self.symbols[f.id]
            d = info(mod).defs.get(name)
            return (d, name, False) if d is not None else None
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            if f.value.id in self.mod_alias:
                d = info(self.mod_alias[f.value.id]).defs.get(f.attr)
                return (d, f"{f.value.id}.{f.attr}", False) if d is not None else None
            if f.value.id in self.tape_vars:
                tape_cls = self._tape_class()
                if tape_cls is not None:
                    m = next((b for b in tape_cls.body if isinstance(b, ast.FunctionDef)
                              and b.name == f.attr), None)
                    if m is not None:
                        return m, f"Tape.{f.attr}", True
        return None

    def _tape_class(self):
        if self.tape_class is None:
            tape_py = ROOT / "prover" / "tape.py"
            d = info(tape_py.resolve()).defs.get("Tape")
            self.tape_class = d if isinstance(d, ast.ClassDef) else None
        return self.tape_class

    def _call(self, call: ast.Call):
        r = self._resolve_callee(call.func)
        if r is None:
            return
        node, what, method = r
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for msg in _check_call(call, node, what, method):
                self.report(call, msg)


def knobs_read() -> set[str]:
    me = Path(__file__).resolve()
    found: set[str] = set()
    for d in KNOB_DIRS:
        for p in (ROOT / d).rglob("*"):
            if p.suffix in (".py", ".rs", ".sh") and p.is_file() and p.resolve() != me \
                    and "target" not in p.parts and "fixtures" not in p.parts:
                found.update(re.findall(r"LIGERO_[A-Z0-9_]+", p.read_text(errors="replace")))
    return found


def unlisted(bench_dir: Path, maintained: list[str]) -> list[str]:
    listed = {(ROOT / m).resolve() for m in maintained}
    out = []
    for p in sorted(bench_dir.glob("*.py")):
        if p.resolve() not in listed and re.search(rf"\b{TOY_DEMO}\b", p.read_text()):
            out.append(f"{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}: imports the toy demo "
                       f"but is not in MAINTAINED (tools/bench_static_check.py)")
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    default = not argv
    files = [ROOT / m for m in MAINTAINED] if default else [Path(a) for a in argv]
    if not files:
        print("bench_static_check: no files to check", file=sys.stderr)
        return 2
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        print(f"bench_static_check: no such file: {', '.join(missing)}", file=sys.stderr)
        return 2
    knobs = knobs_read()
    findings: list[str] = []
    for f in files:
        findings += FileCheck(f.resolve(), knobs).run()
    if default:
        findings += unlisted(ROOT / "analysis" / "bench", MAINTAINED)
    for line in findings:
        print(line)
    print(f"bench_static_check: {len(files)} files, {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
