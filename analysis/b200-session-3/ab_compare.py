"""Side-by-side of two arms' per-sweep tables, the kind lines under each row
(loader seconds and calls by kind, cache hits and stores), the weight-cache
closing line and the fixed costs.
    python3 ab_compare.py mavp-s100-wc-off.log mavp-s100-wc-on.log"""
import re, sys

def parse(path):
    lines = open(path, errors="replace").read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith("sweep ") and "wall" in l)
    head = lines[i].split()
    rows, kinds, cur = {}, {}, None
    for l in lines[i + 1:]:
        p = l.split()
        if not p or p[0] == "proof": break
        if p[0] == "loader":                       # "loader calls by kind: weight X s / N; shard ...; weight-cache hits N, stores N"
            d = {}
            for m in re.finditer(r"(weight|shard|input) ([\d.]+) s / ([\d,]+)", l):
                d[f"{m.group(1)}_s"] = float(m.group(2)); d[f"{m.group(1)}_n"] = int(m.group(3).replace(",", ""))
            m = re.search(r"hits ([\d,]+)", l);  d["hits"] = int(m.group(1).replace(",", "")) if m else 0
            m = re.search(r"stores ([\d,]+)", l); d["stores"] = int(m.group(1).replace(",", "")) if m else 0
            kinds[cur] = d
            continue
        cur = p[0]
        rows[cur] = dict(zip(head[1:], [float(x.replace(",", "")) for x in p[1:]]))
    fixed = {}
    for l in lines:
        for key, pat in (("enroll_s", r"enrolled \d+ weight rows in ([\d.]+)s"), ("reveal_s", r"reveal engine pass ([\d.]+)s"),
                         ("prove_s", r"PROVE WALL ([\d.]+)s"), ("peak_GiB", r"peakGPU=([\d.]+)GiB"), ("build_s", r"build=([\d.]+)s"),
                         ("wc_packed_GB", r"\(([\d.]+) GB packed"), ("wc_pinned_GB", r"([\d.]+) GB pinned after"),
                         ("wc_reserved_GB", r"reserved ([\d.]+) GB"), ("wc_refused", r"([\d,]+) resolutions refused"),
                         ("wc_weights", r"\[weight-cache\] ([\d,]+) weights decoded once"), ("wc_hits", r"([\d,]+) resolutions served")):
            m = re.search(pat, l)
            if m: fixed[key] = float(m.group(1).replace(",", ""))
        if "opened columns match committed leaves" in l: fixed["leaf_check"] = l.split("leaves: ")[1].split(";")[0]
    return head, rows, kinds, fixed

_, off, koff, foff = parse(sys.argv[1]); _, on, kon, fon = parse(sys.argv[2])
cols = ["wall", "witness", "fetch", "aux", "encode", "fold_qlin", "loads", "load_GB", "routed_rd", "proj"]
print(f"{'sweep':6s} {'col':10s} {'off':>10s} {'on':>10s} {'on-off':>10s} {'ratio':>7s}")
for sw in ("R1", "R2", "R3", "fold", "open", "ALL"):
    for c in cols:
        a, b = off[sw].get(c, 0), on[sw].get(c, 0)
        if a == 0 and b == 0: continue
        print(f"{sw:6s} {c:10s} {a:10.1f} {b:10.1f} {b - a:10.1f} {f'{b / a:7.3f}' if a else '    n/a'}")
    for c in ("weight_s", "weight_n", "shard_s", "shard_n", "input_s", "input_n", "hits", "stores"):
        a, b = koff.get(sw, {}).get(c, 0), kon.get(sw, {}).get(c, 0)
        if a == 0 and b == 0: continue
        print(f"{sw:6s} {c:10s} {a:10.1f} {b:10.1f} {b - a:10.1f} {f'{b / a:7.3f}' if a else '    n/a'}")
    print()
print("fixed:", {k: (foff.get(k), fon.get(k)) for k in ("build_s", "enroll_s", "reveal_s", "prove_s", "peak_GiB", "leaf_check")})
print("weight cache (on arm):", {k: fon.get(k) for k in ("wc_weights", "wc_packed_GB", "wc_pinned_GB", "wc_reserved_GB", "wc_hits", "wc_refused")})
po, pn = foff.get("prove_s", 0), fon.get("prove_s", 0)
print(f"prove: off {po:.1f}s, on {pn:.1f}s, saving {po - pn:.1f}s = {100 * (po - pn) / po if po else 0:.1f}%")
w_off = koff.get("ALL", {}).get("weight_s", 0); w_on = kon.get("ALL", {}).get("weight_s", 0)
print(f"dense loader seconds per proof: off {w_off:.1f} ({koff.get('ALL', {}).get('weight_n', 0)} calls), on {w_on:.1f} ({kon.get('ALL', {}).get('weight_n', 0)} calls + {kon.get('ALL', {}).get('hits', 0)} hits)")
