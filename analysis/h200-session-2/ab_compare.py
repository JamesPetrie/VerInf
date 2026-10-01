"""Side-by-side of the two arms' per-sweep tables: seconds and counts per
sweep, the on-minus-off delta, and the fixed costs. Usage:
    python3 s2_ab_compare.py mavp-s100-off.log mavp-s100-on.log"""
import re, sys

def parse(path):
    lines = open(path, errors="replace").read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith("sweep ") and "wall" in l)
    head = lines[i].split()
    rows = {}
    for l in lines[i + 1:]:
        p = l.split()
        if not p or p[0] == "proof": break
        rows[p[0]] = dict(zip(head[1:], [float(x) for x in p[1:]]))
    fixed = {}
    for l in lines:
        m = re.search(r"enrolled \d+ weight rows in ([\d.]+)s", l)
        if m: fixed["enroll_s"] = float(m.group(1))
        m = re.search(r"reveal engine pass ([\d.]+)s", l)
        if m: fixed["reveal_s"] = float(m.group(1))
        m = re.search(r"PROVE WALL ([\d.]+)s .*peakGPU=([\d.]+)GiB", l)
        if m: fixed["prove_s"], fixed["peak_GiB"] = float(m.group(1)), float(m.group(2))
        m = re.search(r"build ([\d.]+)s", l)
        if m: fixed["build_s"] = float(m.group(1))
    return head, rows, fixed

off_h, off, off_f = parse(sys.argv[1]); on_h, on, on_f = parse(sys.argv[2])
cols = ["wall", "witness", "fetch", "aux", "encode", "fold_qlin", "loads", "load_GB", "cache_rd", "routed_rd", "routed_wr", "proj"]
print(f"{'sweep':6s} {'col':10s} {'off':>10s} {'on':>10s} {'on-off':>10s} {'ratio':>7s}")
for sw in ("R1", "R2", "R3", "fold", "open", "ALL"):
    for c in cols:
        a, b = off[sw].get(c, 0), on[sw].get(c, 0)
        if a == 0 and b == 0: continue
        r = f"{b / a:7.3f}" if a else "   n/a"
        print(f"{sw:6s} {c:10s} {a:10.1f} {b:10.1f} {b - a:10.1f} {r}")
    print()
print("fixed (s):", {k: (off_f.get(k), on_f.get(k)) for k in ("build_s", "enroll_s", "reveal_s", "prove_s", "peak_GiB")})
tot_off = off_f.get("enroll_s", 0) + off_f.get("reveal_s", 0) + off_f.get("prove_s", 0)
tot_on = on_f.get("enroll_s", 0) + on_f.get("reveal_s", 0) + on_f.get("prove_s", 0)
print(f"arm total (enroll+reveal+prove): off {tot_off:.0f}s, on {tot_on:.0f}s, saving {tot_off - tot_on:.0f}s = {100 * (tot_off - tot_on) / tot_off:.1f}%")
print(f"prove only: off {off_f.get('prove_s', 0):.0f}s, on {on_f.get('prove_s', 0):.0f}s, saving {100 * (1 - on_f.get('prove_s', 1) / off_f.get('prove_s', 1)):.1f}%")
