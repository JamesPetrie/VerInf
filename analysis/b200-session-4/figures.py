"""Session-4 figures (python3 analysis/b200-session-4/figures.py writes PNG+SVG under figures/), self-contained: the four S=100 arms on the B200 and what
the bridge removed. Light surface, the reference palette's first slots, from
the arm logs copied home. Reads the per-sweep tables and the kind lines."""
import pathlib, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = pathlib.Path(__file__).resolve().parent          # analysis/b200-session-4
sys.path.insert(0, str(S))
from ab_compare import parse

OUT = S / "figures"; OUT.mkdir(exist_ok=True)
DATE = "2026-09-20"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, NEUTRAL, VIOLET, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b1", "#7c5cd6", "#c99a1e"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})
SETTING = ("Session 4, September 19: the VerInf prover on Llama-4 Maverick (48 layers, 128 experts) from the real Q4 GGUF "
           "on one rented NVIDIA B200, 100 tokens (50 prompt + 50 continuation), target proof geometry, the routed-output "
           "cache on in every arm. Each arm is one full proof, accepted by the leaf check; walls are instrumented.")

ARMS = [("session 3 · no cache\n(other host)", S.parent / "b200-session-3/logs/mavp-s100-wc-off.log"),
        ("no weight cache", S / "logs/mavp-s100-wc-off2.log"),
        ("weight cache, GPU tier", S / "logs/mavp-s100-wc-on2.log"),
        ("weight bridge\n(expert weights leave the witness)", S / "logs/mavp-s100-bridge.log")]


def parts(path):
    _, rows, kinds, fx = parse(str(path))
    A, K = rows["ALL"], kinds.get("ALL", {})
    enc = A.get("encode", 0) + A.get("fold_qlin", 0)
    wl, sl = K.get("weight_s", 0), K.get("shard_s", 0)
    wit = A.get("witness", 0)
    sweeps = A["wall"]
    other = max(0.0, sweeps - enc - wl - sl - wit)
    outside = max(0.0, fx["prove_s"] - sweeps)
    return dict(enc=enc, wl=wl, sl=sl, wit=wit, other=other, outside=outside, prove=fx["prove_s"], fx=fx, K=K, A=A)


def frame(fig, title, subtitle, caption, width=118):
    fig.suptitle(title, x=0.02, y=0.995, ha="left", va="top", fontsize=11.5, color=INK, fontweight="bold")
    fig.text(0.02, 0.945, textwrap.fill(subtitle, width), ha="left", va="top", fontsize=8.6, color=INK2, linespacing=1.35)
    fig.text(0.02, 0.01, textwrap.fill(caption, width), ha="left", va="bottom", fontsize=8.6, color=INK, linespacing=1.35)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"VerInf {name} ({DATE}).{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig); print("wrote", name)


P = [(label, parts(path)) for label, path in ARMS]

# ---- 1. where the S=100 prove goes, four arms
fig, ax = plt.subplots(figsize=(11, 6.4)); fig.subplots_adjust(top=0.74, bottom=0.27, left=0.24, right=0.97)
segs = [("enc", "encode + fold of the enrolled weights", BLUE), ("wl", "dense weight loader", ORANGE),
        ("sl", "expert shard loader", GOLD), ("wit", "witness compute", AQUA),
        ("other", "the rest inside the five sweeps", NEUTRAL), ("outside", "outside the sweeps (the bridge's own pass; setup)", VIOLET)]
y = list(range(len(P)))[::-1]
left = [0.0] * len(P)
for key, name, col in segs:
    vals = [p[key] for _, p in P]
    ax.barh(y, vals, left=left, color=col, height=0.62, label=name, edgecolor=SURF, linewidth=1.2)
    left = [l + v for l, v in zip(left, vals)]
for yi, (label, p) in zip(y, P):
    ax.text(p["prove"] + 25, yi, f"{p['prove']:,.0f} s", va="center", ha="left", fontsize=9, color=INK)
ax.set_yticks(y); ax.set_yticklabels([l for l, _ in P], fontsize=9)
ax.set_xlabel("seconds of prove wall (one proof, 100 tokens)"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
ax.set_xlim(0, max(p["prove"] for _, p in P) * 1.14)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8.4)
frame(fig, "Where a Maverick proof spends its time: four arms at 100 tokens",
      SETTING,
      "How to read it: each bar is one proof of the same tape. The blue part is the two passes over the enrolled model "
      "weights, the same in the first three arms and almost gone under the bridge, which authenticates the expert weights "
      "against a one-time registration instead of carrying them in the witness. The orange part is the dense weights being "
      "decoded from the GGUF; the weight cache decodes each once and keeps it on the card. The top arm is session 3's "
      "baseline on a different host with slower disks, so its loader parts are not comparable to the three below it; the "
      "second arm is the same-host baseline. Enrollment, reveal and build are outside these bars.")
save(fig, "session 4 four arms")

# ---- 2. what the bridge removed
off2, bridge = P[1][1], P[3][1]
fig, axes = plt.subplots(1, 4, figsize=(11, 5.4)); fig.subplots_adjust(top=0.70, bottom=0.30, wspace=0.55, left=0.07, right=0.98)
panels = [("enrolled weight rows\nin the witness (millions)", 49.16, 1.97, "M"),
          ("expert shard reads\nper proof (thousands)", 29.988, 11.556, "k"),
          ("encode + fold of the\nenrolled weights (s)", off2["enc"], bridge["enc"], "s"),
          ("verifier time, two-layer\ngate proof (s)", 272.1, 64.7, "s")]
for ax, (title, a, b, unit) in zip(axes, panels):
    bars = ax.bar([0, 1], [a, b], color=[NEUTRAL, VIOLET], width=0.62, edgecolor=SURF, linewidth=1.2)
    for x, v in zip([0, 1], [a, b]):
        ax.text(x, v, f"{v:,.1f}" if v < 100 else f"{v:,.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["committed\nweights", "bridge"], fontsize=8.6)
    ax.set_title(title, fontsize=9, color=INK2, loc="left"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.set_ylim(0, max(a, b) * 1.22); ax.tick_params(axis="y", labelsize=8)
frame(fig, "What the weight bridge takes out of a proof",
      "Session 4, September 19, one B200, Maverick at 100 tokens. 'Committed weights' is the same-host baseline arm; 'bridge' "
      "is the arm with the expert weights under the coefficient-RS registration. The verifier panel is from the two-layer gate proofs.",
      "How to read it: the expert weights are 96 percent of the enrolled block. Under the bridge they are neither committed "
      "as witness rows nor re-encoded per proof, so the rows, the shard reads and the encode time fall to the dense weights "
      "alone, and the verifier no longer folds the weight rows. What the bridge adds per proof is its own pass over the "
      "registration, the violet 'outside the sweeps' part of the previous figure, about 590 s here, plus a one-time "
      "registration of 273 s. The numbers are for the interim form of the bridge, before its committed (zero-knowledge) form.")
save(fig, "session 4 what the bridge removed")
