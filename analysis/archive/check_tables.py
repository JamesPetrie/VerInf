"""Verify every LogUp table's key-domain T is exactly range(T_LEN) (so it can be
serialized as just its length, not the full array), and report which tables carry
a T_Y (the small paired function arrays that stay explicit)."""
import sys, pathlib
_PIPE = pathlib.Path(__file__).resolve().parents[1] / "prover"   # analysis/ -> repo/pipeline
sys.path.insert(0, str(_PIPE)); sys.path.insert(0, str(_PIPE / "tests"))
import torch, core, claims as C, packets as PK, protocol as pr  # noqa
from test_compile_parity import cases

all_range = True
for tag, claim_list, cfg in cases():
    for t in pr._distinct_tables(list(claim_list)):
        T = t.T
        n = len(T)
        Tl = T.cpu().tolist() if torch.is_tensor(T) else [int(x) for x in T]
        is_range = (Tl == list(range(n)))
        if not is_range:
            all_range = False
            print(f"  NON-RANGE T in {tag}: len={n} head={Tl[:8]}")
        print(f"{tag}: T_LEN={n} is_range={is_range} has_TY={t.T_Y is not None}")
print(f"\nALL TABLES T==range(T_LEN): {all_range}")
