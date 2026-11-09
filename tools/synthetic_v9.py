#!/usr/bin/env python3

import argparse
import hashlib
import json
import pathlib
import sys
import numpy as np
from column_data import write_bank
from bank_data import finish_dataset, prepare_directory, default_query
from fast_reference import aggregate


def generate(
    output,
    n=3,
    total=1 << 20,
    nu_in=256,
    nu_out=256,
    K=512,
    seed=733,
    match_rate=None,
    pattern="mixed",
    subjects=10000,
    counterparties=10000,
):
    if match_rate is not None and pattern == "mixed":
        pattern = "P1"
    if (
        n < 2
        or n > 1024
        or min(subjects, counterparties) < 1
        or K < 0
        or total < nu_in + nu_out
        or min(nu_in, nu_out) < 0
        or pattern not in ("mixed", "P1", "P2", "P3", "P4")
        or (match_rate is not None and not 0 <= match_rate <= 100)
    ):
        raise ValueError("invalid synthetic dimensions/rate")
    path = prepare_directory(output)
    rows = [total // n + int(i < total % n) for i in range(n)]
    rng = np.random.default_rng(seed)
    positions = rng.choice(total, nu_in + nu_out, replace=False)
    A = 150000
    rho = 1 if match_rate is None else match_rate / 100
    eligible_y = int(rho * (nu_in if pattern == "P4" else nu_out))
    eligible_p3 = 0 if rho == 0 else round((1 + np.sqrt(1 + 4 * rho * nu_in * (nu_in - 1))) / 2)
    target = []
    for i, pos in enumerate(positions):
        direction = int(i >= nu_in)
        index = i - nu_in if direction else i
        amount = A
        cp = 7
        if pattern == "mixed":
            t = 1000 + index % 2
        elif pattern == "P4":
            t = 1000 + int(not direction)
        else:
            t = 1000 + direction
        if match_rate is not None:
            if pattern == "P3" and not direction and index >= eligible_p3:
                amount = 2 * A
            elif pattern == "P4" and not direction and index >= eligible_y:
                amount = 2 * A
                cp = 8
            elif pattern in ("P1", "P2") and direction and index >= eligible_y:
                amount = 2 * A
        target.append((int(pos), direction, t, amount, cp))
    target.sort()
    banks = []
    offset = 0
    for bank, count in enumerate(rows):
        r = np.random.default_rng(seed + 100003 * (bank + 1))
        cols = {
            "sid": r.integers(100, 100 + subjects, count, dtype=np.uint64),
            "dir": r.integers(0, 2, count, dtype=np.uint8),
            "t": r.integers(0, 86400, count, dtype=np.uint64),
            "a": r.integers(1, 10000000, count, dtype=np.uint64),
            "cp": r.integers(1, counterparties + 1, count, dtype=np.uint64),
        }
        for pos, d, t, a, cp in target:
            if offset <= pos < offset + count:
                at = pos - offset
                cols["sid"][at] = 42
                cols["dir"][at] = d
                cols["t"][at] = t
                cols["a"][at] = a
                cols["cp"][at] = cp
        write_bank(path / f"bank_{bank}.bjcol", bank, n, cols)
        banks.append(
            {
                "bank": bank,
                "file": f"bank_{bank}.bjcol",
                "rows": count,
                "in_rows": int((cols["dir"] == 0).sum()),
                "out_rows": int((cols["dir"] == 1).sum()),
            }
        )
        offset += count
    ins = [(t, a, cp) for _, d, t, a, cp in target if not d]
    outs = [(t, a, cp) for _, d, t, a, cp in target if d]
    counts = {}
    for q in ["P1", "P2", "P3", "P4"]:
        c = {"pattern": q, "p": 5, "q": 100, "a_lo": A, "a_hi": A + 1, "m": 3}
        counts[q] = {
            "full": aggregate(ins, outs, c, 0, 1800),
            "retained": aggregate(ins[:K], outs[:K], c, 0, 1800),
        }
    query = default_query(rows, K, operator="nl", bt=29)
    for q in query["multi"]["templates"]:
        if q["pattern"] == "P3":
            q.update(a_lo=A, a_hi=A + 1, m=3)
    if pattern == "P3":
        actual_rate = eligible_p3 * (eligible_p3 - 1) / (nu_in * (nu_in - 1)) if nu_in > 1 else 0
    else:
        pairs = aggregate(
            ins, outs, {"pattern": "P4" if pattern == "P4" else "P1", "p": 5, "q": 100}, 0, 1800
        )
        actual_rate = pairs / (nu_in * nu_out) if nu_in * nu_out else 0
    source = {
        "kind": "synthetic",
        "generator": "numpy PCG64, public seed, complete row fixture",
        "seed": seed,
        "input_rows": total,
        "nu_in": nu_in,
        "nu_out": nu_out,
        "requested_match_rate_percent": match_rate,
        "actual_matching_pair_fraction": actual_rate,
        "control_pattern": pattern,
        "time_unit": "relative_seconds",
        "amount_unit": "cents",
        "subsampling": False,
    }
    finish_dataset(
        path,
        banks,
        source,
        42,
        {
            "target_rows": target,
            "expected": counts,
            "overflow_either": nu_in > K or nu_out > K,
            "overflow_P3": nu_in > K,
        },
        29,
        40,
        K,
        query,
    )
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--N", type=int, default=1 << 20)
    p.add_argument("--nu-in", type=int, default=256)
    p.add_argument("--nu-out", type=int, default=256)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--seed", type=int, default=733)
    p.add_argument("--match-rate", type=float)
    p.add_argument("--pattern", choices=["mixed", "P1", "P2", "P3", "P4"], default="mixed")
    a = p.parse_args()
    print(generate(a.output, a.n, a.N, a.nu_in, a.nu_out, a.K, a.seed, a.match_rate, a.pattern))


if __name__ == "__main__":
    main()
