#!/usr/bin/env python3

import argparse
import copy
import csv
import json
import pathlib
import random
import subprocess
import tempfile
import time
from run_net_bench import free_ports


def fixture(document, seed=733):
    m = document["multi"]
    c = m["common"]
    n = c.get("n", 3)
    rows = c.get("rows", [6] * n)
    rng = random.Random(seed)
    filters = sorted({t["filter"] for t in m["templates"] if t.get("filter")})
    values = [{"inputs": {}} for _ in range(n)]
    incoming = []
    outgoing = []
    all_selected = []
    unfiltered = any(not t.get("filter") for t in m["templates"])
    cp = any(t.get("pattern", "P1") == "P4" for t in m["templates"])
    for bank, N in enumerate(rows):
        columns = {
            key: []
            for key in [
                "sid",
                "dir",
                "t",
                "a",
                *(["cp"] if cp else []),
                *("f_" + f for f in filters),
            ]
        }
        for i in range(N):
            d = (i + bank) % 2
            t = rng.randrange(5)
            a = 100 + rng.randrange(3)
            party = rng.choice([0, 7, (1 << 64) - 1])
            sid = 42 if i % 5 != 4 else 9
            flags = {f: bool(rng.randrange(2)) for f in filters}
            union = unfiltered or any(flags.values())
            row = {"t": t, "a": a, "cp": party, "flags": flags}
            columns["sid"].append(sid)
            columns["dir"].append(d)
            columns["t"].append(t)
            columns["a"].append(str(a))
            if cp:
                columns["cp"].append(str(party))
            for f in filters:
                columns["f_" + f].append(int(flags[f]))
            if sid == 42 and union:
                (outgoing if d else incoming).append(row)
        values[bank]["inputs"] = {
            name + "_" + str(bank): column for name, column in columns.items()
        }
    values[c.get("initiator", n - 1)]["inputs"]["subject"] = [42]
    kin = c.get("kin", 4)
    kout = c.get("kout", 3)
    omega = int(len(incoming) > kin or len(outgoing) > kout)
    incoming = incoming[:kin]
    outgoing = outgoing[:kout]
    result = {}
    for template in m["templates"]:
        pattern = template.get("pattern", "P1")
        window = template.get("window", template.get("windows", c.get("windows", [[0, 6]]))[0])
        L, U = window
        p = template.get("p", c.get("p", 5))
        q = template.get("q", c.get("q", 100))
        f = template.get("filter")
        flag = lambda r: not f or r["flags"][f]
        match = (
            lambda x, y: L <= y["t"] - x["t"] <= U
            and (q - p) * x["a"] <= q * y["a"] <= (q + p) * x["a"]
            and flag(x)
            and flag(y)
        )
        if pattern == "P2":
            count = sum(y["a"] for y in outgoing if any(match(x, y) for x in incoming))
        elif pattern == "P3":
            lo = template.get("a_lo", c.get("a_lo", 1))
            hi = template.get("a_hi", c.get("a_hi", 1 << c.get("ba", 40)))
            threshold = template.get("m", c.get("m", 3)) - 1
            count = sum(
                flag(y)
                and lo <= y["a"] < hi
                and sum(
                    i != j and flag(x) and lo <= x["a"] < hi and 0 <= y["t"] - x["t"] <= U
                    for i, x in enumerate(incoming)
                )
                >= threshold
                for j, y in enumerate(incoming)
            )
        elif pattern == "P4":
            count = sum(match(x, y) and x["cp"] == y["cp"] for x in outgoing for y in incoming)
        else:
            count = sum(match(x, y) for x in incoming for y in outgoing)
        result[template["name"]] = (
            int(count >= template.get("threshold", c.get("threshold", 1)))
            if template.get("output", c.get("output", "count")) == "alarm"
            else count
        )
    return values, {"omega": omega, "templates": result}


def run_case(
    build,
    document,
    material_mode="prg",
    threads=2,
    seed=733,
    timeout=45,
    reuse=True,
    network_context=None,
):
    build = pathlib.Path(build).resolve()
    values, expected = fixture(document, seed)
    c = document["multi"]["common"]
    n = c.get("n", 3)
    init = c.get("initiator", n - 1)
    with tempfile.TemporaryDirectory(prefix="icde-t7-") as tmp:
        path = pathlib.Path(tmp)
        query = path / "query.json"
        query.write_text(json.dumps(document))
        plan = path / "plan.json"
        binary = build / "bench/multi_bench"
        comp = subprocess.run(
            [str(binary), "--compile", str(query), "--out", str(plan)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if comp.returncode:
            raise RuntimeError(comp.stderr)
        from runtime import check_plan

        check_plan(build, plan, material_mode)
        compiled = json.loads(comp.stdout)
        package = path / "material"
        deal = subprocess.run(
            [
                str(build / "dealer_main"),
                "--plan",
                str(plan),
                "--out",
                str(package),
                "--mode",
                material_mode,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if deal.returncode:
            raise RuntimeError(deal.stderr)
        dealer = json.loads(deal.stdout)
        assert dealer["network_guard_active"] and dealer["online_messages"] == 0
        hosts = path / "hosts.json"
        hosts.write_text(
            json.dumps(
                {
                    "hosts": [
                        {"host": h, "port": p}
                        for h, p in zip(
                            network_context.hosts if network_context else ["127.0.0.1"] * n,
                            free_ports(n),
                        )
                    ],
                    "threads": threads,
                    "timeout_ms": min(1200000, timeout * 1000),
                }
            )
        )
        procs = []
        commands = []
        deadline = time.monotonic() + timeout
        try:
            for party in range(n):
                inp = path / f"input_{party}.json"
                inp.write_text(json.dumps(values[party]))
                command = [
                    str(binary),
                    "--query",
                    str(query),
                    "--plan",
                    str(plan),
                    "--material",
                    str(package / f"party_{party}.bin"),
                    "--input",
                    str(inp),
                    "--hosts",
                    str(hosts),
                    "--id",
                    str(party),
                    "--n",
                    str(n),
                ]
                commands.append(command)
                procs.append(
                    subprocess.Popen(
                        network_context.command(party, command) if network_context else command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            reports = []
            for proc in procs:
                out, err = proc.communicate(timeout=max(0.1, deadline - time.monotonic()))
                if proc.returncode:
                    raise RuntimeError(err)
                reports.append(json.loads(out))
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
            for proc in procs:
                proc.wait()
        for party, row in enumerate(reports):
            assert (
                row["logical_share_bits"]
                == dealer["logical_share_bits_per_party"][party]
                == compiled["logical_share_bits"]
            )
            assert row["open_hops"] == row["open_rounds"] * (
                2 if n >= 3 and c.get("open_mode", "auto") == "auto" else 1
            )
            assert row["online_new_calls"] == 0, (
                party,
                "online allocations",
                row["online_new_calls"],
            )
            assert row["compact"]["open_rounds"] == 11
            s = row["schedule"]
            assert (
                sum(s["wave_rounds"])
                == s["graph_rounds"]
                == s["merged_opens"] + s["merged_shuffle_hops"]
            )
            assert (
                s["merged_opens"] <= s["unmerged_opens"]
                and s["merged_shuffle_hops"] <= s["unmerged_shuffle_hops"]
            )
            union_depth = (
                0
                if any(not t.get("filter") for t in document["multi"]["templates"])
                else (len({t["filter"] for t in document["multi"]["templates"]}) - 1).bit_length()
            )
            assert row["open_rounds"] == 11 + union_depth + s["merged_opens"]
            if party == init:
                assert row["result"] == expected, (row.get("result"), expected)
            else:
                assert "result" not in row
        if reuse:
            retried = subprocess.run(commands[0], capture_output=True, text=True, timeout=10)
            assert retried.returncode and "one-time material" in retried.stderr
        return {
            "configuration": document,
            "compiled": compiled,
            "expected": expected,
            "dealer": dealer,
            "parties": reports,
            "correct": True,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", default=str(pathlib.Path(__file__).resolve().parents[1] / "build"))
    p.add_argument("--query", required=True)
    p.add_argument("--material-mode", choices=["random", "prg"], default="prg")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    doc = json.loads(pathlib.Path(a.query).read_text())
    r = run_case(a.build, doc, a.material_mode, a.threads)
    out = pathlib.Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(r, indent=2))
    rows = []
    for bank in r["parties"]:
        rows.append(
            {
                "party": bank["id"],
                "n": bank["n"],
                "templates": len(doc["multi"]["templates"]),
                "seconds": bank["online_seconds"],
                "open_rounds": bank["open_rounds"],
                "operator_graph_rounds": bank["schedule"]["graph_rounds"],
                "unmerged_opens": bank["schedule"]["unmerged_opens"],
                "merged_opens": bank["schedule"]["merged_opens"],
                "contribution_bits": bank["contribution_bits"],
                "logical_share_bits": bank["logical_share_bits"],
                "online_new_calls": bank["online_new_calls"],
                "correct": True,
            }
        )
    with out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    print(
        json.dumps(
            {
                "correct": True,
                "expected": r["expected"],
                "shared_threshold_tables": r["compiled"]["shared_threshold_tables"],
                "shared_time_groups": r["compiled"]["group_members"],
                "max_seconds": max(x["seconds"] for x in rows),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
