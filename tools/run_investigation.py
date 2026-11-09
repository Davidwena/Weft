#!/usr/bin/env python3

import argparse
import csv
import json
import pathlib
import random
import subprocess
import tempfile
import time
import copy
import os
from run_net_bench import free_ports
from run_primitives import shape, costs


def configuration(
    n=3,
    kin=8,
    kout=None,
    rows=None,
    mode="pipeline",
    variant="C1+C2",
    windows=None,
    king=0,
    output="count",
    small=False,
):
    c = dict(
        n=n,
        kin=kin,
        kout=kin if kout is None else kout,
        rows=rows or [17] * n,
        mode=mode,
        variant=variant,
        windows=windows or [[0, 1800]],
        king=king,
        initiator=n - 1,
        open_mode="auto",
        output=output,
        threshold=3,
    )
    if small:
        c.update(bt=3, ba=3, bq=3, kt=6, ka=8, p=1, q=4, windows=windows or [[0, 2]])
    return {"investigation": c}


def generate(c, scenario="overflow", seed=733):
    rng = random.Random(seed)
    n = c["n"]
    kin = c["kin"]
    kout = c["kout"]
    bt = c.get("bt", 29)
    ba = c.get("ba", 40)
    theta = 1 << bt
    values = [{"inputs": {}} for _ in range(n)]
    subject = 42

    def amount(i):
        return min((1 << ba) - 1, 100 + i % 5)

    def timein(i):
        return min(theta - 1, 1000 + i % 20)

    def timeout(i):
        return min(theta - 1, 1010 + i % 20)

    if c["mode"] == "band":
        tin = [timein(i) for i in range(kin)]
        tout = [timeout(i) for i in range(kout)]
        ain = [amount(i) for i in range(kin)]
        aout = [amount(i) for i in range(kout)]
        if kin and scenario != "exact":
            tin[-1] = theta
            ain[-1] = 0
        if kout and scenario != "exact":
            tout[-1] = (1 << 64) - theta
            aout[-1] = 0
        if scenario == "zero":
            tin = [theta] * kin
            ain = [0] * kin
        oi = int(scenario == "overflow")
        oo = 0
        values[0]["inputs"].update(tin=tin, ain=list(map(str, ain)), omega_in=[oi])
        values[n - 1]["inputs"].update(tout=tout, aout=list(map(str, aout)), omega_out=[oo])
        inbound = list(zip(tin, ain))
        outbound = list(zip(tout, aout))
        omega = oi if c.get("pattern") == "P3" else oi | oo
        if c.get("pattern") == "P4":
            cpin = [i % 3 if a else 0 for i, a in enumerate(ain)]
            cpout = [i % 3 if a else 0 for i, a in enumerate(aout)]
            values[0]["inputs"]["cpin"] = list(map(str, cpin))
            values[n - 1]["inputs"]["cpout"] = list(map(str, cpout))
            inbound = [(*r, cp) for r, cp in zip(inbound, cpin)]
            outbound = [(*r, cp) for r, cp in zip(outbound, cpout)]
    else:
        total = sum(c["rows"])
        want_in = (
            0
            if scenario == "zero"
            else (
                kin
                if scenario == "exact"
                else kin + 3 if scenario == "overflow" else max(0, kin // 2)
            )
        )
        want_out = (
            0
            if scenario == "zero"
            else (
                kout
                if scenario == "exact"
                else kout + 3 if scenario == "overflow" else max(0, kout // 2)
            )
        )
        want_in = min(want_in, total // 2)
        want_out = min(want_out, total - want_in)
        selected = rng.sample(range(total), want_in + want_out)
        incoming = set(selected[:want_in])
        outgoing = set(selected[want_in:])
        inbound = []
        outbound = []
        offset = 0
        for bank, count in enumerate(c["rows"]):
            sid = []
            direction = []
            t = []
            a = []
            cp = []
            for i in range(count):
                row = offset + i
                hit = row in incoming or row in outgoing
                d = int(row in outgoing) if hit else i % 2
                sid.append(subject if hit else 1000 + i % 97)
                direction.append(d)
                tv = timeout(i) if d else timein(i)
                av = amount(i)
                t.append(tv)
                a.append(av)
                cp.append(i % 3)
                if hit:
                    (outbound if d else inbound).append(
                        (tv, av, i % 3) if c.get("pattern") == "P4" else (tv, av)
                    )
            values[bank]["inputs"].update(
                {
                    f"sid_{bank}": sid,
                    f"dir_{bank}": direction,
                    f"t_{bank}": t,
                    f"a_{bank}": list(map(str, a)),
                }
            )
            if c.get("pattern") == "P4":
                values[bank]["inputs"][f"cp_{bank}"] = list(map(str, cp))
            offset += count
        values[c["initiator"]]["inputs"]["subject"] = [subject]
        omega = int(len(inbound) > kin or (c.get("pattern") != "P3" and len(outbound) > kout))
        inbound = inbound[:kin]
        outbound = outbound[:kout]
    if (c.get("operator") == "dc" or c.get("pattern", "P1") != "P1") and c["mode"] == "band":
        vin = [int(t != theta) for t in tin]
        vout = [int(t != (1 << 64) - theta) for t in tout]
        values[0]["inputs"]["vin"] = vin
        values[n - 1]["inputs"]["vout"] = vout
        inbound = [x for x, v in zip(inbound, vin) if v]
        outbound = [x for x, v in zip(outbound, vout) if v]
    p = c.get("p", 5)
    q = c.get("q", 100)
    result = []
    for L, U in c["windows"]:
        pattern = c.get("pattern", "P1")

        def match(x, y):
            return L <= y[0] - x[0] <= U and (q - p) * x[1] <= q * y[1] <= (q + p) * x[1]

        if pattern in ("P2", "P3", "P4"):
            from fast_reference import aggregate

            count = aggregate(inbound, outbound, c, L, U)
        else:
            count = retained_count(inbound, outbound, L, U, p, q)
        result.append(
            int(count >= c["threshold"]) if c.get("output", "count") == "alarm" else count
        )
    if c["mode"] == "compact":
        result = []
    return values, {"omega": omega, "alarm" if c.get("output") == "alarm" else "count": result}


def retained_count(inbound, outbound, L, U, p, q):

    from bisect import bisect_left, bisect_right

    ins = sorted(inbound)
    amounts = sorted({a for _, a in ins})
    tree = [0] * (len(amounts) + 1)
    events = []
    for t, a in outbound:
        lo = (q * a + q + p - 1) // (q + p)
        hi = q * a // (q - p)
        events.extend([(t - L, 1, lo, hi), (t - U - 1, -1, lo, hi)])

    def prefix(i):
        total = 0
        while i:
            total += tree[i]
            i -= i & -i
        return total

    result = 0
    index = 0
    for t, sign, lo, hi in sorted(events):
        while index < len(ins) and ins[index][0] <= t:
            pos = bisect_left(amounts, ins[index][1]) + 1
            while pos < len(tree):
                tree[pos] += 1
                pos += pos & -pos
            index += 1
        if lo <= hi:
            result += sign * (prefix(bisect_right(amounts, hi)) - prefix(bisect_left(amounts, lo)))
    return result


def band_cost(c):
    kt = c.get("kt", 32)
    ka = c.get("ka", 49)
    v = c["variant"]
    W = len(c["windows"])
    P = c["kin"] * c["kout"]
    zero = v in ("C1+C2", "C1+zero")
    interval = v in ("C1+C2", "+interval")
    amount = v in ("C1+C2", "+amount")
    if interval:
        offsets = {x for L, U in c["windows"] for x in (L, U + 1)}
        table, inputs = shape(kt - 1)
        B = (inputs + 1) // 2
        d = min(range(1, 9), key=lambda d: shape(kt - 1, d)[0])
        t = B * (1 << d)
        pre = t + len(offsets) * (table - t) + (0 if zero else kt)
        online = len(offsets) * inputs + (0 if zero else kt)
    else:
        pre, online, _ = costs("msb_known" if zero else "msb_fresh", kt)
        pre *= 2
        online *= 2
    a, b, _ = costs("msb_known" if zero else "msb_fresh", ka)
    arity = (1 if interval else 2) + (1 if amount else 2)
    pre += 2 * a + W * (arity + ((1 << arity) - 1) * 64)
    online += 2 * b + W * arity
    rounds = 2 if zero else 3
    pre *= P
    online *= P
    if c.get("output") == "alarm":
        a, b, r = costs("msb_fresh", 32)
        pre += a * W
        online += b * W
        rounds += r
    return pre, online + 2, rounds


def compact_cost(c):
    N = sum(c["rows"])
    K = c["kin"] + c["kout"]
    kp = c.get("kp", max(22, (N + max(c["kin"], c["kout"])).bit_length() + 1))
    pre, online, _ = costs("msb_fresh", kp)
    gate = 2303 * N + 2 * 98 * N + 2 * pre * (N + 1) + 2 * 227 * N + pre * K + 33 * K + 64 * (N + K)
    bits = (
        10 * N
        + 4 * N
        + 2 * online * (N + 1)
        + 6 * N
        + online * K
        + K
        + 64 * (N + K)
        + 2 * (N + K)
        + (161 + (96 if c.get("pattern") == "P4" else 0)) * K
    )
    return gate, bits


def band_payload(c):
    P = c["kin"] * c["kout"]
    kt = c.get("kt", 32)
    ka = c.get("ka", 49)
    v = c["variant"]
    W = len(c["windows"])
    zero = v in ("C1+C2", "C1+zero")
    interval = v in ("C1+C2", "+interval")
    amount = v in ("C1+C2", "+amount")
    ceil = lambda x: (x + 7) // 8
    lt = shape(kt - 1)[1]
    la = shape(ka - 1)[1]
    offsets = {x for L, U in c["windows"] for x in (L, U + 1)}
    payload = 0 if zero else ((ceil(kt * P) if interval else 2 * ceil(kt * P)) + 2 * ceil(ka * P))
    payload += ceil(((len(offsets) if interval else 2) * lt + 2 * la) * P)
    payload += ceil(W * ((1 if interval else 2) + (1 if amount else 2)) * P + 2)
    if c.get("output") == "alarm":
        payload += 4 * W + ceil(13 * W)
    return payload


def compact_payload(c):
    N = sum(c["rows"])
    ki = c["kin"]
    ko = c["kout"]
    K = ki + ko
    M = N + K
    kp = c.get("kp", max(22, (N + max(ki, ko)).bit_length() + 1))
    lt = shape(kp - 1)[1]
    ceil = lambda x: (x + 7) // 8
    return sum(
        [
            ceil(10 * N),
            ceil(4 * N),
            2 * ceil(kp * (N + 1)),
            ceil(2 * lt * (N + 1)),
            ceil(6 * N),
            ceil(kp * ki) + ceil(kp * ko),
            ceil(lt * K),
            ceil(K),
            8 * M,
            ceil(2 * M),
            (20 + (12 if c.get("pattern") == "P4" else 0)) * K + ceil(K),
        ]
    )


def validate_traffic(c, id, row):
    n = c["n"]
    king = c["king"]
    star = n > 2 and c["open_mode"] == "auto"
    for phase in ["compact", "nl"]:
        info = row[phase]
        payload = (
            compact_payload(c)
            if phase == "compact" and c["mode"] != "band"
            else (
                band_payload(c)
                if phase == "nl" and c["mode"] != "compact"
                else 1 if phase == "nl" and c["mode"] == "compact" else 0
            )
        )
        for peer in info["peers"]:
            other = peer["peer"]
            active = other != id and (not star or id == king or other == king)
            sent = received = payload if active else 0
            sm = rm = info["open_rounds"] if active else 0
            raw_send = raw_recv = 0
            if phase == "compact" and c["mode"] != "band" and other != id:
                table = row["compact"]["shuffle_table_bytes"]
                if id == n - 1 or other == id + 1:
                    sent += table
                    sm += 1
                    raw_send += 1
                if (id > 0 and other == id - 1) or (id < n - 1 and other == n - 1):
                    received += table
                    rm += 1
                    raw_recv += 1
            assert peer["sent_payload"] == sent and peer["received_payload"] == received, (
                phase,
                id,
                peer,
                sent,
                received,
            )
            assert peer["sent_messages"] == sm and peer["received_messages"] == rm, (
                phase,
                peer,
                sm,
                rm,
            )
            assert peer["sent_wire"] == sent + 16 * (sm - raw_send) + 8 * raw_send
            assert peer["received_wire"] == received + 16 * (rm - raw_recv) + 8 * raw_recv


def validate_dc(c, row):
    d = row["dc"]
    N = c["kin"] + 2 * c["kout"]
    h = (N - 1).bit_length() if N else 0
    if not c["kin"] or not c["kout"]:
        assert d["logical_rounds"] == 1
        return
    dt = d["sorts"][0]["depth"]
    da = max(x["depth"] for x in d["sorts"][1:])
    rounds = (
        3 * dt
        + 3 * da
        + 2 * c["n"]
        + (h - 1) * (c["n"] + 4)
        + 3
        + (2 if c.get("output") == "alarm" else 0)
    )
    assert d["logical_rounds"] == rounds, (d["logical_rounds"], rounds)

    online = d["sorts"][0]["comparisons"] * 82 + sum(x["comparisons"] for x in d["sorts"][1:]) * 90
    online += (
        2 * N * h * 135
        + 2 * N * sum(128 + 3 + ell for ell in range(1, h))
        + 2 * N * (144 + 3 + h)
        + 2
    )
    if c.get("output") == "alarm":
        online += 45
    assert d["contribution_bits"] == online, (d["contribution_bits"], online)
    from dc_cost import dc_cost

    model = dc_cost(
        c["kin"],
        c["kout"],
        c["n"],
        [x["comparisons"] for x in d["sorts"]],
        [x["depth"] for x in d["sorts"]],
        c.get("output") == "alarm",
    )
    assert d["positional_mask_bits"] == model["positional_mask_bits"]
    if "online_logical_bits" in d:
        assert d["online_logical_bits"] == model["online_bits"]

    comp = (
        d["sorts"][0]["comparisons"] * 1983 + sum(x["comparisons"] for x in d["sorts"][1:]) * 3143
    )
    pre = comp + 2 * N * h * 807 + 2 * N * (h - 1) * 96 + d["positional_mask_bits"]
    if c.get("output") == "alarm":
        pre += costs("msb_fresh", 32)[0]
    assert d["preprocessing_consumed_bits"] == pre, (d["preprocessing_consumed_bits"], pre)
    B = (N * (283 + 2 * (147 + h) + 2 * sum(35 + ell for ell in range(1, h))) + 7) // 8

    ceil = lambda bits: (bits + 7) // 8
    B = (
        8 * N
        + 3 * 9 * N
        + 3 * ceil(N)
        + 2 * (18 * N + (3 + h) * ceil(N))
        + 2 * sum(4 * N + (3 + ell) * ceil(N) for ell in range(1, h))
    )
    id = row["id"]
    n = c["n"]
    king = c["king"]
    star = n > 2 and c["open_mode"] == "auto"
    rounds = d["open_rounds"]
    chains = h + 1
    opened_payload = None
    for peer in d["peers"]:
        other = peer["peer"]
        if other == id:
            continue
        raw_sent = int(other == id + 1) + int(id == n - 1)
        raw_recv = int(id > 0 and other == id - 1) + int(id < n - 1 and other == n - 1)
        active = not star or id == king or other == king
        sent = peer["sent_payload"] - raw_sent * B
        received = peer["received_payload"] - raw_recv * B
        if active:
            if opened_payload is None:
                opened_payload = sent
            assert sent == received == opened_payload
        else:
            assert sent == received == 0
        assert peer["sent_messages"] == int(active) * rounds + raw_sent * chains
        assert peer["received_messages"] == int(active) * rounds + raw_recv * chains
        assert (
            peer["sent_wire"]
            == peer["sent_payload"] + 16 * int(active) * rounds + 8 * raw_sent * chains
        )
        assert (
            peer["received_wire"]
            == peer["received_payload"] + 16 * int(active) * rounds + 8 * raw_recv * chains
        )


def run_case(
    build,
    document,
    mode="random",
    threads=2,
    scenario="overflow",
    reuse=True,
    local_repeats=0,
    fixture=None,
    timeout=240,
    allow_abort=False,
    view_directory=None,
    network_context=None,
):
    c = document["investigation"]
    values, expected = fixture if fixture is not None else generate(c, scenario)
    build = pathlib.Path(build).resolve()
    work = pathlib.Path(
        os.environ.get(
            "WEFT_SCRATCH_ROOT", str(pathlib.Path(tempfile.gettempdir()) / "weft-operators")
        )
    )
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run-", dir=work) as tmp:
        directory = pathlib.Path(tmp)
        plan = directory / "query.json"
        plan.write_text(json.dumps(document))
        material = directory / "material"
        for id in range(c["n"]):
            (directory / f"input{id}.json").write_text(json.dumps(values[id]))
            (directory / f"input{id}.json").chmod(0o600)
        from runtime import check_plan

        check_plan(build, plan, mode)
        protocol_start_ns = time.monotonic_ns()
        dealer = subprocess.run(
            [
                str(build / "dealer_main"),
                "--plan",
                str(plan),
                "--out",
                str(material),
                "--mode",
                mode,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if dealer.returncode:
            raise RuntimeError(dealer.stderr)
        generation = json.loads(dealer.stdout)
        assert generation["network_guard_active"] and generation["online_messages"] == 0
        from runtime import checkpoint_materials

        checkpoint_materials(generation, mode)
        hosts = directory / "hosts.json"
        hosts.write_text(
            json.dumps(
                {
                    "hosts": [
                        {"host": h, "port": port}
                        for h, port in zip(
                            network_context.hosts if network_context else ["127.0.0.1"] * c["n"],
                            free_ports(c["n"]),
                        )
                    ],
                    "threads": threads,
                    "timeout_ms": min(600000, timeout * 1000),
                }
            )
        )
        executable = (
            build
            / "bench"
            / (
                "compact_bench"
                if c["mode"] == "compact"
                else "dc_bench" if c.get("operator") == "dc" else "band_bench"
            )
        )
        commands = []
        for id in range(c["n"]):
            commands.append(
                [
                    str(executable),
                    "--id",
                    str(id),
                    "--n",
                    str(c["n"]),
                    "--hosts",
                    str(hosts),
                    "--plan",
                    str(plan),
                    "--material",
                    str(material / f"party_{id}.bin"),
                    "--input",
                    str(directory / f"input{id}.json"),
                    "--local-repeats",
                    str(local_repeats),
                ]
            )
        procs = []
        rows = []
        try:
            env = os.environ.copy()
            if view_directory is not None:
                env["BJ_DEBUG_VIEW_DIR"] = str(pathlib.Path(view_directory).resolve())
            procs = [
                subprocess.Popen(
                    network_context.command(i, cmd) if network_context else cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                for i, cmd in enumerate(commands)
            ]
            deadline = protocol_start_ns / 1e9 + timeout
            for proc in procs:
                out, err = proc.communicate(timeout=max(0.1, deadline - time.monotonic()))
                if proc.returncode:
                    raise RuntimeError(err)
                rows.append(json.loads(out))
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
            for proc in procs:
                proc.wait()
        if any(row.get("aborted") for row in rows):
            assert all(
                row.get("aborted") == 1 and row.get("output") == "bottom" and "result" not in row
                for row in rows
            ), rows
            if allow_abort:
                return {
                    "configuration": c,
                    "dealer": generation,
                    "parties": rows,
                    "expected": expected,
                    "aborted": True,
                    "correct": True,
                }
            raise RuntimeError("DC comparison pool exhausted; all banks returned bottom")
        for id, row in enumerate(rows):
            if id == c["initiator"]:
                assert row["result"] == expected, (row["result"], expected)
            else:
                assert "result" not in row
            assert row["online_new_calls"] == 0, (id, "online allocations", row["online_new_calls"])
            if c["mode"] != "band":
                pre, bits = compact_cost(c)
                assert row["compact_base_preprocessing_bits"] == pre, (
                    row["compact_base_preprocessing_bits"],
                    pre,
                )
                assert row["compact"]["contribution_bits"] == bits, (
                    row["compact"]["contribution_bits"],
                    bits,
                )
                assert (
                    row["compact"]["open_rounds"] == 11
                    and row["compact"]["logical_rounds"] == 11 + c["n"]
                )
            if (
                c["mode"] != "compact"
                and c.get("operator") != "dc"
                and c.get("pattern", "P1") == "P1"
            ):
                pre, bits, r = band_cost(c)
                assert row["band_preprocessing_bits"] == pre, (row["band_preprocessing_bits"], pre)
                assert row["nl"]["contribution_bits"] == bits and row["nl"]["open_rounds"] == r, (
                    row["nl"],
                    bits,
                    r,
                )
            for phase in ["compact", "nl"]:
                info = row[phase]
                assert info["network_hops"] == info["open_rounds"] * (
                    2 if c["n"] > 2 and c["open_mode"] == "auto" else 1
                )
            if c.get("pattern", "P1") != "P1":
                from pattern_cost import validate_pattern

                validate_pattern(c, row)
            elif c.get("operator") != "dc":
                validate_traffic(c, id, row)
            else:
                validate_dc(c, row)
            assert row["material_file_bytes"] == generation["file_bytes_per_party"][id]
            assert all(
                x["sent_payload"] == 0
                for x in row["private_output"]["peers"]
                if id == c["initiator"]
            )
            if id != c["initiator"]:
                assert all(
                    x["sent_payload"] == 0
                    for x in row["private_output"]["peers"]
                    if x["peer"] != c["initiator"]
                )
        if reuse:
            attempt = subprocess.run(commands[0], capture_output=True, text=True, timeout=10)
            assert attempt.returncode and "one-time material" in attempt.stderr
        return {
            "end_to_end_seconds": (
                (rows[c["initiator"]]["output_ready_monotonic_ns"] - protocol_start_ns) / 1e9
                if "output_ready_monotonic_ns" in rows[c["initiator"]]
                else None
            ),
            "configuration": c,
            "material_mode": mode,
            "threads": threads,
            "scenario": scenario,
            "dealer": generation,
            "parties": rows,
            "expected": expected,
            "correct": True,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", default=str(pathlib.Path(__file__).resolve().parents[1] / "build"))
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--N", type=int, default=51)
    p.add_argument("--mode", choices=["band", "compact", "pipeline"], default="pipeline")
    p.add_argument("--pattern", choices=["P1", "P2", "P3", "P4"], default="P1")
    p.add_argument("--a-lo", type=int, default=1)
    p.add_argument("--a-hi", type=int, default=1 << 40)
    p.add_argument("--m", type=int, default=2)
    p.add_argument("--variants", nargs="+", default=["C1+C2"])
    p.add_argument("--material-mode", default="random")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--scenario", default="overflow")
    p.add_argument("--local-repeats", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    path = pathlib.Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = []
    with path.open("w") as f:
        writer = None
        for variant in a.variants:
            doc = configuration(
                a.n,
                a.K,
                rows=[a.N // a.n + (i < a.N % a.n) for i in range(a.n)],
                mode=a.mode,
                variant=variant,
            )
            doc["investigation"].update(pattern=a.pattern, a_lo=a.a_lo, a_hi=a.a_hi, m=a.m)
            result = run_case(
                a.build, doc, a.material_mode, a.threads, a.scenario, local_repeats=a.local_repeats
            )
            raw.append(result)
            for r in result["parties"]:
                row = dict(
                    mode=a.mode,
                    pattern=a.pattern,
                    variant=variant,
                    n=a.n,
                    N=a.N,
                    K=a.K,
                    party=r["id"],
                    threads=a.threads,
                    material_mode=a.material_mode,
                    scenario=a.scenario,
                    dealer_seconds=result["dealer"]["generation_seconds"],
                    offline_seconds=r["offline_seconds"],
                    compact_seconds=r["compact"]["seconds"],
                    nl_seconds=r["nl"]["seconds"],
                    compact_rounds=r["compact"].get("logical_rounds", 0),
                    nl_rounds=r["nl"]["open_rounds"],
                    compact_bits=r["compact"]["contribution_bits"],
                    nl_bits=r["nl"]["contribution_bits"],
                    band_preprocessing_bits=r["band_preprocessing_bits"],
                    compact_preprocessing_bits=r.get("compact_base_preprocessing_bits", 0),
                    peak_rss_kib=r["peak_rss_kib"],
                    correct=True,
                )
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
            f.flush()
            path.with_suffix(".json").write_text(json.dumps(raw, indent=2))
            print(
                f"{a.mode} {variant} n={a.n} N={a.N} K={a.K}: correct, no online new; "
                f'compact {max(r["compact"]["seconds"] for r in result["parties"]):.4f}s '
                f'NL {max(r["nl"]["seconds"] for r in result["parties"]):.4f}s',
                flush=True,
            )


if __name__ == "__main__":
    main()
