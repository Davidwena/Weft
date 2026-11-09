#!/usr/bin/env python3

import argparse
import csv
import json
import math
import pathlib
import random
import struct
import subprocess
import tempfile
import time
from run_net_bench import free_ports


def shape(k, d=None, eq=False):
    def cost(d):
        b = (k + d - 1) // d
        return (
            b * (1 << d) + ((1 << b) - 1 if eq else (1 << (b - 1)) + (1 << b) - 2)
            if b <= 11
            else 10**30
        )

    d = d or min(range(1, 10), key=cost)
    b = (k + d - 1) // d
    return cost(d), b if eq else 2 * b - 1


def eda(k, eq):
    if eq:
        return k - 1, (k - 1).bit_length()
    if k <= 2:
        return 0, 0
    return sum(
        2 * (k - 1 - d) for d in (1 << j for j in range((k - 2).bit_length())) if d < k - 1
    ), max(0, (k - 2).bit_length())


def costs(kind, k, d=None):
    if kind.startswith("eda_"):
        g, l = eda(k, kind == "eda_eq")
        return 2 * k + 3 * g, k + 2 * g, 1 + l
    if kind in ("msb_known", "msb_fresh"):
        pre, bits = shape(k - 1, d) if k > 1 else (0, 0)
        return (
            pre + 1 + (k if kind == "msb_fresh" else 0),
            bits + (k if kind == "msb_fresh" else 0),
            (1 if k > 1 else 0) + (kind == "msb_fresh"),
        )
    if kind.startswith("lts_"):
        pre, bits = shape(k, d)
        return (
            3 * pre + 1 + (2 * k if kind == "lts_fresh" else 0),
            3 * bits + (2 * k if kind == "lts_fresh" else 0),
            1 + (kind == "lts_fresh"),
        )
    pre, bits = shape(k, d, kind in ("eq", "eq_fresh"))
    return (
        pre + (k if kind == "eq_fresh" else 0),
        bits + (k if kind == "eq_fresh" else 0),
        2 if kind == "eq_fresh" else 1,
    )


def make_plan(n, count, kinds, k, king=0, policy="auto", evaluations=1):
    inputs = [
        {"name": "x", "owner": 0, "type": "ring", "k": k, "count": count},
        {"name": "y", "owner": n - 1, "type": "ring", "k": k, "count": count},
    ]
    comparisons = [
        {
            "name": kind,
            "kind": kind,
            "k": k,
            "count": count,
            "inputs": ["x", "y"] if kind.startswith("lts_") else ["x"],
            "evaluations": evaluations if kind in ("ltbits", "msb_known", "eq") else 1,
        }
        for kind in kinds
    ]
    return {
        "version": 1,
        "n": n,
        "king": king,
        "open_mode": policy,
        "output_party": n - 1,
        "inputs": inputs,
        "comparisons": comparisons,
    }


def run_case(
    build,
    plan,
    mode="random",
    threads=2,
    local_repeats=0,
    reverse=False,
    values=None,
    reuse_check=True,
    network="loopback",
    network_context=None,
    timeout=1800,
):
    build = pathlib.Path(build).resolve()
    n = plan["n"]
    k = plan["inputs"][0]["k"]
    count = plan["inputs"][0]["count"]
    rng = random.Random(721)
    if values is None:
        values = {s["name"]: [rng.getrandbits(k) for _ in range(count)] for s in plan["inputs"]}
        boundary = [0, 1, (1 << k) - 1, 1 << (k - 1), (1 << (k - 1)) - 1, (1 << (k - 1)) + 1]
        for s in plan["inputs"]:
            for i, v in enumerate(boundary[:count]):
                values[s["name"]][i] = v
        if count > 6:
            values["x"][6] = 0
    with tempfile.TemporaryDirectory(prefix="bj-prim-") as tmp:
        path = pathlib.Path(tmp)
        pf = path / "plan.json"
        pf.write_text(json.dumps(plan))
        package = path / "material"

        for id in range(n):
            owned = {
                spec["name"]: (
                    [str(v) for v in values[spec["name"]]] if k > 64 else values[spec["name"]]
                )
                for spec in plan["inputs"]
                if spec["owner"] == id
            }
            (path / f"input{id}.json").write_text(json.dumps({"inputs": owned}))
            (path / f"input{id}.json").chmod(0o600)
        from runtime import check_plan

        check_plan(build, pf, mode)
        start_ns = time.monotonic_ns()
        d = subprocess.run(
            [str(build / "dealer_main"), "--plan", str(pf), "--out", str(package), "--mode", mode],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if d.returncode:
            raise RuntimeError(d.stderr)
        dealer = json.loads(d.stdout)
        assert dealer["network_guard_active"] and dealer["online_messages"] == 0
        from runtime import checkpoint_materials

        checkpoint_materials(dealer, mode)

        mask = None
        if any(c["kind"] == "ltbits" for c in plan["comparisons"]):
            with (package / "party_0.bin").open("rb") as f:
                header = f.read(256)
                num = struct.unpack_from("<I", header, 36)[0]
                records = [f.read(64) for _ in range(num)]
                offset = struct.unpack_from("<Q", records[1], 24)[0]
                size = struct.unpack_from("<Q", records[1], 32)[0]
                f.seek(offset)
                raw = f.read(size)
                packed = int.from_bytes(raw, "little")
                mask = [(packed >> (i * k)) & ((1 << k) - 1) for i in range(count)]
        expected = []
        for c in plan["comparisons"]:
            kind = c["kind"]
            x = values["x"]
            y = values["y"]
            if kind == "ltbits":
                bits = [
                    int(
                        (((x[i] + mask[i]) % (1 << k)) < mask[i])
                        if reverse
                        else (mask[i] < ((x[i] + mask[i]) % (1 << k)))
                    )
                    for i in range(count)
                ]
            elif "msb" in kind:
                bits = [v >> (k - 1) for v in x]
            elif kind in ("eq", "eq_fresh", "eda_eq"):
                bits = [int(v == 0) for v in x]
            else:
                bits = [int(a < b) for a, b in zip(x, y)]
            expected.extend(bits * c.get("evaluations", 1))
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
                    "timeout_ms": min(600000, timeout * 1000),
                }
            )
        )
        commands = []
        for id in range(n):
            inp = path / f"input{id}.json"
            commands.append(
                [
                    str(build / "bench/prim_bench"),
                    "--id",
                    str(id),
                    "--n",
                    str(n),
                    "--hosts",
                    str(hosts),
                    "--plan",
                    str(pf),
                    "--material",
                    str(package / f"party_{id}.bin"),
                    "--input",
                    str(inp),
                    "--result-file",
                    str(path / f"out{id}"),
                    "--local-repeats",
                    str(local_repeats),
                    "--reverse",
                    str(int(reverse)),
                ]
            )
        procs = []
        rows = []
        try:
            procs = [
                subprocess.Popen(
                    network_context.command(i, cmd) if network_context else cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for i, cmd in enumerate(commands)
            ]
            deadline = start_ns / 1e9 + timeout
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
        actual = (path / f"out{n-1}").read_bytes()
        want = bytearray((len(expected) + 7) // 8)
        for i, b in enumerate(expected):
            want[i // 8] |= b << (i % 8)
        assert actual == want, "private output mismatch"

        layers = []
        ring_payload = []
        pre = 0
        bits_total = 0
        for c in plan["comparisons"]:
            p, b, r = costs(c["kind"], k, c.get("d"))
            ev = c.get("evaluations", 1)
            if ev > 1:
                table = math.ceil(
                    (k - 1 if c["kind"] == "msb_known" else k)
                    / (
                        c.get("d")
                        or min(
                            range(1, 10),
                            key=lambda d: shape(
                                k - 1 if c["kind"] == "msb_known" else k, d, c["kind"] == "eq"
                            )[0],
                        )
                    )
                )

                d = c.get("d") or min(
                    range(1, 10),
                    key=lambda d: shape(
                        k - 1 if c["kind"] == "msb_known" else k, d, c["kind"] == "eq"
                    )[0],
                )
                p = (table * (1 << d) + (1 if c["kind"] == "msb_known" else 0)) + (
                    p - table * (1 << d) - (1 if c["kind"] == "msb_known" else 0)
                ) * ev
            pre += p * count
            bits_total += b * count * ev
            if c["kind"].startswith("eda_"):
                gates, l = eda(k, c["kind"] == "eda_eq")
                per = [k * count]
                if c["kind"] == "eda_eq":
                    m = k
                    while m > 1:
                        per.append(2 * (m // 2) * count)
                        m = (m + 1) // 2
                else:
                    per += [4 * (k - 1 - d) * count for d in (1 << j for j in range(l))]
            elif c["kind"] in ("msb_fresh", "lts_fresh", "eq_fresh"):
                per = [
                    count * k * (2 if c["kind"] == "lts_fresh" else 1),
                    (b - k * (2 if c["kind"] == "lts_fresh" else 1)) * count,
                ]
            else:
                per = [b * count * ev] if r else [0]
            while len(layers) < len(per):
                layers.append(0)
                ring_payload.append(0)
            for i, v in enumerate(per):
                is_ring = i == 0 and c["kind"] in (
                    "msb_fresh",
                    "eq_fresh",
                    "lts_fresh",
                    "eda_msb",
                    "eda_eq",
                )
                if is_ring:
                    ring_payload[i] += ((k * count + 7) // 8) * (
                        2 if c["kind"] == "lts_fresh" else 1
                    )
                else:
                    layers[i] += v
        rounds = sum(b > 0 or r > 0 for b, r in zip(layers, ring_payload))
        payload = sum((b + 7) // 8 + r for b, r in zip(layers, ring_payload))
        assert dealer["gate_bits_per_party"] == [pre] * n, (dealer["gate_bits_per_party"], pre)
        for id, row in enumerate(rows):
            assert row["online"]["contribution_bits"] == bits_total, (row["online"], bits_total)
            assert row["online"]["open_rounds"] == rounds
            assert row["online"]["network_hops"] == rounds * (
                2 if n > 2 and plan["open_mode"] == "auto" else 1
            )
            assert row["gate_preprocessing_bits"] == pre
            for peer in row["online"]["peers"]:
                active = peer["peer"] != id and (
                    n == 2
                    or plan["open_mode"] == "broadcast"
                    or id == plan["king"]
                    or peer["peer"] == plan["king"]
                )
                assert peer["sent_payload"] == (payload if active else 0), (peer, payload)
                assert peer["received_payload"] == (payload if active else 0)
                assert peer["sent_messages"] == (rounds if active else 0)
                assert peer["sent_wire"] == ((payload + 16 * rounds) if active else 0)
            if id != n - 1:
                assert "result" not in row and not (path / f"out{id}").exists()
        if reuse_check:
            retry = subprocess.run(commands[0], capture_output=True, text=True, timeout=10)
            assert retry.returncode and "one-time material" in retry.stderr
        return {
            "end_to_end_seconds": (
                (rows[n - 1]["output_ready_monotonic_ns"] - start_ns) / 1e9
                if "output_ready_monotonic_ns" in rows[n - 1]
                else None
            ),
            "plan": plan,
            "mode": mode,
            "threads": threads,
            "network": network,
            "dealer": dealer,
            "parties": rows,
            "correct": True,
            "online_bits": bits_total,
            "preprocessing_bits": pre,
            "rounds": rounds,
        }


def csv_rows(result):
    p = result["plan"]
    c = p["comparisons"][0]
    for r in result["parties"]:
        yield dict(
            system="inframework" if c["kind"].startswith("eda_") else "C1",
            protocol=c["kind"],
            k=c["k"],
            count=c["count"],
            n=p["n"],
            threads=result["threads"],
            network=result["network"],
            material_mode=result["mode"],
            id=r["id"],
            king=r["king"],
            open_mode=r["open_mode"],
            online_seconds=r["online_seconds"],
            throughput=c["count"] / r["online_seconds"] if r["online_seconds"] else 0,
            online_bits=result["online_bits"],
            sent_payload=sum(v["sent_payload"] for v in r["online"]["peers"]),
            received_payload=sum(v["received_payload"] for v in r["online"]["peers"]),
            open_rounds=r["online"]["open_rounds"],
            network_hops=r["online"]["network_hops"],
            preprocessing_bits=result["preprocessing_bits"],
            dealer_seconds=result["dealer"]["generation_seconds"],
            offline_local_seconds=r["offline_local_seconds"],
            peak_rss_kib=r["peak_rss_kib"],
            correct=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", default=str(pathlib.Path(__file__).resolve().parents[1] / "build"))
    p.add_argument(
        "--protocols",
        nargs="+",
        default=["msb_known", "msb_fresh", "eq", "lts_known", "lts_fresh", "eda_msb", "eda_eq"],
    )
    p.add_argument("--widths", type=int, nargs="+", default=[32, 64])
    p.add_argument("--counts", type=int, nargs="+", default=[1024])
    p.add_argument("--parties", type=int, nargs="+", default=[3])
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--mode", default="random")
    p.add_argument("--open-mode", default="auto")
    p.add_argument("--network", default="loopback")
    p.add_argument("--local-repeats", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = pathlib.Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = []
    with out.open("w") as f:
        writer = None
        for n in a.parties:
            for k in a.widths:
                for count in a.counts:
                    for kind in a.protocols:
                        result = run_case(
                            a.build,
                            make_plan(n, count, [kind], k, policy=a.open_mode),
                            a.mode,
                            a.threads,
                            a.local_repeats,
                            network=a.network,
                        )
                        raw.append(result)
                        for row in csv_rows(result):
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=row)
                                writer.writeheader()
                            writer.writerow(row)
                        f.flush()
                        print(
                            f'{kind} k={k} count={count} n={n}: correct, {max(r["online_seconds"] for r in result["parties"])*1000:.3f} ms',
                            flush=True,
                        )
                        out.with_suffix(".json").write_text(json.dumps(raw, indent=2))


if __name__ == "__main__":
    main()
