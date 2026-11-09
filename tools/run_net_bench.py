#!/usr/bin/env python3

import argparse
import json
import pathlib
import socket
import statistics
import subprocess
import tempfile
import time


def free_ports(n):
    sockets = []
    try:
        for _ in range(n):
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            sockets.append(s)
        return [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()


def validate_parties(rows):
    n = len(rows)
    assert [r["id"] for r in rows] == list(range(n))
    for i, row in enumerate(rows):
        assert row["payload_verified"] and row["n"] == n
        it, size = row["iterations"], row["share_bytes"]
        star = row["mode"] == "open" and row["effective_open_mode"] == "king"
        assert row["open_rounds"] == (it if row["mode"] == "open" else 0)
        assert row["network_hops"] == (it * (2 if star else 1) if row["mode"] == "open" else 0)
        assert row["exchange_rounds"] == (it if row["mode"] == "exchange" else 0)
        assert row["contribution_bits"] == (it * row["share_bits"] if row["mode"] == "open" else 0)
        assert row["receive_scratch_bytes"] == (n - 1) * 65536
        for p, peer in enumerate(row["peers"]):
            connected = p != i and (not star or i == row["king"] or p == row["king"])
            messages = it if connected else 0
            assert peer["sent_payload"] == peer["received_payload"] == messages * size, row
            assert peer["sent_messages"] == peer["received_messages"] == messages, row
            header = 16 if row["mode"] == "open" else 8
            assert peer["sent_wire"] == peer["received_wire"] == messages * (size + header), row
        if star and i != row["king"]:
            assert row["reduced_chunks"] == 0


def run(
    build,
    n,
    size,
    iterations=5,
    warmup=2,
    mode="open",
    binary="bench/net_bench",
    open_mode="auto",
    king=0,
    kind="ring64",
):
    with tempfile.TemporaryDirectory(prefix="bj-net-") as d:
        config = pathlib.Path(d) / "hosts.json"
        config.write_text(
            json.dumps(
                {
                    "hosts": [{"host": "127.0.0.1", "port": p} for p in free_ports(n)],
                    "threads": 1,
                    "timeout_ms": 30000,
                }
            )
        )
        procs = []
        try:
            for i in range(n):
                procs.append(
                    subprocess.Popen(
                        [
                            str(pathlib.Path(build).resolve() / binary),
                            "--id",
                            str(i),
                            "--n",
                            str(n),
                            "--hosts",
                            str(config),
                            "--bytes",
                            str(size),
                            "--iterations",
                            str(iterations),
                            "--warmup",
                            str(warmup),
                            "--mode",
                            mode,
                            "--open-mode=" + open_mode,
                            "--king",
                            str(king),
                            "--kind",
                            kind,
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            deadline = time.monotonic() + 180
            rows = []
            for proc in procs:
                stdout, stderr = proc.communicate(timeout=max(0.1, deadline - time.monotonic()))
                if proc.returncode:
                    raise RuntimeError(f"party exited {proc.returncode}: {stderr}")
                rows.append(json.loads(stdout))
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
            for proc in procs:
                proc.wait()
    validate_parties(rows)
    it = rows[0]["iterations"]
    elapsed = max(r["seconds"] for r in rows)
    total = sum(p["sent_payload"] for r in rows for p in r["peers"])
    root = rows[king]
    sent = sum(p["sent_payload"] for p in root["peers"])
    received = sum(p["received_payload"] for p in root["peers"])
    critical = [max(r["iteration_seconds"][i] for r in rows) for i in range(it)]
    return {
        "n": n,
        "share_bytes": rows[0]["share_bytes"],
        "share_bits": rows[0]["share_bits"],
        "kind": kind,
        "iterations": it,
        "mode": mode,
        "open_mode": open_mode,
        "effective_open_mode": rows[0]["effective_open_mode"],
        "king": king,
        "wall_seconds_max": elapsed,
        "amortized_round_ms": elapsed / it * 1000,
        "median_max_local_duration_ms": statistics.median(critical) * 1000,
        "aggregate_payload_Gbps": total * 8 / elapsed / 1e9,
        "king_sent_payload": sent,
        "king_received_payload": received,
        "king_send_Gbps": sent * 8 / elapsed / 1e9,
        "king_receive_Gbps": received * 8 / elapsed / 1e9,
        "king_combined_Gbps": (sent + received) * 8 / elapsed / 1e9,
        "total_sent_payload": total,
        "parties": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[2, 3, 4, 8])
    ap.add_argument("--bytes", type=int, default=67108864)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--mode", choices=["open", "exchange", "pingpong"], default="open")
    ap.add_argument("--open-mode", choices=["auto", "broadcast"], default="auto")
    ap.add_argument("--king", type=int, default=0)
    ap.add_argument(
        "--kind",
        choices=["bits", "ring22", "ring32", "ring49", "ring64", "mixed"],
        default="ring64",
    )
    ap.add_argument("--output")
    args = ap.parse_args()
    result = [
        run(
            args.build,
            n,
            args.bytes,
            args.iterations,
            args.warmup,
            args.mode,
            open_mode=args.open_mode,
            king=args.king,
            kind=args.kind,
        )
        for n in args.n
    ]
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "parties"} for r in result], indent=2))


if __name__ == "__main__":
    main()
