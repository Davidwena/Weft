import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from bank_data import default_query
from netns_v9 import NetemNetwork, privacy_proof
from run_net_bench import free_ports
from runtime import capture, check_capacity, ram_workspace, save

ROOT = Path(__file__).resolve().parents[1]
BANDWIDTH = {"lan": 1e10, "wan": 1e9, "low-wan": 1e8, "local": 1e10}


class LocalNetwork:
    def __init__(self, n):
        self.hosts = ["127.0.0.1"] * n

    def command(self, party, args):
        return list(map(str, args))

    def metadata(self):
        return {"profile": "local", "topology": "unshaped private loopback"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def network(args, n):
    return (
        LocalNetwork(n) if args.profile == "local" else NetemNetwork(n, args.profile, args.threads)
    )


def query_for(data, args):
    doc = json.loads((data / "query.json").read_text())
    doc["multi"]["common"].update(kin=args.K, kout=args.K)
    doc["multi"]["templates"] = [
        q for q in doc["multi"]["templates"] if q["pattern"] == args.pattern
    ]
    for query in doc["multi"]["templates"]:
        query["operator"] = args.operator
    return doc


def data_for(args, scratch):
    if args.data:
        path = Path(args.data).resolve()
        if not (path / "READY").is_file():
            raise ValueError("Expected a converted dataset with a READY marker")
        return path
    from synthetic_v9 import generate

    nin = args.nu_in if args.nu_in is not None else min(args.K, 256, args.N // 2)
    nout = args.nu_out if args.nu_out is not None else min(args.K, 256, args.N - nin)
    return generate(
        scratch / "synthetic",
        n=args.n,
        total=args.N,
        nu_in=nin,
        nu_out=nout,
        K=args.K,
        pattern=args.pattern,
    )


def primitives(args, scratch):
    from run_primitives import make_plan, run_case

    results = []
    with network(args, args.n) as net:
        for width in args.widths:
            for count in args.counts:
                for protocol in args.protocols:
                    plan = make_plan(args.n, count, [protocol], width)
                    results.append(
                        run_case(
                            args.build,
                            plan,
                            args.mode,
                            args.threads,
                            network=args.profile,
                            network_context=net,
                            timeout=args.timeout,
                        )
                    )
    return results


def operators(args, scratch):
    from run_investigation import configuration, run_case

    choices = ["nl", "dc"] if args.operator == "both" else [args.operator]
    if args.phase == "compact":
        choices = ["nl"]
    rows = [args.N // args.n + int(i < args.N % args.n) for i in range(args.n)]
    results = []
    with network(args, args.n) as net:
        for operator in choices:
            doc = configuration(n=args.n, kin=args.K, rows=rows, mode=args.phase)
            doc["investigation"].update(
                pattern=args.pattern, operator=operator, a_lo=1, a_hi=1 << 40, m=3
            )
            results.append(
                run_case(
                    args.build,
                    doc,
                    args.mode,
                    args.threads,
                    scenario="exact",
                    timeout=args.timeout,
                    network_context=net,
                )
            )
    return results


def investigation(args, scratch):
    from run_investigate import run

    data = data_for(args, scratch)
    query = scratch / "selected_query.json"
    save(query, query_for(data, args))
    with network(
        args, args.n if not args.data else json.loads((data / "metadata.json").read_text())["n"]
    ) as net:
        result = run(
            args.build,
            data,
            query=query,
            K=args.K,
            operator=args.operator,
            threads=args.threads,
            verify=True,
            subject_file=args.subject_file,
            timeout=args.timeout,
            network_context=net,
            material_mode=args.mode,
        )
    if result.get("aborted"):
        return result
    transfer = sum(result["dealer"]["file_bytes_per_party"]) * 8 / BANDWIDTH[args.profile]
    init = result["compiled"]["canonical_plan"]["multi"]["common"]["initiator"]
    bank_wall = (
        result["parties"][init]["output_ready_monotonic_ns"] - result["bank_start_monotonic_ns"]
    ) / 1e9
    result.update(
        ideal_transfer_seconds=transfer,
        total_excluding_persistence_seconds=result["dealer"]["generation_seconds"]
        + transfer
        + bank_wall,
        transfer_model="sum of all bank package bytes / shared dealer egress bandwidth",
    )
    return result


def dealer(args, scratch):
    results = []
    for n in args.banks:
        rows = [args.N // n + int(i < args.N % n) for i in range(n)]
        query = default_query(rows, args.K, operator=args.operator)
        query["multi"]["templates"] = [
            q for q in query["multi"]["templates"] if q["pattern"] == args.pattern
        ]
        query_path, plan = scratch / f"query-{n}.json", scratch / f"plan-{n}.json"
        save(query_path, query)
        subprocess.run(
            [
                str(args.build / "apps/investigate"),
                "--compile",
                str(query_path),
                "--out",
                str(plan),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            timeout=args.timeout,
        )
        layouts = capture([args.build / "apps/plan_info", plan])["parties"]
        for mode in args.modes:
            sizes = [r["file_bytes"] for r in layouts if r["mode"] == mode]
            for repeat in range(args.repeats):
                check_capacity(sizes, scratch)
                destination = scratch / f"material-{n}-{mode}-{repeat}"
                try:
                    report = capture(
                        [
                            args.build / "dealer_main",
                            "--plan",
                            plan,
                            "--out",
                            destination,
                            "--mode",
                            mode,
                        ],
                        timeout=args.timeout,
                    )
                    assert report["network_guard_active"] and report["online_messages"] == 0
                    assert report["file_bytes_per_party"] == sizes
                    actual_sizes = [
                        (destination / f"party_{i}.bin").stat().st_size for i in range(n)
                    ]
                    assert actual_sizes == sizes
                    transfer = sum(sizes) * 8 / BANDWIDTH[args.profile]
                    row = dict(
                        n=n,
                        mode=mode,
                        repeat=repeat,
                        generation_seconds=report["generation_seconds"],
                        ideal_transfer_seconds=transfer,
                        generation_plus_transfer_seconds=report["generation_seconds"] + transfer,
                        material_bytes_per_bank=sizes,
                        native_report=report,
                    )
                    results.append(row)
                    print(json.dumps(row), flush=True)
                finally:
                    shutil.rmtree(destination, ignore_errors=True)
    return results


def batch_data(args, scratch):
    import numpy as np
    from bank_data import finish_dataset, prepare_directory
    from column_data import write_bank

    if args.data:
        return Path(args.data).resolve(), args.subjects
    subjects = list(range(42, 106))
    path = prepare_directory(scratch / "batch_input")
    rng = np.random.default_rng(733)
    activity = []
    low, high = (50, 513) if args.K >= 512 else (1, args.K + 1)
    for j, subject in enumerate(subjects):
        nin, nout = map(int, rng.integers(low, high, size=2))
        for direction, count in [(0, nin), (1, nout)]:
            for _ in range(count):
                activity.append(
                    (
                        subject,
                        direction,
                        100000 + j * 86400 + int(rng.integers(0, 3600)),
                        150000 + int(rng.integers(-5000, 5001)),
                        7 + int(rng.integers(0, 3)),
                    )
                )
    if len(activity) > args.N:
        raise ValueError("N is too small for the synthetic batch; reduce K or increase N")
    positions = rng.choice(args.N, len(activity), replace=False)
    order = np.argsort(positions)
    positions = positions[order]
    activity = np.asarray(activity, dtype=np.uint64)[order]
    rows = [args.N // args.n + int(i < args.N % args.n) for i in range(args.n)]
    banks, offset = [], 0
    for bank, N in enumerate(rows):
        r = np.random.default_rng(733 + 100003 * (bank + 1))
        cols = {
            "sid": r.integers(1000000, 1010000, N, dtype=np.uint64),
            "dir": r.integers(0, 2, N, dtype=np.uint8),
            "t": r.integers(0, 16 * 86400, N, dtype=np.uint64),
            "a": r.integers(1, 10000000, N, dtype=np.uint64),
            "cp": r.integers(1, 10001, N, dtype=np.uint64),
        }
        selected = (positions >= offset) & (positions < offset + N)
        local = positions[selected] - offset
        for index, key in enumerate(["sid", "dir", "t", "a", "cp"]):
            cols[key][local] = activity[selected, index]
        write_bank(path / f"bank_{bank}.bjcol", bank, args.n, cols)
        banks.append(dict(bank=bank, file=f"bank_{bank}.bjcol", rows=N))
        offset += N
    finish_dataset(
        path,
        banks,
        {"kind": "synthetic", "seed": 733},
        subjects[0],
        {"suspects": subjects},
        K=args.K,
    )
    return path, subjects


def batch_reference(data, subjects, pattern, K):
    import numpy as np
    from fast_reference import aggregate

    groups = {s: [[], []] for s in subjects}
    meta = json.loads((data / "metadata.json").read_text())
    for bank in range(meta["n"]):
        cols = column_views(data / f"bank_{bank}.bjcol")
        for subject in subjects:
            for at in np.flatnonzero(cols["sid"] == subject):
                groups[subject][int(cols["dir"][at])].append(
                    tuple(int(cols[k][at]) for k in ["t", "a", "cp"])
                )
    c = dict(pattern=pattern, p=5, q=100, a_lo=1, a_hi=1 << 40, m=3)
    return [
        dict(
            aggregate=aggregate(groups[s][0][:K], groups[s][1][:K], c, 0, 1800),
            omega=int(len(groups[s][0]) > K or (pattern != "P3" and len(groups[s][1]) > K)),
        )
        for s in subjects
    ]


def column_views(path):
    import numpy as np
    import struct

    with path.open("rb") as stream:
        header = stream.read(256)
    count = struct.unpack_from("<Q", header, 24)[0]
    result = {}
    for index, name in enumerate(["sid", "dir", "t", "a", "cp"]):
        offset, size, width, _ = struct.unpack_from("<QQII", header, 64 + index * 24)
        if not count:
            result[name] = np.zeros(0, dtype=np.uint64)
        elif width == 1:
            result[name] = np.unpackbits(
                np.memmap(path, mode="r", dtype="u1", offset=offset, shape=(size,)),
                bitorder="little",
            )[:count]
        elif width == 96:
            raw = np.memmap(path, mode="r", dtype="<u4", offset=offset, shape=(count, 3))
            if raw[:, 2].any():
                raise ValueError("Plaintext batch reference supports unsigned 64-bit data values")
            result[name] = raw[:, 0].astype(np.uint64) + (raw[:, 1].astype(np.uint64) << 32)
        else:
            result[name] = np.memmap(path, mode="r", dtype="<u8", offset=offset, shape=(count,))
    return result


def batch(args, scratch):
    data, subjects = batch_data(args, scratch)
    if not subjects or len(set(subjects)) != len(subjects) or len(subjects) < max(args.suspects):
        raise ValueError("Supply enough distinct entity IDs with --subjects")
    meta = json.loads((data / "metadata.json").read_text())
    results = []
    for S in args.suspects:
        selected = subjects[:S]
        reference = batch_reference(data, selected, args.pattern, args.K)
        work = scratch / f"batch-S{S}"
        work.mkdir(mode=0o700)
        doc = {
            "num_suspects": S,
            "investigation": dict(
                n=meta["n"],
                initiator=0,
                king=0,
                open_mode="auto",
                rows=[b["rows"] for b in meta["banks"]],
                kin=args.K,
                kout=args.K,
                bt=meta["bt"],
                ba=meta["ba"],
                p=5,
                q=100,
                windows=[[0, 1800]],
                pattern=args.pattern,
                operator="nl",
                mode="pipeline",
                variant="C1+C2",
            ),
        }
        query, plan = work / "query.json", work / "plan.json"
        save(query, doc)
        command = [args.build / "bench/batch_suspects", "--query", query]
        layout = capture([*command, "--estimate", args.mode])
        native_bytes = meta["n"] * (
            sum(doc["investigation"]["rows"]) * (450 * S + 360) + S * args.K * args.K * 600
        )
        check_capacity(layout["file_bytes_per_bank"], scratch, native_bytes)
        subprocess.run(
            list(map(str, [*command, "--compile", plan])), check=True, timeout=args.timeout
        )
        dealer_report = capture(
            [
                args.build / "dealer_main",
                "--plan",
                plan,
                "--out",
                work / "material",
                "--mode",
                args.mode,
            ],
            timeout=args.timeout,
        )
        assert dealer_report["network_guard_active"] and dealer_report["online_messages"] == 0
        assert dealer_report["file_bytes_per_party"] == layout["file_bytes_per_bank"]
        n = meta["n"]
        with network(args, n) as net:
            save(
                work / "hosts.json",
                dict(
                    hosts=[dict(host=h, port=p) for h, p in zip(net.hosts, free_ports(n))],
                    threads=args.threads,
                    timeout_ms=min(1200000, args.timeout * 1000),
                ),
            )
            reports, processes = [], []
            started = time.monotonic_ns()
            try:
                for bank in range(n):
                    private = (
                        {f"suspect{j}.subject": [s] for j, s in enumerate(selected)}
                        if bank == 0
                        else {}
                    )
                    save(work / f"input{bank}.json", {"inputs": private})
                    args_bank = [
                        *command,
                        "--id",
                        bank,
                        "--hosts",
                        work / "hosts.json",
                        "--input",
                        work / f"input{bank}.json",
                        "--bank-file",
                        data / f"bank_{bank}.bjcol",
                        "--material",
                        work / f"material/party_{bank}.bin",
                    ]
                    processes.append(
                        subprocess.Popen(
                            net.command(bank, args_bank),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                    )
                deadline = time.monotonic() + args.timeout
                for process in processes:
                    stdout, stderr = process.communicate(
                        timeout=max(0.1, deadline - time.monotonic())
                    )
                    if process.returncode:
                        raise RuntimeError(stderr)
                    reports.append(json.loads(stdout))
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                for process in processes:
                    process.wait()
            assert reports[0]["results"] == reference
            assert all("results" not in report for report in reports[1:])
            for report in reports:
                assert report["slot_counts"] == [[args.K, args.K]] * S
            transfer = sum(layout["file_bytes_per_bank"]) * 8 / BANDWIDTH[args.profile]
            bank_wall = (reports[0]["output_ready_monotonic_ns"] - started) / 1e9
            result = dict(
                S=S,
                query=doc,
                dealer=dealer_report,
                banks=reports,
                expected=reference,
                correct=True,
                ideal_transfer_seconds=transfer,
                total_excluding_persistence_seconds=dealer_report["generation_seconds"]
                + transfer
                + bank_wall,
                network=net.metadata(),
            )
            results.append(result)
            print(
                json.dumps(
                    dict(
                        S=S,
                        correct=True,
                        online_seconds=reports[0]["online_seconds"],
                        total_seconds=result["total_excluding_persistence_seconds"],
                    )
                ),
                flush=True,
            )
        shutil.rmtree(work)
    return results


def multi(args, scratch):
    from run_multi import run_case

    doc = json.loads(Path(args.query).read_text())
    doc["multi"]["common"]["n"] = args.n
    if len(doc["multi"]["common"]["rows"]) != args.n:
        raise ValueError("Query row counts must match --n")
    with network(args, args.n) as net:
        return run_case(
            args.build, doc, args.mode, args.threads, timeout=args.timeout, network_context=net
        )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="experiment", required=True)
    for name in ["primitives", "operators", "investigation", "dealer", "batch", "multi"]:
        p = sub.add_parser(name)
        p.add_argument("--build", type=Path, default=ROOT / "build")
        p.add_argument("--output", type=Path)
        p.add_argument("--scratch", default="/dev/shm")
        p.add_argument("--profile", choices=list(BANDWIDTH), default="lan")
        p.add_argument("--threads", type=int, default=16)
        p.add_argument("--timeout", type=int, default=3600)
        p.add_argument("--mode", choices=["prg", "random"], default="prg")
        p.add_argument("--private-network", action="store_true", help=argparse.SUPPRESS)
        p.add_argument("--n", type=int, default=3)
        p.add_argument("--N", type=int, default=1 << 20)
        p.add_argument("--K", type=int, default=512)
        p.add_argument("--pattern", choices=["P1", "P2", "P3", "P4"], default="P1")
        if name == "primitives":
            p.add_argument("--widths", nargs="+", type=int, default=[32, 64])
            p.add_argument("--counts", nargs="+", type=int, default=[1 << 10])
            p.add_argument(
                "--protocols", nargs="+", default=["msb_known", "msb_fresh", "eq", "eq_fresh"]
            )
        if name in ["operators", "investigation", "dealer"]:
            p.add_argument(
                "--operator",
                choices=["nl", "dc", "both"] if name == "operators" else ["nl", "dc"],
                default="both" if name == "operators" else "nl",
            )
        if name == "operators":
            p.add_argument("--phase", choices=["band", "compact", "pipeline"], default="band")
        if name in ["investigation", "batch"]:
            p.add_argument("--data")
        if name == "investigation":
            p.add_argument("--subject-file")
            p.add_argument("--nu-in", type=int)
            p.add_argument("--nu-out", type=int)
        if name == "dealer":
            p.add_argument("--banks", nargs="+", type=int, default=list(range(2, 9)))
            p.add_argument(
                "--modes", nargs="+", choices=["prg", "random"], default=["prg", "random"]
            )
            p.add_argument("--repeats", type=int, default=2)
        if name == "batch":
            p.add_argument("--suspects", nargs="+", type=int, default=[1, 4, 8, 16])
            p.add_argument("--subjects", nargs="+", type=int)
        if name == "multi":
            p.add_argument("--query", default=str(ROOT / "examples/multi_query.json"))
    args = parser.parse_args()
    if (
        not 2 <= args.n <= 128
        or not 1 <= args.threads <= 128
        or args.N < 1
        or args.K < 1
        or args.timeout < 1
    ):
        parser.error("Invalid public dimensions or timeout")
    if args.experiment == "dealer" and (
        args.repeats < 1 or any(n < 2 or n > 128 for n in args.banks)
    ):
        parser.error("Invalid dealer sweep")
    if args.experiment == "batch" and any(S < 1 or S > 64 for S in args.suspects):
        parser.error("Batch sizes must be between 1 and 64")
    args.build = args.build.resolve()
    if not (args.build / "dealer_main").is_file():
        parser.error("Build the artifact first with scripts/build.sh")
    if args.output is None:
        args.output = ROOT / "outputs" / f"{args.experiment}-{time.time_ns()}.json"
    if args.output.exists():
        parser.error("Output already exists; choose a fresh --output")
    if not args.private_network:
        os.execvp(
            "unshare",
            [
                "unshare",
                "--user",
                "--map-root-user",
                "--net",
                "--",
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--private-network",
            ],
        )
    privacy_proof()
    subprocess.run(["ip", "link", "set", "lo", "up"], check=True)
    os.umask(0o077)
    os.environ.update(
        BJ_MATERIAL_DURABILITY="nondurable",
        OMP_NUM_THREADS=str(args.threads),
        OMP_WAIT_POLICY="PASSIVE",
        GOMP_SPINCOUNT="0",
        OPENBLAS_NUM_THREADS="1",
    )
    with ram_workspace(args.scratch) as scratch:
        result = globals()[args.experiment](args, scratch)
    save(
        args.output,
        dict(
            experiment=args.experiment,
            parameters={
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()
                if k != "private_network"
            },
            results=result,
        ),
    )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
