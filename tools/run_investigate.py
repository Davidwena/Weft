#!/usr/bin/env python3

import argparse
import copy
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
from bank_data import private_json
from run_net_bench import free_ports


def checked(command, timeout):
    r = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"{pathlib.Path(str(command[0])).name}: {r.stderr}")
    return json.loads(r.stdout)


def run(
    build,
    data,
    query=None,
    subject_file=None,
    K=None,
    operator="plan",
    threads=2,
    verify=False,
    private_inputs=None,
    timeout=600,
    keep_work=False,
    budget_bits=(1 << 64) - 1,
    latency_ms=0.1,
    bandwidth_mbps=1000,
    network_context=None,
    material_mode="prg",
    expected=None,
):
    build = pathlib.Path(build).resolve()
    data = pathlib.Path(data).resolve()
    meta = json.loads((data / "metadata.json").read_text())
    if not (data / "READY").is_file() or meta.get("format") != "BJDATA01":
        raise ValueError("dataset package is not ready")
    doc = json.loads(pathlib.Path(query or data / "query.json").read_text())
    c = doc["multi"]["common"]
    n = meta["n"]
    c.update(n=n, rows=[bank["rows"] for bank in meta["banks"]])
    c["bt"] = meta["bt"]
    c["ba"] = meta["ba"]
    if K is not None:
        c.update(kin=K, kout=K)
    if operator in ("nl", "dc"):
        for q in doc["multi"]["templates"]:
            q["operator"] = operator
    root = pathlib.Path(
        os.environ.get(
            "WEFT_SCRATCH_ROOT", str(pathlib.Path(tempfile.gettempdir()) / "weft-investigations")
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(tempfile.mkdtemp(prefix="inv-", dir=root))
    os.chmod(work, 0o700)
    binary = build / "apps/investigate"
    subject_path = pathlib.Path(subject_file or data / "private/subject.json")
    try:
        qfile = work / "query.json"
        private_json(qfile, doc)
        model = None
        if operator == "auto":
            model = checked(
                [
                    binary,
                    "--model",
                    qfile,
                    "--budget-bits",
                    budget_bits,
                    "--latency-ms",
                    latency_ms,
                    "--bandwidth-mbps",
                    bandwidth_mbps,
                ],
                timeout,
            )
            if model["selected_mask"] is None:
                raise ValueError("no operator combination fits preprocessing budget")
            mask = model["selected_mask"]
            for i, q in enumerate(doc["multi"]["templates"]):
                q["operator"] = "dc" if (mask >> i) & 1 else "nl"
            qfile.write_text(json.dumps(doc))
        plan = work / "plan.json"
        compiled = checked([binary, "--compile", qfile, "--out", plan], timeout)
        subject = json.loads(subject_path.read_text())["subject"]
        init = compiled["canonical_plan"]["multi"]["common"]["initiator"]
        inputs = work / "inputs"
        inputs.mkdir(mode=0o700)
        for bank in range(n):
            values = {"inputs": {}}
            if private_inputs:
                values = json.loads(
                    (pathlib.Path(private_inputs) / f"input_{bank}.json").read_text()
                )
            if bank == init:
                values["inputs"]["subject"] = [subject]
            private_json(inputs / f"input_{bank}.json", values)
        reference = {"expected": expected} if expected is not None else None
        if verify and reference is None:
            reference = checked(
                [
                    build / "apps/plain_reference",
                    "--data",
                    data,
                    "--query",
                    qfile,
                    "--subject-file",
                    subject_path,
                    "--inputs",
                    inputs,
                ],
                timeout,
            )
        from runtime import check_plan

        check_plan(build, plan, material_mode)
        protocol_start_ns = time.monotonic_ns()
        protocol_start = protocol_start_ns / 1e9
        package = work / "material"
        dealer = checked(
            [build / "dealer_main", "--plan", plan, "--out", package, "--mode", material_mode],
            timeout,
        )
        if not dealer["network_guard_active"] or dealer["online_messages"] != 0:
            raise AssertionError("dealer network guard")
        from runtime import checkpoint_materials

        checkpoint_materials(dealer, material_mode)
        hosts = work / "hosts.json"
        private_json(
            hosts,
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
            },
        )
        commands = []
        processes = []
        deadline = protocol_start + timeout
        bank_start_ns = time.monotonic_ns()
        try:
            for bank in range(n):
                command = list(
                    map(
                        str,
                        [
                            binary,
                            "--query",
                            qfile,
                            "--plan",
                            plan,
                            "--material",
                            package / f"party_{bank}.bin",
                            "--input",
                            inputs / f"input_{bank}.json",
                            "--bank-file",
                            data / f"bank_{bank}.bjcol",
                            "--hosts",
                            hosts,
                            "--id",
                            bank,
                            "--n",
                            n,
                        ],
                    )
                )
                commands.append(command)
                processes.append(
                    subprocess.Popen(
                        network_context.command(bank, command) if network_context else command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            reports = []
            for process in processes:
                out, error = process.communicate(timeout=max(0.1, deadline - time.monotonic()))
                if process.returncode:
                    raise RuntimeError(error)
                reports.append(json.loads(out))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
            for process in processes:
                process.wait()
        protocol_wall_seconds = time.monotonic() - protocol_start
        if any(r.get("aborted") for r in reports):
            assert all(
                r.get("aborted") == 1 and r.get("output") == "bottom" and "result" not in r
                for r in reports
            )
            return {
                "configuration": doc,
                "dealer": dealer,
                "parties": reports,
                "aborted": True,
                "verified": False,
                "protocol_wall_seconds": protocol_wall_seconds,
                "reference": reference,
            }
        for bank, report in enumerate(reports):
            assert report["online_new_calls"] == 0
            assert report["logical_share_bits"] == dealer["logical_share_bits_per_party"][bank]
            if bank == init:
                if reference:
                    assert report["result"] == reference["expected"], (
                        report["result"],
                        reference["expected"],
                    )
            else:
                assert "result" not in report
            stages = report["stages"]
            assert stages["input"]["open_rounds"] == stages["private_output"]["open_rounds"] == 0
            assert (
                sum(
                    stages[name]["open_rounds"] for name in ["filter_union", "compact", "operators"]
                )
                == report["open_rounds"]
            )
            for field in ["sent_payload_bytes", "received_payload_bytes"]:

                observed = sum(
                    stages[name][field]
                    for name in ["filter_union", "compact", "operators", "private_output"]
                )
                raw = sum(
                    peer["sent_payload" if field == "sent_payload_bytes" else "received_payload"]
                    for peer in report["network"]["peers"]
                )
                assert observed == raw, (field, observed, raw)
        retry = subprocess.run(commands[0], capture_output=True, text=True, timeout=10)
        assert retry.returncode and "one-time material" in retry.stderr
        result = {
            "configuration": doc,
            "dataset_metadata": meta,
            "compiled": compiled,
            "dealer": dealer,
            "parties": reports,
            "reference": reference,
            "verified": bool(reference),
            "correct": True if reference else None,
            "material_reuse_rejected": True,
            "network": network_context.metadata() if network_context else "unshaped loopback TCP",
            "model": model,
            "end_to_end_seconds": (
                (reports[init]["output_ready_monotonic_ns"] - protocol_start_ns) / 1e9
                if "output_ready_monotonic_ns" in reports[init]
                else None
            ),
            "protocol_wall_seconds": protocol_wall_seconds,
            "protocol_wall_scope": (
                "dealer generation through all bank exits; excludes fixture generation, "
                "plan compilation, trusted reference and post-run validation"
            ),
        }
        result["bank_start_monotonic_ns"] = bank_start_ns
        if keep_work:
            result["private_work_directory"] = str(work)
        return result
    finally:
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", default=str(pathlib.Path(__file__).resolve().parents[1] / "build"))
    p.add_argument("--data", required=True)
    p.add_argument("--query")
    p.add_argument("--subject-file")
    p.add_argument("--K", type=int)
    p.add_argument("--operator", choices=["plan", "nl", "dc", "auto"], default="plan")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--private-inputs")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--keep-work", action="store_true")
    p.add_argument("--budget-bits", type=int, default=(1 << 64) - 1)
    p.add_argument("--latency-ms", type=float, default=0.1)
    p.add_argument("--bandwidth-mbps", type=float, default=1000)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    r = run(
        a.build,
        a.data,
        a.query,
        a.subject_file,
        a.K,
        a.operator,
        a.threads,
        a.verify,
        a.private_inputs,
        a.timeout,
        a.keep_work,
        a.budget_bits,
        a.latency_ms,
        a.bandwidth_mbps,
    )
    out = pathlib.Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.write_text(json.dumps(r, indent=2) + "\n")
        os.chmod(out, 0o600)
    else:
        private_json(out, r)
    print(
        json.dumps(
            {
                "verified": r["verified"],
                "result": next(p["result"] for p in r["parties"] if "result" in p),
                "max_online_seconds": max(p["online_seconds"] for p in r["parties"]),
                "max_rss_kib": max(p["peak_rss_kib"] for p in r["parties"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
