import contextlib
import fcntl
import json
import os
import pathlib
import signal
import subprocess
import time

PROFILE = {
    "lan": ("100us", "10gbit", 0.1, 10000),
    "wan": ("10ms", "1gbit", 10, 1000),
    "low-wan": ("10ms", "100mbit", 10, 100),
}


def privacy_proof():

    rows = [
        list(map(int, line.split()))
        for line in pathlib.Path("/proc/self/uid_map").read_text().splitlines()
    ]
    if len(rows) != 1 or rows[0][0] != 0 or rows[0][1] == 0 or rows[0][2] != 1:
        raise RuntimeError("Require a fresh --user --map-root-user namespace")
    with open("/proc/self/ns/net", "rb") as ns:
        owner = fcntl.ioctl(ns.fileno(), 0xB701)
        try:
            a = os.fstat(owner)
            b = os.stat("/proc/self/ns/user")
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                raise RuntimeError(
                    "Network namespace belongs to another user namespace; refusing mutations"
                )
        finally:
            os.close(owner)
    return {
        "uid_map": rows,
        "net_namespace": os.readlink("/proc/self/ns/net"),
        "user_namespace": os.readlink("/proc/self/ns/user"),
        "net_owner_is_current_private_user_namespace": True,
    }


def checked(args, timeout=15):
    p = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"{args}: {p.stderr}")
    return p.stdout


class NetemNetwork:
    def __init__(self, n, profile, threads=16):
        if not 2 <= n <= 128 or profile not in PROFILE:
            raise ValueError("network dimensions/profile")
        self.privacy = privacy_proof()
        self.n = n
        self.profile = profile
        self.holders = []
        self.hosts = [f"10.203.0.{10+i}" for i in range(n)]
        self.cpus = sorted(os.sched_getaffinity(0))
        self.bindings = [
            sorted({self.cpus[(i * threads + j) % len(self.cpus)] for j in range(threads)})
            for i in range(n)
        ]
        self.closed = False
        try:
            checked(["ip", "link", "set", "lo", "up"])
            checked(["ip", "link", "add", "weftbr0", "type", "bridge"])
            checked(["ip", "addr", "add", "10.203.0.1/24", "dev", "weftbr0"])
            checked(["ip", "link", "set", "weftbr0", "up"])
            for i in range(n):
                process = subprocess.Popen(
                    ["unshare", "--net", "--", "sleep", "10000000"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                self.holders.append(process)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(process.stderr.read().decode())
                    if os.readlink(f"/proc/{process.pid}/ns/net") != os.readlink(
                        "/proc/self/ns/net"
                    ):
                        break
                    time.sleep(0.01)
                else:
                    raise RuntimeError("child netns startup timeout")
                outer = f"wp{i}"
                peer = f"wb{i}"
                checked(["ip", "link", "add", outer, "type", "veth", "peer", "name", peer])
                checked(["ip", "link", "set", peer, "netns", process.pid])
                checked(["ip", "link", "set", outer, "master", "weftbr0"])
                checked(["ip", "link", "set", outer, "up"])
                self.inside(i, ["ip", "link", "set", peer, "name", "eth0"])
                self.inside(i, ["ip", "addr", "add", self.hosts[i] + "/24", "dev", "eth0"])
                self.inside(i, ["ip", "link", "set", "eth0", "up"])
                self.inside(i, ["ip", "link", "set", "lo", "up"])
                delay, rate, _, _ = PROFILE[profile]
                self.inside(
                    i,
                    [
                        "tc",
                        "qdisc",
                        "replace",
                        "dev",
                        "eth0",
                        "root",
                        "netem",
                        "limit",
                        "100000",
                        "delay",
                        delay,
                        "rate",
                        rate,
                    ],
                )
        except BaseException:
            self.close()
            raise

    def command(self, party, args, bind=True):
        if self.closed or not 0 <= party < self.n:
            raise ValueError("closed/invalid network party")
        prefix = ["nsenter", "--target", str(self.holders[party].pid), "--net", "--"]
        if bind:
            prefix += ["taskset", "-c", ",".join(map(str, self.bindings[party]))]
        from runtime import bank_command

        return prefix + bank_command(party, args)

    def inside(self, party, args, **kwargs):
        return checked(self.command(party, args, bind=False), **kwargs)

    def counters(self):
        return [
            json.loads(self.inside(i, ["tc", "-s", "-j", "qdisc", "show", "dev", "eth0"]))[0]
            for i in range(self.n)
        ]

    def drain(self, timeout=3):
        deadline = time.monotonic() + timeout
        while True:
            rows = self.counters()
            if all(r.get("backlog", 0) == 0 and r.get("qlen", 0) == 0 for r in rows):
                return rows
            if time.monotonic() >= deadline:
                raise RuntimeError("private qdisc did not drain")
            time.sleep(0.01)

    def metadata(self):
        return {
            "topology": "one private namespace and independent egress netem veth per bank, connected by a private bridge",
            "profile": self.profile,
            "target_oneway_ms": PROFILE[self.profile][2],
            "target_per_bank_mbps": PROFILE[self.profile][3],
            "outer_namespace": os.readlink("/proc/self/ns/net"),
            "privacy_proof": self.privacy,
            "cpu_bindings": self.bindings,
            "allowed_cpus": self.cpus,
            "nodes": [
                {
                    "party": i,
                    "pid": p.pid,
                    "namespace": os.readlink(f"/proc/{p.pid}/ns/net"),
                    "host": self.hosts[i],
                    "qdisc": json.loads(
                        self.inside(i, ["tc", "-s", "-j", "qdisc", "show", "dev", "eth0"])
                    ),
                    "links": json.loads(self.inside(i, ["ip", "-j", "addr"])),
                }
                for i, p in enumerate(self.holders)
            ],
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        for p in self.holders:
            if p.poll() is None:
                p.terminate()
        for p in self.holders:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            if p.stderr:
                p.stderr.close()

        for i in range(self.n):
            subprocess.run(["ip", "link", "del", f"wp{i}"], capture_output=True)
        subprocess.run(["ip", "link", "del", "weftbr0"], capture_output=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
