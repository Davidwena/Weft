import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def bank_command(party, args):
    return ["env", "OMP_WAIT_POLICY=PASSIVE", "GOMP_SPINCOUNT=0", *map(str, args)]


def checkpoint_materials(report, mode=None):
    if not report.get("network_guard_active") or report.get("online_messages") != 0:
        raise RuntimeError("Dealer isolation check failed")
    if report["generation_seconds"] < 0:
        raise RuntimeError("Invalid generation time")


def capture(command, timeout=3600, **kwargs):
    result = subprocess.run(
        list(map(str, command)),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        **kwargs,
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, indent=2)
        stream.write("\n")


def memory_available():
    lines = Path("/proc/meminfo").read_text().splitlines()
    return next(int(line.split()[1]) * 1024 for line in lines if line.startswith("MemAvailable:"))


def check_capacity(sizes, workspace, native_bytes=0):
    total = sum(sizes)
    if total > 0.8 * shutil.disk_usage(workspace).free:
        raise MemoryError("Insufficient RAM filesystem space for fresh materials")
    if max(8 * total, total + native_bytes) + (1 << 30) > 0.8 * memory_available():
        raise MemoryError("Insufficient available memory for this configuration")


def check_plan(build, plan, mode):
    rows = capture([Path(build) / "apps/plan_info", plan])["parties"]
    sizes = [row["file_bytes"] for row in rows if row["mode"] == mode]
    check_capacity(sizes, Path(plan).parent)


@contextlib.contextmanager
def ram_workspace(base="/dev/shm"):
    base = Path(base).resolve()
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mount = Path(left.split()[4].replace("\\040", " "))
        if base == mount or mount in base.parents:
            mounts.append((len(str(mount)), right.split()[0]))
    if not mounts or max(mounts)[1] != "tmpfs":
        raise ValueError("--scratch must refer to a tmpfs filesystem")
    with tempfile.TemporaryDirectory(prefix="weft-", dir=base) as directory:
        original = tempfile.tempdir
        previous = os.environ.get("WEFT_SCRATCH_ROOT")
        tempfile.tempdir = directory
        os.environ["WEFT_SCRATCH_ROOT"] = directory
        try:
            yield Path(directory)
        finally:
            tempfile.tempdir = original
            if previous is None:
                os.environ.pop("WEFT_SCRATCH_ROOT", None)
            else:
                os.environ["WEFT_SCRATCH_ROOT"] = previous
