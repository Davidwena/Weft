import json
import os
import pathlib
import struct
import tempfile
import shutil

WIDTHS = (64, 1, 64, 96, 96)
NAMES = ("sid", "dir", "t", "a", "cp")
MAGIC = b"BJDATA01"


def private_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


class BankWriter:
    def __init__(self, directory, bank, n, bt=29, ba=40):
        self.directory = pathlib.Path(directory)
        self.bank = bank
        self.n = n
        self.bt = bt
        self.ba = ba
        self.count = 0
        self.bit = 0
        self.byte = 0
        self.work = pathlib.Path(tempfile.mkdtemp(prefix=f".bank{bank}-", dir=directory))
        self.files = [open(self.work / name, "w+b", buffering=1024 * 1024) for name in NAMES]

    def append(self, sid, direction, t, amount, cp):
        if (
            not 0 < sid < 1 << 64
            or direction not in (0, 1)
            or not 0 <= t < 1 << self.bt
            or not 1 <= amount < 1 << self.ba
            or not 0 <= cp < 1 << 64
        ):
            raise ValueError("row outside sid/time/amount/counterparty range")
        self.files[0].write(struct.pack("<Q", sid))
        self.byte |= direction << self.bit
        self.bit += 1
        if self.bit == 8:
            self.files[1].write(bytes([self.byte]))
            self.bit = 0
            self.byte = 0
        self.files[2].write(struct.pack("<Q", t))
        self.files[3].write(amount.to_bytes(12, "little"))
        self.files[4].write(cp.to_bytes(12, "little"))
        self.count += 1

    def finish(self):
        if self.bit:
            self.files[1].write(bytes([self.byte]))
            self.bit = 0
        path = self.directory / f"bank_{self.bank}.bjcol"
        header = bytearray(256)
        header[:8] = MAGIC
        struct.pack_into(
            "<IIIIQII", header, 8, 1, 256, self.bank, self.n, self.count, self.bt, self.ba
        )
        offset = 256
        for j, width in enumerate(WIDTHS):
            offset = (offset + 63) // 64 * 64
            size = (self.count * width + 7) // 8
            struct.pack_into("<QQII", header, 64 + 24 * j, offset, size, width, 0)
            offset += size
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(header)
            for j, f in enumerate(self.files):
                start = struct.unpack_from("<Q", header, 64 + 24 * j)[0]
                out.write(bytes(start - out.tell()))
                f.flush()
                f.seek(0)
                shutil.copyfileobj(f, out, 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        for f in self.files:
            f.close()
        shutil.rmtree(self.work)
        return {"bank": self.bank, "file": path.name, "rows": self.count}

    def abort(self):
        for f in self.files:
            f.close()
        shutil.rmtree(self.work, ignore_errors=True)


def prepare_directory(path):
    path = pathlib.Path(path)
    if path.exists():
        raise FileExistsError(f"output must be a new directory: {path}")
    path.mkdir(parents=True, mode=0o700)
    os.chmod(path, 0o700)
    (path / "private").mkdir(mode=0o700)
    return path


def default_query(rows, K=512, operator="dc", bt=29, ba=40):
    n = len(rows)
    return {
        "multi": {
            "common": {
                "n": n,
                "initiator": n - 1,
                "king": 0,
                "open_mode": "auto",
                "rows": rows,
                "kin": K,
                "kout": K,
                "bt": bt,
                "ba": ba,
                "windows": [[0, min(1800, (1 << bt) - 1)]],
            },
            "templates": [
                {
                    "name": "P1",
                    "pattern": "P1",
                    "window": [0, min(1800, (1 << bt) - 1)],
                    "operator": operator,
                },
                {
                    "name": "P2",
                    "pattern": "P2",
                    "window": [0, min(1800, (1 << bt) - 1)],
                    "operator": operator,
                },
                {
                    "name": "P3",
                    "pattern": "P3",
                    "window": [0, min(1800, (1 << bt) - 1)],
                    "a_lo": 1,
                    "a_hi": 1 << ba,
                    "m": 3,
                    "operator": operator,
                },
                {
                    "name": "P4",
                    "pattern": "P4",
                    "window": [0, min(1800, (1 << bt) - 1)],
                    "operator": operator,
                },
            ],
        }
    }


def finish_dataset(path, banks, source, subject, private, bt=29, ba=40, K=512, query=None):
    path = pathlib.Path(path)
    private_json(
        path / "metadata.json",
        {
            "format": "BJDATA01",
            "version": 1,
            "n": len(banks),
            "bt": bt,
            "ba": ba,
            "columns": dict(zip(NAMES, WIDTHS)),
            "banks": banks,
            "source": source,
            "identity": (
                "entity"
                if source.get("identity") == "entity"
                else source.get("identity", "synthetic")
            ),
        },
    )
    private_json(
        path / "query.json",
        query if query is not None else default_query([b["rows"] for b in banks], K, bt=bt, ba=ba),
    )
    private_json(path / "private/subject.json", {"subject": subject})
    private_json(path / "private/provenance.json", private)
    private_json(path / "READY", {"format": "BJDATA01", "banks": len(banks)})
