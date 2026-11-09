import os
import pathlib
import struct
import numpy as np


def write_bank(path, bank, n, columns):
    count = len(columns["sid"])
    widths = (64, 1, 64, 96, 96)
    header = bytearray(256)
    header[:8] = b"BJDATA01"
    struct.pack_into("<IIIIQII", header, 8, 1, 256, bank, n, count, 29, 40)
    offset = 256
    for j, width in enumerate(widths):
        offset = (offset + 63) // 64 * 64
        size = (count * width + 7) // 8
        struct.pack_into("<QQII", header, 64 + j * 24, offset, size, width, 0)
        offset += size
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb", buffering=0) as f:
        f.write(header)
        for j, name in enumerate(["sid", "dir", "t", "a", "cp"]):
            start = struct.unpack_from("<Q", header, 64 + j * 24)[0]
            f.write(bytes(start - f.tell()))
            values = columns[name]
            if j == 1:
                np.packbits(np.asarray(values, dtype=np.uint8), bitorder="little").tofile(f)
            elif j < 3:
                np.asarray(values, dtype="<u8").tofile(f)
            else:
                wide = np.zeros((count, 3), dtype="<u4")
                u = np.asarray(values, dtype=np.uint64)
                wide[:, 0] = u & 0xFFFFFFFF
                wide[:, 1] = u >> 32
                wide.tofile(f)
        f.flush()
        os.fsync(f.fileno())
    assert pathlib.Path(path).stat().st_size == offset
    return count
