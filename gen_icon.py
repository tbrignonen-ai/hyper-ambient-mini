#!/usr/bin/env python3
"""Generate icon.ico: the hot-pink dot (matches the tray icon), pure stdlib.

ICO container with embedded PNGs (256/48/32/16 px), Windows Vista+.
"""
import struct
import zlib


def make_png(size: int) -> bytes:
    r = size * 0.42
    cx = cy = (size - 1) / 2
    raw = bytearray()
    for y in range(size):
        raw.append(0)                        # filter: none
        for x in range(size):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            if d2 <= (r * 0.92) ** 2:
                # vertical two-tone: lighter pink top, hot pink bottom
                t = y / max(size - 1, 1)
                col = (int(0xff + (0xd8 - 0xff) * t),
                       int(0x5c + (0x1b - 0x5c) * t),
                       int(0x9d + (0x60 - 0x9d) * t))
                px = bytes(col) + b"\\xff"
            elif d2 <= r * r:
                px = b"\\xd8\\x1b\\x60\\xff"  # rim
            else:
                px = b"\\x00\\x00\\x00\\x00"  # transparent
            raw.extend(px)
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (b"\\x89PNG\\r\\n\\x1a\\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b""))


def make_icon(path: str) -> None:
    sizes = [256, 48, 32, 16]
    pngs = [make_png(s) for s in sizes]
    out = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    for s, png in zip(sizes, pngs):
        out += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32,
                           len(png), offset)
        offset += len(png)
    out += b"".join(pngs)
    with open(path, "wb") as f:
        f.write(out)
    print(f"wrote {path} ({offset} bytes)")


if __name__ == "__main__":
    make_icon("icon.ico")
