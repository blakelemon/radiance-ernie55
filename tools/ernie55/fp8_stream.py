#!/usr/bin/env python3
"""Single-tensor streaming reader for the block-FP8 ERNIE checkpoint (story 5-0-iqr-v4-target).

Reads a shard's JSON header and then ONLY the byte range of the one tensor (and its `<name>_scale_inv` plane): os.pread of an offset and a length, never the shard, never a
whole-file mmap. Block FP8 e4m3 (OCP e4m3fn) with a 128 x 128 scale block; the scale plane is F32 (as this checkpoint stores the ue8m0 values) or F8_E8M0 (2^(b-127)).
BF16 / F32 tensors are returned widened to f32. `bytes_read` counts every byte pread (headers included) so a test can show the shard was not read.

  fp8_stream.py dequant --checkpoint DIR --tensor NAME --out FILE.f32 [--trace trace.json]     (f32 little-endian [N, K], written in row bands)
  fp8_stream.py info --checkpoint DIR --tensor NAME
numpy is used when present (the real run); a pure-python path serves the tests and small tensors.
"""
import argparse, json, os, struct, sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by the sealed tests under `python3 -I`
    np = None

BLOCK = 128
DT_BYTES = {"F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4}


class Fail(Exception):
    pass


def e4m3_value(b):
    s = -1.0 if b & 0x80 else 1.0
    e, m = (b >> 3) & 15, b & 7
    if e == 15 and m == 7:
        return float("nan")
    if e == 0:
        return s * (m / 8.0) * 2.0 ** -6
    return s * (1 + m / 8.0) * 2.0 ** (e - 7)


E4M3 = [e4m3_value(b) for b in range(256)]


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        self.bytes_read = 0
        self._hdr = {}
        self._fd = {}
        idx = self.root / "model.safetensors.index.json"
        self.map = json.loads(idx.read_text())["weight_map"] if idx.exists() else None

    def _open(self, shard):
        if shard not in self._fd:
            self._fd[shard] = os.open(self.root / shard, os.O_RDONLY)
        return self._fd[shard]

    def _pread(self, fd, n, off):
        b = os.pread(fd, n, off)
        if len(b) != n:
            raise Fail(f"short read ({len(b)} of {n} at {off})")
        self.bytes_read += n
        return b

    def header(self, shard):
        if shard not in self._hdr:
            fd = self._open(shard)
            n = struct.unpack("<Q", self._pread(fd, 8, 0))[0]
            if n > 256 << 20:
                raise Fail(f"{shard}: header of {n} bytes")
            self._hdr[shard] = (json.loads(self._pread(fd, n, 8)), 8 + n)
        return self._hdr[shard]

    def locate(self, name):
        if self.map is not None:
            shard = self.map.get(name)
            if shard is None:
                raise Fail(f"tensor {name} is not in the index")
            h, base = self.header(shard)
            if name not in h:
                raise Fail(f"tensor {name} is not in {shard}")
            return shard, h[name], base
        for p in sorted(self.root.glob("*.safetensors")):
            h, base = self.header(p.name)
            if name in h:
                return p.name, h[name], base
        raise Fail(f"tensor {name} not found")

    def raw(self, name, row0=0, rows=None):
        shard, t, base = self.locate(name)
        shape = t["shape"]
        if len(shape) == 0:
            raise Fail(f"{name}: a 0-d tensor has no rows")
        item = DT_BYTES[t["dtype"]]
        per_row = item
        for s in shape[1:]:
            per_row *= s
        rows = shape[0] - row0 if rows is None else rows
        a, b = t["data_offsets"]
        total = per_row * shape[0]
        if b - a != total:
            raise Fail(f"{name}: the header declares {b - a} bytes at offsets {a}..{b} but shape {shape} of {t['dtype']} is {total}")
        if row0 < 0 or rows < 0 or row0 + rows > shape[0]:
            raise Fail(f"{name}: rows {row0}..{row0 + rows} are outside the tensor's {shape[0]} rows")
        off = base + a + row0 * per_row
        return t["dtype"], shape, self._pread(self._open(shard), rows * per_row, off)

    def close(self):
        for fd in self._fd.values():
            os.close(fd)


def widen(dtype, buf, n):
    """Raw bytes -> f32 list/array for a float dtype."""
    if np is not None:
        if dtype == "F32":
            return np.frombuffer(buf, dtype="<f4").astype(np.float32)
        if dtype == "BF16":
            return (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        if dtype == "F16":
            return np.frombuffer(buf, dtype="<f2").astype(np.float32)
    else:
        if dtype == "F32":
            return list(struct.unpack(f"<{n}f", buf))
        if dtype == "BF16":
            return list(struct.unpack(f"<{n}f", b"".join(b"\0\0" + buf[i:i + 2] for i in range(0, len(buf), 2))))
    raise Fail(f"cannot widen dtype {dtype}")


def scales_of(ck, name, grid):
    sname = name + "_scale_inv"
    try:
        dt, shape, buf = ck.raw(sname)
    except Fail as e:
        raise Fail(f"{name}: no scale plane {sname} ({e})")
    if list(shape) != list(grid):
        raise Fail(f"{sname}: scale shape {shape} is not the block grid {grid}")
    n = shape[0] * shape[1]
    if dt == "F8_E8M0":
        return [float("nan") if b == 255 else 2.0 ** (b - 127) for b in buf], n
    if dt == "F32":
        return list(struct.unpack(f"<{n}f", buf)), n
    raise Fail(f"{sname}: scale dtype {dt}")


def dequant_rows(ck, name, out_fh, band=512):
    """Stream the dequantised tensor to out_fh in row bands; returns (N, K, tensor_bytes, scale_bytes)."""
    shard, t, base = ck.locate(name)
    if t["dtype"] != "F8_E4M3" or len(t["shape"]) != 2:
        raise Fail(f"{name}: {t['dtype']} {t['shape']} is not a 2-D block-FP8 weight")
    N, K = t["shape"]
    gr, gc = -(-N // BLOCK), -(-K // BLOCK)
    sc, nsc = scales_of(ck, name, [gr, gc])
    scale_bytes = ck.locate(name + "_scale_inv")[1]["data_offsets"]
    scale_bytes = scale_bytes[1] - scale_bytes[0]
    if np is not None:
        lut = np.array(E4M3, dtype=np.float32)
        srow = np.array(sc, dtype=np.float32).reshape(gr, gc)
    for r0 in range(0, N, band):
        n = min(band, N - r0)
        _, _, buf = ck.raw(name, r0, n)
        if np is not None:
            codes = np.frombuffer(buf, dtype=np.uint8).reshape(n, K)
            vals = lut[codes]
            g0, g1 = r0 // BLOCK, (r0 + n - 1) // BLOCK + 1
            s_full = np.repeat(np.repeat(srow[g0:g1], BLOCK, axis=0), BLOCK, axis=1)[r0 - g0 * BLOCK:r0 - g0 * BLOCK + n, :K]
            out_fh.write((vals * s_full).astype("<f4").tobytes())
        else:
            for i in range(n):
                gi = (r0 + i) // BLOCK
                row = buf[i * K:(i + 1) * K]
                out_fh.write(struct.pack(f"<{K}f", *[E4M3[row[j]] * sc[gi * gc + j // BLOCK] for j in range(K)]))
    return N, K, N * K, scale_bytes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dequant")
    a.add_argument("--checkpoint", required=True); a.add_argument("--tensor", required=True); a.add_argument("--out", required=True); a.add_argument("--trace")
    b = sub.add_parser("info")
    b.add_argument("--checkpoint", required=True); b.add_argument("--tensor", required=True)
    args = ap.parse_args(argv)
    ck = Checkpoint(args.checkpoint)
    try:
        if args.cmd == "info":
            shard, t, base = ck.locate(args.tensor)
            print(json.dumps({"shard": shard, "dtype": t["dtype"], "shape": t["shape"], "bytes": t["data_offsets"][1] - t["data_offsets"][0]}))
            return 0
        tmp = args.out + ".part"
        try:
            with open(tmp, "wb") as fh:
                N, K, tb, sb = dequant_rows(ck, args.tensor, fh)
            os.replace(tmp, args.out)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        if args.trace:
            Path(args.trace).write_text(json.dumps({"tensor": args.tensor, "shape": [N, K], "tensor_bytes": tb, "scale_bytes": sb, "bytes_read": ck.bytes_read}))
        return 0
    except (Fail, OSError, KeyError, ValueError) as e:
        print(f"fp8_stream: {e}", file=sys.stderr)
        return 2
    finally:
        ck.close()


if __name__ == "__main__":
    sys.exit(main())
