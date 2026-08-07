"""
sevenz_stream.py — Stream-read .7z archives via system libarchive (ctypes).
No full extraction needed: yields decompressed bytes chunk-by-chunk, so the
6.7 GB user_logs archive can be aggregated without ~30 GB of temp disk.

Usage:
    from sevenz_stream import stream_7z_lines, list_7z
    for name, size in list_7z("transactions.csv.7z"): ...
    for line in stream_7z_lines("transactions.csv.7z"):  # str lines
        ...
"""
import ctypes
import ctypes.util
import io
import os

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        path = ctypes.util.find_library("archive") or "libarchive.so.13"
        la = ctypes.CDLL(path)
        la.archive_read_new.restype = ctypes.c_void_p
        la.archive_entry_pathname.restype = ctypes.c_char_p
        la.archive_entry_size.restype = ctypes.c_longlong
        la.archive_read_data.restype = ctypes.c_ssize_t
        la.archive_error_string.restype = ctypes.c_char_p
        _LIB = la
    return _LIB


class _Reader:
    def __init__(self, path):
        self.la = _lib()
        self.a = self.la.archive_read_new()
        self.la.archive_read_support_format_all(ctypes.c_void_p(self.a))
        self.la.archive_read_support_filter_all(ctypes.c_void_p(self.a))
        r = self.la.archive_read_open_filename(
            ctypes.c_void_p(self.a), path.encode(), ctypes.c_size_t(1 << 20))
        if r != 0:
            err = self.la.archive_error_string(ctypes.c_void_p(self.a))
            raise IOError(f"libarchive open failed: {err}")

    def entries(self):
        entry = ctypes.c_void_p()
        while self.la.archive_read_next_header(
                ctypes.c_void_p(self.a), ctypes.byref(entry)) == 0:
            name = self.la.archive_entry_pathname(entry).decode()
            size = self.la.archive_entry_size(entry)
            yield name, size

    def read_chunks(self, chunk_size=1 << 22):
        buf = ctypes.create_string_buffer(chunk_size)
        while True:
            n = self.la.archive_read_data(
                ctypes.c_void_p(self.a), buf, ctypes.c_size_t(chunk_size))
            if n == 0:
                return
            if n < 0:
                err = self.la.archive_error_string(ctypes.c_void_p(self.a))
                raise IOError(f"libarchive read failed: {err}")
            yield buf.raw[:n]

    def close(self):
        self.la.archive_read_free(ctypes.c_void_p(self.a))


def list_7z(path):
    r = _Reader(path)
    try:
        return list(r.entries())
    finally:
        r.close()


def stream_7z_bytes(path, member=None, chunk_size=1 << 22):
    """Yield decompressed byte chunks of the first (or named) member."""
    r = _Reader(path)
    try:
        for name, _size in r.entries():
            if member is None or name == member:
                yield from r.read_chunks(chunk_size)
                return
        raise KeyError(f"{member} not found in {path}")
    finally:
        r.close()


class SevenZFile(io.RawIOBase):
    """Read-only file-like over a .7z member (for pd.read_csv etc.)."""

    def __init__(self, path, member=None, chunk_size=1 << 22):
        self._gen = stream_7z_bytes(path, member, chunk_size)
        self._chunks = []
        self._len = 0

    def readable(self):
        return True

    def readinto(self, b):
        need = len(b)
        while self._len < need:
            try:
                c = next(self._gen)
            except StopIteration:
                break
            self._chunks.append(c)
            self._len += len(c)
        if not self._len:
            return 0
        data = b"".join(self._chunks)
        n = min(need, len(data))
        b[:n] = data[:n]
        rest = data[n:]
        self._chunks = [rest] if rest else []
        self._len = len(rest)
        return n


def open_7z_buffered(path, member=None):
    return io.BufferedReader(SevenZFile(path, member), 1 << 22)


def stream_7z_lines(path, member=None, encoding="utf-8"):
    """Yield decoded text lines (str, no trailing newline)."""
    tail = b""
    for chunk in stream_7z_bytes(path, member):
        data = tail + chunk
        lines = data.split(b"\n")
        tail = lines.pop()
        for ln in lines:
            yield ln.rstrip(b"\r").decode(encoding, "replace")
    if tail:
        yield tail.rstrip(b"\r").decode(encoding, "replace")
