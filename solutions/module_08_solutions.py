"""
Solutions — Module 08: Memory-Efficient Design Patterns
"""
import os
import gc
import time
import pickle
import contextlib
import dataclasses
from dataclasses import dataclass
from typing import Optional
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def get_rss_mb() -> float:
    if not _HAS_PSUTIL:
        return -1.0
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


# ---------------------------------------------------------------------------
# Exercise 1 — Extended ColDescriptor with offset_bytes (single segment)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColDescriptorWithOffset:
    col_name: str
    shm_name: str
    dtype_str: str
    shape: tuple
    offset_bytes: int = 0

    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * np.dtype(self.dtype_str).itemsize

    def attach_array(self) -> tuple[SharedMemory, np.ndarray]:
        shm = SharedMemory(name=self.shm_name, create=False)
        arr = np.ndarray(
            self.shape,
            dtype=np.dtype(self.dtype_str),
            buffer=shm.buf,
            offset=self.offset_bytes,
        )
        return shm, arr


class SharedDataFramePacked:
    """
    Packs all columns into a SINGLE shared memory segment using offsets.
    Trades off simpler OS overhead for more complex offset management.
    """

    def __init__(self, df: pd.DataFrame, columns: Optional[list] = None):
        cols = columns if columns is not None else list(df.columns)
        # Compute total size and individual offsets
        arrays = {}
        offsets = {}
        offset = 0
        for col in cols:
            raw = np.ascontiguousarray(df[col].to_numpy())
            arrays[col] = raw
            offsets[col] = offset
            # Align to next 8-byte boundary
            offset += raw.nbytes
            remainder = offset % 8
            if remainder != 0:
                offset += 8 - remainder

        self._shm = SharedMemory(create=True, size=max(offset, 1))
        self._descriptors: list[ColDescriptorWithOffset] = []

        for col in cols:
            raw = arrays[col]
            off = offsets[col]
            view = np.ndarray(raw.shape, dtype=raw.dtype,
                              buffer=self._shm.buf, offset=off)
            view[:] = raw
            self._descriptors.append(ColDescriptorWithOffset(
                col_name=col,
                shm_name=self._shm.name,
                dtype_str=raw.dtype.str,
                shape=raw.shape,
                offset_bytes=off,
            ))

    @property
    def descriptors(self):
        return list(self._descriptors)

    def close(self):
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def exercise1():
    print("Exercise 1 — Packed single-segment SharedDataFrame:")
    N = 100_000
    df = pd.DataFrame({
        "a": np.random.randn(N),
        "b": np.random.rand(N),
        "c": np.arange(N, dtype=np.int32).astype(np.float64),
    })

    with SharedDataFramePacked(df) as sdf:
        for desc in sdf.descriptors:
            shm, arr = desc.attach_array()
            print(f"  {desc.col_name}: offset={desc.offset_bytes}, "
                  f"mean={arr.mean():.4f}, sum_check={np.isclose(arr.mean(), df[desc.col_name].mean())}")
            shm.close()
    print("  Packed into ONE segment ✓")


# ---------------------------------------------------------------------------
# Exercise 4 — Benchmark: shared memory vs pickle DataFrame
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColDescriptor:
    col_name: str
    shm_name: str
    dtype_str: str
    shape: tuple

    def nbytes(self):
        return int(np.prod(self.shape)) * np.dtype(self.dtype_str).itemsize


def write_col_to_shm(series: pd.Series, col_name: str) -> tuple[SharedMemory, ColDescriptor]:
    raw = np.ascontiguousarray(series.to_numpy())
    shm = SharedMemory(create=True, size=raw.nbytes)
    view = np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm.buf)
    view[:] = raw
    return shm, ColDescriptor(col_name=col_name, shm_name=shm.name,
                               dtype_str=raw.dtype.str, shape=raw.shape)


# Module-level worker functions (must be picklable)
def worker_shm(desc: ColDescriptor) -> dict:
    shm = SharedMemory(name=desc.shm_name, create=False)
    arr = np.ndarray(desc.shape, dtype=np.dtype(desc.dtype_str), buffer=shm.buf)
    result = {"col": desc.col_name, "mean": float(arr.mean()), "std": float(arr.std())}
    shm.close()
    return result


def worker_pickle(df_col: pd.Series) -> dict:
    arr = df_col.to_numpy()
    return {"col": df_col.name, "mean": float(arr.mean()), "std": float(arr.std())}


def exercise4_benchmark():
    print("\nExercise 4 — Benchmark: shared memory vs pickle DataFrame:")
    N_ROWS = 500_000
    N_FEATURES = 20
    N_WORKERS = 4

    np.random.seed(42)
    df = pd.DataFrame({f"feature_{i}": np.random.randn(N_ROWS) for i in range(N_FEATURES)})
    feature_cols = list(df.columns)
    print(f"  DataFrame: {N_ROWS:,} rows × {N_FEATURES} features = "
          f"{df.memory_usage(deep=True).sum()/1e6:.1f} MB")

    # --- Approach (a): pickle full column ---
    rss_before = get_rss_mb()
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        results_pickle = list(ex.map(worker_pickle, [df[c] for c in feature_cols]))
    t_pickle = time.perf_counter() - t0
    rss_after_pickle = get_rss_mb()

    # --- Approach (b): shared memory ---
    segments = []
    descs = []
    for col in feature_cols:
        shm, desc = write_col_to_shm(df[col], col)
        segments.append(shm)
        descs.append(desc)

    rss_before_shm = get_rss_mb()
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        results_shm = list(ex.map(worker_shm, descs))
    t_shm = time.perf_counter() - t0
    rss_after_shm = get_rss_mb()

    for shm in segments:
        shm.close()
        shm.unlink()

    print(f"\n  Pickle approach:       {t_pickle:.2f}s  (RSS delta: {rss_after_pickle-rss_before:+.0f} MB)")
    print(f"  Shared memory approach: {t_shm:.2f}s   (RSS delta: {rss_after_shm-rss_before_shm:+.0f} MB)")
    print(f"  Speedup: {t_pickle/t_shm:.1f}×")

    # Verify results match
    r_p = {r["col"]: r["mean"] for r in results_pickle}
    r_s = {r["col"]: r["mean"] for r in results_shm}
    match = all(np.isclose(r_p[c], r_s[c]) for c in feature_cols)
    print(f"  Results match: {match}")


if __name__ == "__main__":
    exercise1()
    exercise4_benchmark()
