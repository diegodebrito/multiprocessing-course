"""
Solutions — Module 11: Capstone Exercises
"""
import os
import gc
import sys
import shutil
import tempfile
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


# ---------------------------------------------------------------------------
# Shared infrastructure (mirrors module 11)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColDescriptor:
    col_name: str
    shm_name: str
    dtype_str: str
    shape: tuple

    def nbytes(self): return int(np.prod(self.shape)) * np.dtype(self.dtype_str).itemsize


class SharedDataFrame:
    def __init__(self, df: pd.DataFrame, columns=None):
        cols = columns or list(df.columns)
        self._segs, self._descs = [], []
        for col in cols:
            raw = np.ascontiguousarray(df[col].to_numpy())
            shm = SharedMemory(create=True, size=raw.nbytes)
            np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm.buf)[:] = raw
            self._segs.append(shm)
            self._descs.append(ColDescriptor(col, shm.name, raw.dtype.str, raw.shape))

    @property
    def descriptors(self): return list(self._descs)

    def close(self):
        for shm in self._segs:
            try: shm.close(); shm.unlink()
            except Exception: pass
        self._segs.clear(); self._descs.clear()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def make_dataset(tmp_dir: str, n_rows: int, n_features: int) -> None:
    np.random.seed(0)
    df = pd.DataFrame({
        "row_id": np.arange(n_rows, dtype=np.int64),
        "weight": np.random.rand(n_rows).astype(np.float64),
        **{f"feature_{j:02d}": np.random.randn(n_rows).astype(np.float64)
           for j in range(n_features)},
    })
    if _HAS_PYARROW:
        import pyarrow as pa
        pq.write_table(pa.Table.from_pandas(df), os.path.join(tmp_dir, "data.parquet"))
    else:
        df.to_parquet(os.path.join(tmp_dir, "data.parquet"))


# ---------------------------------------------------------------------------
# Exercise 1 — Add median to computed stats
# ---------------------------------------------------------------------------

def compute_with_median(always_descs, feature_col, dataset_path, threshold=0.0):
    always_shms, always_arrays = [], {}
    for desc in always_descs:
        shm = SharedMemory(name=desc.shm_name, create=False)
        arr = np.ndarray(desc.shape, dtype=np.dtype(desc.dtype_str), buffer=shm.buf)
        always_arrays[desc.col_name] = arr
        always_shms.append(shm)

    if _HAS_PYARROW:
        worker_ds = ds.dataset(dataset_path, format="parquet")
        feature_arr = np.ascontiguousarray(
            worker_ds.to_table(columns=[feature_col]).to_pandas()[feature_col].to_numpy()
        )
    else:
        feature_arr = np.ascontiguousarray(
            pd.read_parquet(os.path.join(dataset_path, "data.parquet"),
                            columns=[feature_col])[feature_col].to_numpy()
        )

    result = {
        "feature": feature_col,
        "mean":   float(np.mean(feature_arr)),
        "std":    float(np.std(feature_arr)),
        "median": float(np.median(feature_arr)),   # ← O(n log n), partitions data
        "weighted_mean": float(np.average(
            feature_arr, weights=always_arrays.get("weight", np.ones(len(feature_arr)))
        )),
    }
    for shm in always_shms:
        shm.close()
    return result


def exercise1():
    print("Exercise 1 — Adding median:")
    tmp = tempfile.mkdtemp()
    N, F = 100_000, 6
    try:
        make_dataset(tmp, N, F)
        df_always = pd.read_parquet(
            os.path.join(tmp, "data.parquet"), columns=["row_id", "weight"]
        )
        feature_cols = [f"feature_{j:02d}" for j in range(F)]

        t0 = time.perf_counter()
        with SharedDataFrame(df_always, ["row_id", "weight"]) as sdf:
            with ProcessPoolExecutor(max_workers=4) as ex:
                futs = {ex.submit(compute_with_median, sdf.descriptors, c, tmp): c
                        for c in feature_cols}
                for fut in as_completed(futs):
                    r = fut.result()
                    print(f"  {r['feature']}: mean={r['mean']:+.4f}, "
                          f"median={r['median']:+.4f}, std={r['std']:.4f}")
        print(f"  Wall time with median: {time.perf_counter()-t0:.2f}s")
        print("  median is O(n log n) per task — noticeable for large N.")
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# Exercise 3 — Benchmark naive vs shared memory
# ---------------------------------------------------------------------------

def naive_worker(df_slice: pd.DataFrame) -> dict:
    """Receives a copy of the relevant columns — pickle cost per task."""
    arr = df_slice["feature"].to_numpy()
    weight = df_slice["weight"].to_numpy()
    return {
        "feature": df_slice["feature"].name if hasattr(df_slice["feature"], "name") else "?",
        "mean": float(arr.mean()),
        "weighted_mean": float(np.average(arr, weights=weight)),
    }


def naive_worker_col(col_data: np.ndarray, weight_data: np.ndarray, col_name: str) -> dict:
    return {
        "feature": col_name,
        "mean": float(col_data.mean()),
        "weighted_mean": float(np.average(col_data, weights=weight_data)),
    }


def shm_worker(always_descs, feature_desc) -> dict:
    always_shms, always_arrays = [], {}
    for desc in always_descs:
        shm = SharedMemory(name=desc.shm_name, create=False)
        always_arrays[desc.col_name] = np.ndarray(
            desc.shape, dtype=np.dtype(desc.dtype_str), buffer=shm.buf
        )
        always_shms.append(shm)
    feat_shm = SharedMemory(name=feature_desc.shm_name, create=False)
    feat_arr = np.ndarray(feature_desc.shape, dtype=np.dtype(feature_desc.dtype_str),
                          buffer=feat_shm.buf)
    result = {
        "feature": feature_desc.col_name,
        "mean": float(feat_arr.mean()),
        "weighted_mean": float(np.average(feat_arr, weights=always_arrays["weight"])),
    }
    feat_shm.close()
    for shm in always_shms:
        shm.close()
    return result


def exercise3():
    print("\nExercise 3 — Naive vs shared memory benchmark:")
    import pickle
    N_ROWS, N_FEATURES, N_WORKERS = 500_000, 20, 4
    np.random.seed(42)
    df = pd.DataFrame({
        "weight": np.random.rand(N_ROWS),
        **{f"feature_{i:02d}": np.random.randn(N_ROWS) for i in range(N_FEATURES)},
    })
    feature_cols = [c for c in df.columns if c.startswith("feature")]

    # Measure naive pickle cost per task
    sample_payload = df[["weight", feature_cols[0]]].copy()
    pickle_mb = len(pickle.dumps(sample_payload)) / 1e6
    print(f"  Pickle per task (weight + one feature col): {pickle_mb:.1f} MB")
    print(f"  Total pickle for {N_FEATURES} tasks: {pickle_mb * N_FEATURES:.0f} MB")

    # Naive: pass relevant columns via pickle
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(naive_worker_col,
                          df[c].to_numpy().copy(),
                          df["weight"].to_numpy().copy(), c)
                for c in feature_cols]
        r_naive = [f.result() for f in futs]
    t_naive = time.perf_counter() - t0

    # Shared memory
    always_segs, feature_segs = [], []
    always_descs, feature_descs = [], []
    for col in ["weight"]:
        raw = np.ascontiguousarray(df[col].to_numpy())
        shm = SharedMemory(create=True, size=raw.nbytes)
        np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm.buf)[:] = raw
        always_segs.append(shm)
        always_descs.append(ColDescriptor(col, shm.name, raw.dtype.str, raw.shape))
    for col in feature_cols:
        raw = np.ascontiguousarray(df[col].to_numpy())
        shm = SharedMemory(create=True, size=raw.nbytes)
        np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm.buf)[:] = raw
        feature_segs.append(shm)
        feature_descs.append(ColDescriptor(col, shm.name, raw.dtype.str, raw.shape))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(shm_worker, always_descs, fd) for fd in feature_descs]
        r_shm = [f.result() for f in futs]
    t_shm = time.perf_counter() - t0

    for shm in always_segs + feature_segs:
        shm.close(); shm.unlink()

    print(f"  Naive (numpy copy per task): {t_naive:.2f}s")
    print(f"  Shared memory:               {t_shm:.2f}s")
    print(f"  Speedup: {t_naive/t_shm:.1f}×")

    # Verify results match
    naive_means = {r["feature"]: r["mean"] for r in r_naive}
    shm_means   = {r["feature"]: r["mean"] for r in r_shm}
    match = all(np.isclose(naive_means[c], shm_means[c]) for c in feature_cols)
    print(f"  Results match: {match}")


# ---------------------------------------------------------------------------
# Exercise 6 — Cross-platform compatibility (spawn mode)
# ---------------------------------------------------------------------------
#
# Key changes for spawn compatibility:
#   1. All worker functions at module level (already true).
#   2. Workers close shm handles before returning (already true).
#   3. The parent should NOT call unlink() while ANY worker handle is still open.
#      → Ensure all workers have called close() before parent calls unlink().
#      → ProcessPoolExecutor shutdown(wait=True) guarantees this.
#
# The SharedDataFrame context manager already calls close+unlink AFTER the
# executor shuts down (because the executor is nested inside the SharedDataFrame
# context). So the pattern is already correct for spawn.
#
# Additional: on Windows, SharedMemory.unlink() may raise if any handle is open.
# Solution: workers should copy-then-close immediately (see module 05 pattern).

def spawn_safe_worker(desc: ColDescriptor) -> float:
    """
    Spawn-safe worker: attach, copy immediately, close handle, then compute.
    This works on Windows because we release the handle before the parent unlinks.
    """
    shm = SharedMemory(name=desc.shm_name, create=False)
    arr_view = np.ndarray(desc.shape, dtype=np.dtype(desc.dtype_str), buffer=shm.buf)
    local = arr_view.copy()   # copy to local memory
    shm.close()               # release handle IMMEDIATELY
    return float(local.mean())


def exercise6():
    print("\nExercise 6 — Cross-platform (spawn) compatibility:")
    N = 50_000
    df = pd.DataFrame({"x": np.random.randn(N)})

    # Test with spawn context if available
    available = multiprocessing.get_all_start_methods()
    method = "spawn" if "spawn" in available else "fork"
    print(f"  Using start method: {method!r}")

    ctx = multiprocessing.get_context(method)

    with SharedDataFrame(df, ["x"]) as sdf:
        desc = sdf.descriptors[0]
        # Use a spawn-compatible pool
        with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
            results = [ex.submit(spawn_safe_worker, desc).result() for _ in range(4)]
        print(f"  Results: {results}")
        print(f"  Expected: ~{df['x'].mean():.6f}")
    print("  Completed with spawn-safe pattern ✓")
    print("""
  Spawn-safety checklist:
    ✓ Worker functions at module level (not lambdas/closures)
    ✓ All worker arguments are picklable (ColDescriptor = dataclass of primitives)
    ✓ Worker copies shm data and closes handle immediately (before return)
    ✓ Parent's executor.shutdown(wait=True) runs before SharedDataFrame.close()
    ✓ SharedDataFrame.close() calls unlink() only after all workers have exited
  """)


if __name__ == "__main__":
    exercise1()
    exercise3()
    exercise6()
    # Exercises 2, 4, 5 require more scaffolding — see comments in module_11_capstone.py
    print("\nNote: Exercises 2 (chunked), 4 (write-back results), and 5 (nullable)")
    print("require more substantial refactoring. The patterns are covered in")
    print("modules 05, 06, and 07 respectively.")
