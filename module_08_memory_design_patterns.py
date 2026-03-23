"""
Module 08 — Memory-Efficient Design Patterns: ColDescriptor & SharedDataFrame
==============================================================================

Learning objectives
-------------------
1. Understand the ColDescriptor pattern for passing column metadata to workers.
2. Build a SharedDataFrame class that stores columns in shared memory.
3. Reason about peak memory usage with and without shared memory.
4. Understand the "always-needed vs per-task" column split.
5. Know when shared memory is worth the complexity.

Run this module:
    python module_08_memory_design_patterns.py
"""

import os
import sys
import time
import dataclasses
from dataclasses import dataclass
from typing import Optional
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. THE PROBLEM: REPEATED LARGE COPIES TO WORKERS
# ---------------------------------------------------------------------------
#
# Naive parallel feature computation:
#
#   def compute_feature(df, feature_col):          ← df is 4 GB
#       return df[feature_col].mean()
#
#   with ProcessPoolExecutor(4) as ex:
#       futures = [ex.submit(compute_feature, df, col) for col in feature_cols]
#
# What actually happens:
#   - df is pickled once per submit() call.
#   - With 100 feature columns and 4 workers, df is pickled 100 times.
#   - Each pickle sends 4 GB over the IPC pipe → 400 GB total serialisation!
#   - Peak RAM: parent (4 GB) + worker unpickling (4 GB) × 4 = 20 GB.
#
# The fix: put the SHARED data in shared memory ONCE.
# Workers attach by name, process their column, close the handle.
# Only the small per-column data needs to be passed via normal IPC.


# ---------------------------------------------------------------------------
# 2. COLDESCRIPTOR — THE METADATA STRUCT
# ---------------------------------------------------------------------------
#
# A ColDescriptor stores everything a worker needs to:
#   1. Attach to the shared memory segment holding this column's data.
#   2. Reconstruct the numpy array from raw bytes.
#   3. Know which column this corresponds to in the output DataFrame.
#
# All fields are plain Python types → fully picklable → can be sent to workers.

@dataclass(frozen=True)
class ColDescriptor:
    """
    Immutable metadata describing one column stored in a shared memory segment.

    Fields:
        col_name  : pandas column name (for rebuilding the output DataFrame)
        shm_name  : OS shared memory segment name (for attaching)
        dtype_str : numpy dtype string, e.g. '<f8' (for interpreting bytes)
        shape     : tuple of ints (for np.ndarray reconstruction)
    """
    col_name: str
    shm_name: str
    dtype_str: str
    shape: tuple

    def nbytes(self) -> int:
        """Total bytes this column occupies in shared memory."""
        return int(np.prod(self.shape)) * np.dtype(self.dtype_str).itemsize

    def attach_array(self) -> tuple["SharedMemory", np.ndarray]:
        """
        Attach to the shared memory segment and return (shm, array_view).
        Caller is responsible for calling shm.close() when done.
        """
        shm = SharedMemory(name=self.shm_name, create=False)
        arr = np.ndarray(self.shape, dtype=np.dtype(self.dtype_str), buffer=shm.buf)
        return shm, arr

    def read_copy(self) -> np.ndarray:
        """
        Attach, copy the array locally, and immediately close the handle.
        Safe pattern for workers that hold the data beyond the task lifetime.
        """
        shm, arr_view = self.attach_array()
        local = arr_view.copy()
        shm.close()
        return local


def write_series_to_shm(series: pd.Series, col_name: str) -> tuple[SharedMemory, ColDescriptor]:
    """
    Write a pandas Series into a new shared memory segment.

    Returns (shm, descriptor).  The caller OWNS the shm and must call
    shm.close() and shm.unlink() when done.
    """
    raw = np.ascontiguousarray(series.to_numpy())
    shm = SharedMemory(create=True, size=raw.nbytes)
    view = np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm.buf)
    view[:] = raw
    desc = ColDescriptor(
        col_name=col_name,
        shm_name=shm.name,
        dtype_str=raw.dtype.str,
        shape=raw.shape,
    )
    return shm, desc


# ---------------------------------------------------------------------------
# 3. SharedDataFrame — MANAGING MULTIPLE COLUMNS
# ---------------------------------------------------------------------------
#
# A SharedDataFrame holds N columns, each in its own shared memory segment.
# It manages:
#   - Creating segments when constructed from a DataFrame.
#   - Returning ColDescriptors for workers.
#   - Cleaning up all segments on close/exit.
#
# Design decisions:
#   - One segment per column (simplest; avoids alignment bookkeeping).
#   - Parent owns all segments; workers only read.
#   - Use as a context manager to guarantee cleanup.

class SharedDataFrame:
    """
    A subset of a DataFrame's columns held in shared memory.

    Usage:
        with SharedDataFrame(df, columns=["a", "b", "c"]) as sdf:
            descriptors = sdf.descriptors
            # Pass descriptors to workers
    """

    def __init__(self, df: pd.DataFrame, columns: Optional[list] = None):
        cols = columns if columns is not None else list(df.columns)
        self._segments: list[SharedMemory] = []
        self._descriptors: list[ColDescriptor] = []

        for col in cols:
            shm, desc = write_series_to_shm(df[col], col_name=col)
            self._segments.append(shm)
            self._descriptors.append(desc)

    @property
    def descriptors(self) -> list[ColDescriptor]:
        """Return the list of ColDescriptors (safe to pass to workers)."""
        return list(self._descriptors)

    def close(self) -> None:
        """Detach and destroy all shared memory segments."""
        for shm in self._segments:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._segments.clear()
        self._descriptors.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self):
        cols = [d.col_name for d in self._descriptors]
        total_mb = sum(d.nbytes() for d in self._descriptors) / 1e6
        return f"SharedDataFrame(columns={cols}, total={total_mb:.2f} MB)"


# ---------------------------------------------------------------------------
# 4. WORKER FUNCTIONS
# ---------------------------------------------------------------------------
#
# Workers receive DESCRIPTORS (picklable metadata), not the actual data.
# They attach to shared memory, do work, and return a small result.

def compute_column_stats(
    always_needed_descs: list[ColDescriptor],
    feature_desc: ColDescriptor,
) -> dict:
    """
    Example worker: compute stats using some always-needed columns and one feature.

    always_needed_descs: columns that are needed for every computation
        (e.g. a mask column, a weight column)
    feature_desc: the specific feature column to process
    """
    # Attach always-needed columns
    always_shms = []
    always_arrs = []
    for desc in always_needed_descs:
        shm, arr = desc.attach_array()
        always_shms.append(shm)
        always_arrs.append(arr)

    # Attach the feature column
    feat_shm, feat_arr = feature_desc.attach_array()

    # Do computation (trivial example; real code would be more complex)
    result = {
        "col": feature_desc.col_name,
        "mean": float(np.mean(feat_arr)),
        "std": float(np.std(feat_arr)),
        # Use always_needed columns:
        "weighted_sum": float(np.dot(feat_arr, always_arrs[0])) if always_arrs else None,
    }

    # Close ALL shared memory handles before returning
    feat_shm.close()
    for shm in always_shms:
        shm.close()

    return result


# ---------------------------------------------------------------------------
# 5. PUTTING IT TOGETHER: PARALLEL FEATURE PROCESSING
# ---------------------------------------------------------------------------

def demo_parallel_feature_processing():
    """
    Build a realistic pipeline:
      - Large "always_needed" columns in shared memory (shared across all tasks)
      - One "feature" column per task (also in shared memory for consistency)
      - Workers receive only descriptors (tiny picklable structs)
    """
    N_ROWS = 200_000
    N_FEATURES = 12

    print(f"  Building DataFrame: {N_ROWS:,} rows, {N_FEATURES} feature columns")
    np.random.seed(42)
    df = pd.DataFrame({
        "weight":  np.random.rand(N_ROWS).astype(np.float64),  # always needed
        **{f"feature_{i}": np.random.randn(N_ROWS).astype(np.float64)
           for i in range(N_FEATURES)},
    })
    print(f"  DataFrame size: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Separate always-needed from feature columns
    always_needed_cols = ["weight"]
    feature_cols = [c for c in df.columns if c.startswith("feature_")]

    print(f"  Putting always-needed columns ({always_needed_cols}) into shared memory...")
    with SharedDataFrame(df, columns=always_needed_cols) as always_sdf:
        always_descs = always_sdf.descriptors
        print(f"  SharedDataFrame for always-needed: {always_sdf}")

        # Put feature columns in shared memory too
        with SharedDataFrame(df, columns=feature_cols) as feature_sdf:
            feature_descs = feature_sdf.descriptors
            print(f"  SharedDataFrame for features: {feature_sdf}")

            print(f"  Launching {N_FEATURES} parallel tasks...")
            t0 = time.perf_counter()
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(compute_column_stats, always_descs, fd): fd.col_name
                    for fd in feature_descs
                }
                results = []
                for future in as_completed(futures):
                    r = future.result()
                    results.append(r)
                    print(f"    {r['col']:>12}: mean={r['mean']:+.4f}, std={r['std']:.4f}, "
                          f"weighted_sum={r['weighted_sum']:.1f}")

            elapsed = time.perf_counter() - t0
            print(f"  Total wall time: {elapsed:.2f}s")
            print(f"  (Shared memory segments stay alive until the 'with' block exits)")

    print("  All shared memory cleaned up ✓")


# ---------------------------------------------------------------------------
# 6. PEAK MEMORY ANALYSIS
# ---------------------------------------------------------------------------
#
# WHY DOES THIS DESIGN SAVE MEMORY?
#
# Naive approach (pickle DataFrame to each worker):
#   Parent process:    DataFrame (D GB)
#   Worker 0 unpack:   DataFrame copy (D GB)   ← per worker
#   Worker 1 unpack:   DataFrame copy (D GB)
#   ...
#   Peak = D * (1 + N_workers) GB
#   For D=4 GB, 4 workers: peak = 20 GB
#
# Shared memory approach:
#   Parent process:    DataFrame (D GB)  [can be freed after writing to shm]
#   Shared memory:     always_needed columns (S GB)
#                      one feature column per worker (F GB × N_workers)
#   Worker memory:     local result dict (tiny)
#   Peak ≈ D + S + F * N_workers GB
#   For D=4, S=0.5, F=0.05, 4 workers: peak ≈ 4.7 GB
#
# Even better: free the original DataFrame after writing to shared memory.
#   Peak = S + F * N_workers
#         = 0.5 + 0.2 = 0.7 GB  (if the original 4 GB is freed)

def demo_memory_math():
    """Print the memory maths for different approaches."""
    print("  Peak memory comparison:")
    print()

    D = 4.0   # DataFrame size in GB
    N = 4     # number of workers
    S = 0.5   # always-needed columns in GB
    F = 0.05  # one feature column in GB

    naive_peak = D * (1 + N)
    shm_with_df = D + S + F * N
    shm_freed = S + F * N

    print(f"    DataFrame size:             {D:.1f} GB")
    print(f"    Number of workers:          {N}")
    print(f"    Always-needed columns:      {S:.1f} GB")
    print(f"    One feature column:         {F:.2f} GB")
    print()
    print(f"    Naive (pickle df per task): {naive_peak:.1f} GB peak")
    print(f"    Shared memory (df kept):    {shm_with_df:.1f} GB peak")
    print(f"    Shared memory (df freed):   {shm_freed:.2f} GB peak")
    print(f"    Reduction vs naive:         {naive_peak / shm_freed:.0f}× less memory")


# ---------------------------------------------------------------------------
# 7. WHEN IS SHARED MEMORY WORTH THE COMPLEXITY?
# ---------------------------------------------------------------------------
#
# Shared memory adds real complexity:
#   - Explicit lifetime management (close/unlink)
#   - Platform differences (Windows vs Linux)
#   - No automatic garbage collection
#   - Debugging is harder (can't just print(obj))
#
# It's worth it when:
#   ✓ The data shared between parent and workers is LARGE (>100 MB)
#   ✓ You have many tasks that all need the same data
#   ✓ The data is read-only in workers (no synchronisation needed)
#   ✓ You're on spawn-based systems where COW doesn't help
#
# It's NOT worth it when:
#   ✗ Data is small (just pickle it)
#   ✗ Workers need to write back to the shared data (use queues/locks instead)
#   ✗ You're on Linux with fork and the data is read-only (COW is free)

def when_to_use_shared_memory():
    """Print a decision guide."""
    print("  When to use shared memory:")
    print()
    print("  USE shared memory if:")
    print("    ✓ Large read-only data (> ~100 MB)")
    print("    ✓ Many tasks (> ~10) all need the same data")
    print("    ✓ Using spawn start method (macOS, Windows, or explicit)")
    print("    ✓ Peak memory is a constraint")
    print()
    print("  SKIP shared memory if:")
    print("    ✗ Data fits comfortably in pickle (< ~50 MB)")
    print("    ✗ Linux fork + read-only access (COW is nearly free)")
    print("    ✗ Workers need to mutate shared state (use Queue/Lock instead)")
    print("    ✗ One-off parallelism where simplicity matters more")


# ---------------------------------------------------------------------------
# DEMO RUNNER
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODULE 08 — Memory-Efficient Design Patterns")
    print("=" * 60)

    print("\n[1] ColDescriptor: metadata for shared columns")
    # Show that ColDescriptor is fully picklable
    import pickle
    desc = ColDescriptor(col_name="x", shm_name="psm_test", dtype_str="<f8", shape=(1000,))
    roundtrip = pickle.loads(pickle.dumps(desc))
    print(f"  ColDescriptor pickling: {roundtrip == desc}  (roundtrip identical)")
    print(f"  {desc}")
    print(f"  nbytes={desc.nbytes()}")

    print("\n[2] SharedDataFrame context manager")
    N = 100_000
    df_small = pd.DataFrame({
        "weight": np.random.rand(N),
        "feat_a": np.random.randn(N),
        "feat_b": np.random.randn(N),
    })
    with SharedDataFrame(df_small) as sdf:
        print(f"  {sdf}")
        for d in sdf.descriptors:
            shm, arr = d.attach_array()
            print(f"    {d.col_name}: mean={arr.mean():.4f}")
            shm.close()
    print("  Cleaned up ✓")

    print("\n[3] Parallel feature processing pipeline")
    demo_parallel_feature_processing()

    print("\n[4] Peak memory analysis")
    demo_memory_math()

    print("\n[5] When to use shared memory")
    when_to_use_shared_memory()

    print("\n[6] Key takeaways")
    print("""
    KEY PATTERNS:
      1. ColDescriptor stores (shm_name, dtype_str, shape, col_name).
         All fields are plain Python types → fully picklable → tiny IPC cost.

      2. SharedDataFrame manages N segments as a context manager.
         Parent creates + owns; workers attach + read + close.

      3. Workers close shm handles immediately after reading.
         Never unlink in workers. Unlink only in the owner (parent).

      4. Peak memory: shared memory approach can be 10–30× lower than naive
         pickle-DataFrame-to-every-worker approach.

    THE CORE INSIGHT:
      Pass the NAME of the data (cheap string), not the DATA (expensive copy).
      Workers find the data by name in the OS-managed shared memory.
    """)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
#
# Exercise 1 — Extend ColDescriptor
# -----------------------------------
# Add a field `offset_bytes: int = 0` to ColDescriptor, which allows multiple
# columns to share a single SharedMemory segment.
# Modify write_series_to_shm to accept an existing SharedMemory and an offset.
# Write a SharedDataFrame2 that packs all columns into ONE segment.
# Verify correctness. Compare memory overhead (one segment vs N segments).
#
# Exercise 2 — Read-only enforcement
# ------------------------------------
# Modify the worker function so that the numpy view from attach_array() is
# marked read-only (arr.flags['WRITEABLE'] = False) before any computation.
# This prevents accidental mutations in workers.
# Verify: attempting to write to arr in the worker raises ValueError.
#
# Exercise 3 — Graceful cleanup on worker crash
# -----------------------------------------------
# Simulate a worker that crashes (raises an exception) after attaching to
# shared memory but before calling shm.close().
# Use ProcessPoolExecutor and observe: do the shm handles leak?
# (On Linux, the kernel closes file descriptors when a process exits, so the
#  handle IS released. But is the segment name unlinked? Who calls unlink?)
# Write a context manager that guarantees cleanup even if workers crash.
#
# Exercise 4 — Benchmark: shared memory vs pickle DataFrame
# ----------------------------------------------------------
# Build the same feature computation pipeline in two ways:
#   a) Pass the full DataFrame to each worker via pickle
#   b) Use SharedDataFrame + ColDescriptor
# Measure wall time and peak RSS (using psutil.Process().memory_info().rss)
# for N_ROWS=500_000, N_FEATURES=20, 4 workers.
# At what DataFrame size does (b) start winning on TIME (not just memory)?
