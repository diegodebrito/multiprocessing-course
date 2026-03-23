"""
Module 11 — Capstone: Full Parallel Feature-Engineering Pipeline
================================================================

This capstone brings together every concept from the course into one realistic,
production-flavoured pipeline:

  1. PyArrow Dataset → read raw data (module 09)
  2. SharedDataFrame / ColDescriptor → always-needed columns in shared memory (modules 05, 06, 08)
  3. ProcessPoolExecutor → parallel feature computation (module 03)
  4. Pickling discipline → workers receive metadata, not data (module 04)
  5. Start method awareness → fork on Linux (module 02)
  6. Context managers → resource safety (module 10)
  7. Pandas ↔ NumPy memory → careful extraction (module 07)

PIPELINE OVERVIEW:
------------------
  Given a dataset with:
    - "always needed" columns: row_id, weight  (needed in every computation)
    - "feature" columns: feature_00 … feature_N (one independent computation each)

  For each feature column, compute:
    - mean, std
    - weighted mean (using the weight column from shared memory)
    - n_above_threshold (count of values > threshold, using weight > 0.5 as mask)

  Output: a DataFrame with one row per feature, with the computed stats.

Run this module:
    python module_11_capstone.py

Expected output: a summary DataFrame and timing / memory report.
"""

import os
import sys
import gc
import time
import shutil
import tempfile
import contextlib
import dataclasses
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd

try:
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ===========================================================================
# SECTION 1: COLUMN DESCRIPTOR
# (Revisited from module 08 — fully self-contained here)
# ===========================================================================

@dataclass(frozen=True)
class ColDescriptor:
    """
    Immutable metadata that workers use to attach to a shared memory segment.
    All fields are plain Python types → fully picklable.
    """
    col_name: str
    shm_name: str
    dtype_str: str
    shape: tuple

    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * np.dtype(self.dtype_str).itemsize


# ===========================================================================
# SECTION 2: SHARED DATA FRAME
# (Context-managed owner of multiple shared memory segments)
# ===========================================================================

class SharedDataFrame:
    """
    Holds a subset of DataFrame columns in shared memory.
    Parent process creates; workers attach by name.
    Context manager guarantees cleanup on exit or exception.
    """

    def __init__(self, df: pd.DataFrame, columns: Optional[list] = None):
        cols = columns if columns is not None else list(df.columns)
        self._segments: list[SharedMemory] = []
        self._descriptors: list[ColDescriptor] = []
        for col in cols:
            raw = np.ascontiguousarray(df[col].to_numpy())
            shm = SharedMemory(create=True, size=raw.nbytes)
            view = np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm.buf)
            view[:] = raw
            self._segments.append(shm)
            self._descriptors.append(ColDescriptor(
                col_name=col,
                shm_name=shm.name,
                dtype_str=raw.dtype.str,
                shape=raw.shape,
            ))

    @property
    def descriptors(self) -> list[ColDescriptor]:
        return list(self._descriptors)

    def close(self) -> None:
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
        total_mb = sum(d.nbytes() for d in self._descriptors) / 1e6
        return f"SharedDataFrame(cols={len(self._descriptors)}, {total_mb:.2f} MB)"


# ===========================================================================
# SECTION 3: DATA GENERATION
# ===========================================================================

def generate_dataset(
    tmp_dir: str,
    n_rows: int,
    n_features: int,
    n_files: int = 4,
) -> str:
    """Write a partitioned Parquet dataset to disk."""
    np.random.seed(0)
    rows_per_file = n_rows // n_files

    for i in range(n_files):
        df = pd.DataFrame({
            "row_id": np.arange(i * rows_per_file, (i + 1) * rows_per_file, dtype=np.int64),
            "weight": np.random.rand(rows_per_file).astype(np.float64),
            **{
                f"feature_{j:02d}": np.random.randn(rows_per_file).astype(np.float64)
                for j in range(n_features)
            },
        })
        path = os.path.join(tmp_dir, f"part_{i:03d}.parquet")
        df.to_parquet(path, index=False)

    return tmp_dir


# ===========================================================================
# SECTION 4: WORKER FUNCTION
# ===========================================================================
#
# This function must be defined at MODULE TOP LEVEL (not inside main or any
# other function) because it needs to be picklable for ProcessPoolExecutor.
#
# What this worker receives (all picklable):
#   - always_descs: list of ColDescriptor (tiny structs with shm names + metadata)
#   - feature_col:  the column name to read from disk (plain string)
#   - dataset_path: path to the Parquet dataset directory (plain string)
#   - threshold:    a float
#
# What the worker does:
#   1. Attach to always-needed shared memory segments.
#   2. Read the feature column from disk via PyArrow (or fallback to pandas).
#   3. Compute stats using both sources.
#   4. Close all shared memory handles.
#   5. Return a small result dict.

def compute_feature_stats(
    always_descs: list[ColDescriptor],
    feature_col: str,
    dataset_path: str,
    threshold: float = 0.0,
) -> dict:
    """
    Worker function: computes stats for one feature column.

    Uses:
      - always_needed columns from shared memory (weight, row_id)
      - feature column read from disk (one column only — column projection)
    """
    # --- Attach always-needed columns from shared memory ---
    always_shms: list[SharedMemory] = []
    always_arrays: dict[str, np.ndarray] = {}
    for desc in always_descs:
        shm = SharedMemory(name=desc.shm_name, create=False)
        arr = np.ndarray(desc.shape, dtype=np.dtype(desc.dtype_str), buffer=shm.buf)
        arr.flags["WRITEABLE"] = False  # enforce read-only in workers
        always_arrays[desc.col_name] = arr
        always_shms.append(shm)

    # --- Read feature column from disk ---
    if _HAS_PYARROW:
        worker_ds = ds.dataset(dataset_path, format="parquet")
        table = worker_ds.to_table(columns=[feature_col])
        feature_series = table.to_pandas()[feature_col]
    else:
        # Fallback: read all parquet files and concat
        import glob
        dfs = [pd.read_parquet(p, columns=[feature_col])
               for p in sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))]
        feature_series = pd.concat(dfs, ignore_index=True)[feature_col]

    feature_arr = np.ascontiguousarray(feature_series.to_numpy())

    # --- Compute stats ---
    weight = always_arrays.get("weight")
    result = {
        "feature":          feature_col,
        "mean":             float(np.mean(feature_arr)),
        "std":              float(np.std(feature_arr)),
        "min":              float(np.min(feature_arr)),
        "max":              float(np.max(feature_arr)),
        "n_above_threshold": int(np.sum(feature_arr > threshold)),
        "weighted_mean":    float(np.average(feature_arr, weights=weight))
                            if weight is not None else None,
    }

    # --- Close ALL shared memory handles ---
    # Critical: close before returning; never unlink (parent owns lifetime)
    for shm in always_shms:
        shm.close()

    return result


# ===========================================================================
# SECTION 5: MEMORY MONITORING
# ===========================================================================

def get_rss_mb() -> float:
    """Return current process RSS in MB, or -1 if psutil not available."""
    if not _HAS_PSUTIL:
        return -1.0
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1e6


# ===========================================================================
# SECTION 6: MAIN PIPELINE
# ===========================================================================

def run_pipeline(
    n_rows: int = 200_000,
    n_features: int = 16,
    n_workers: int = 4,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Full parallel feature-engineering pipeline.

    Returns a DataFrame with one row per feature column and computed stats.
    """
    print("=" * 65)
    print("CAPSTONE PIPELINE")
    print("=" * 65)
    print(f"  Rows: {n_rows:,}  |  Features: {n_features}  |  Workers: {n_workers}")
    print()

    rss_start = get_rss_mb()

    # ------------------------------------------------------------------
    # STEP 1: Generate / locate the dataset
    # ------------------------------------------------------------------
    print("[Step 1] Generating Parquet dataset...")
    tmp_dir = tempfile.mkdtemp(prefix="capstone_")
    try:
        t0 = time.perf_counter()
        generate_dataset(tmp_dir, n_rows=n_rows, n_features=n_features)
        t_gen = time.perf_counter() - t0
        print(f"         {n_files := 4} files in {tmp_dir} ({t_gen:.2f}s)")

        # ------------------------------------------------------------------
        # STEP 2: Load always-needed columns into memory, then into shared memory
        # ------------------------------------------------------------------
        print("\n[Step 2] Loading always-needed columns → shared memory...")
        t0 = time.perf_counter()

        if _HAS_PYARROW:
            dataset = ds.dataset(tmp_dir, format="parquet")
            always_table = dataset.to_table(columns=["row_id", "weight"])
            always_df = always_table.to_pandas()
        else:
            import glob
            dfs = [pd.read_parquet(p, columns=["row_id", "weight"])
                   for p in sorted(glob.glob(os.path.join(tmp_dir, "*.parquet")))]
            always_df = pd.concat(dfs, ignore_index=True)

        rss_after_load = get_rss_mb()
        print(f"         Loaded {len(always_df):,} rows, "
              f"{always_df.memory_usage(deep=True).sum()/1e6:.1f} MB "
              f"(RSS: {rss_after_load:.0f} MB)")

        # ------------------------------------------------------------------
        # STEP 3: Parallel computation
        # ------------------------------------------------------------------
        print("\n[Step 3] Parallel feature computation...")
        feature_cols = [f"feature_{j:02d}" for j in range(n_features)]

        results = []
        with SharedDataFrame(always_df, columns=["row_id", "weight"]) as sdf:
            always_descs = sdf.descriptors
            rss_after_shm = get_rss_mb()
            print(f"         SharedDataFrame: {sdf}  (RSS: {rss_after_shm:.0f} MB)")

            # Free the in-memory DataFrame — workers will read features from disk
            # and always-needed from shared memory.
            del always_df
            gc.collect()
            rss_after_del = get_rss_mb()
            print(f"         Freed in-memory DataFrame (RSS: {rss_after_del:.0f} MB)")

            t0 = time.perf_counter()
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        compute_feature_stats,
                        always_descs,
                        col,
                        tmp_dir,
                        threshold,
                    ): col
                    for col in feature_cols
                }
                for future in as_completed(futures):
                    col = futures[future]
                    if future.exception():
                        print(f"         ERROR in {col}: {future.exception()}")
                    else:
                        r = future.result()
                        results.append(r)
                        print(f"         {r['feature']:>12}: "
                              f"mean={r['mean']:+.4f}, "
                              f"std={r['std']:.4f}, "
                              f"weighted_mean={r['weighted_mean']:+.4f}")

            t_compute = time.perf_counter() - t0
            rss_peak = get_rss_mb()

        # sdf.__exit__ called here: all shared memory cleaned up
        print(f"\n         Parallel compute: {t_compute:.2f}s")

        # ------------------------------------------------------------------
        # STEP 4: Assemble results
        # ------------------------------------------------------------------
        print("\n[Step 4] Assembling result DataFrame...")
        result_df = pd.DataFrame(results).set_index("feature").sort_index()

        # ------------------------------------------------------------------
        # STEP 5: Report
        # ------------------------------------------------------------------
        rss_end = get_rss_mb()
        print("\n" + "=" * 65)
        print("RESULT SUMMARY")
        print("=" * 65)
        print(result_df.to_string())
        print()
        print("PERFORMANCE REPORT")
        print(f"  Features computed:  {len(results)}")
        print(f"  Workers:            {n_workers}")
        print(f"  Parallel wall time: {t_compute:.2f}s")
        print(f"  Avg per feature:    {t_compute/n_features*1000:.0f}ms")
        if _HAS_PSUTIL:
            print(f"  RSS start:          {rss_start:.0f} MB")
            print(f"  RSS after SHM:      {rss_after_shm:.0f} MB")
            print(f"  RSS after del df:   {rss_after_del:.0f} MB")
            print(f"  RSS end:            {rss_end:.0f} MB")

        return result_df

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\n  Dataset cleaned up: {tmp_dir}")


# ===========================================================================
# SECTION 7: WHAT TO OBSERVE WHEN RUNNING
# ===========================================================================

OBSERVATION_GUIDE = """
WHAT TO OBSERVE WHEN RUNNING THIS CAPSTONE
===========================================

1. PICKLE SIZE vs DATA SIZE
   - Each task submission pickles: always_descs (tiny), feature_col (string), dataset_path (string)
   - Total pickle per task ≈ a few KB, NOT the full dataset
   - Without shared memory: would pickle N_ROWS * 8 bytes ≈ 1.6 MB per always-needed column per task

2. SHARED MEMORY LIFETIME
   - SharedDataFrame is created before the pool, outlives all worker processes
   - Workers attach, read, close — they never unlink
   - After the `with SharedDataFrame` block exits, segments are cleaned up
   - Verify: ls /dev/shm shows no leftover segments

3. MEMORY RSS
   - RSS after SharedDataFrame creation ≈ RSS after loading (shared memory is in kernel, not RSS)
   - RSS after deleting the DataFrame drops (memory returned to OS)
   - Worker processes add ~overhead for Python interpreter + one feature column each
   - Peak RSS is much lower than: parent_df + N_workers * parent_df

4. PYARROW COLUMN PROJECTION
   - Each worker reads ONLY its feature column from disk
   - The heavy_col_* equivalent columns are never loaded into worker memory
   - This is the to_table(columns=[col]) call inside the worker

5. START METHOD
   - Linux default is fork → workers start fast (< 1ms)
   - Workers inherit the parent's Python state but NOT the SharedMemory segments
     (which they must re-attach by name)
   - The SharedDataFrame segments are accessible by name in workers because
     the OS kernel holds them — not because they're inherited via fork

6. EXCEPTION SAFETY
   - If a worker raises, its shared memory handles are closed when the process exits
   - The parent's SharedDataFrame context manager still calls close+unlink on exit
   - Result: no leaked segments even when workers crash
"""


# ===========================================================================
# SECTION 8: EXERCISES
# ===========================================================================

EXERCISES = """
CAPSTONE EXERCISES
==================

Exercise 1 — Add a new statistic
----------------------------------
Add `median` to the computed stats.
Note: np.median() is O(n log n) and can't be computed in a single pass.
How does this affect per-task time? Measure it.

Exercise 2 — Chunked processing
---------------------------------
Modify the pipeline to handle datasets that don't fit in RAM.
Instead of loading all always-needed columns at once:
  - Read always-needed columns in chunks of 50k rows.
  - Process only the features corresponding to each chunk.
  - Aggregate chunk-level stats into global stats.
What changes in the ColDescriptor / SharedDataFrame design?

Exercise 3 — Benchmark naive vs shared memory
----------------------------------------------
Implement the naive version:
  - Load the FULL DataFrame into RAM.
  - Pass df[[always_cols + [feature_col]]].copy() to each worker via pickle.
Compare wall time and peak RSS for:
  n_rows=500_000, n_features=20, n_workers=4
Measure the overhead of always-needed column pickling per task.

Exercise 4 — Add write-back results to shared memory
-----------------------------------------------------
Modify workers to write their result (mean, std, etc.) into a pre-allocated
results array in shared memory (one row per feature column).
Collect the results array in the parent WITHOUT any IPC return values.
This eliminates the result-pickling step entirely.
Hint: you need a pre-allocated (n_features, n_stats) float64 array in shm.

Exercise 5 — Handle nullable columns
---------------------------------------
Add a column with nullable integers (pd.array([...], dtype="Int64")) to the dataset.
Trace what happens when a worker calls to_numpy() on it.
Fix the worker so it handles nullable columns correctly (convert to float64 with na_value=0).
When does this copy happen, and is it acceptable?

Exercise 6 — Cross-platform compatibility
------------------------------------------
The current code uses Linux fork semantics.
Rewrite the pipeline to work correctly with spawn (macOS/Windows):
  - Workers can no longer rely on fork'd state.
  - Ensure all data passed to workers is explicitly picklable.
  - Ensure shared memory handles are closed by workers BEFORE the parent
    calls unlink() (Windows requires this).
Test: set multiprocessing.set_start_method('spawn') at the top and verify.
"""


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    print(OBSERVATION_GUIDE)

    # Run the pipeline
    result_df = run_pipeline(
        n_rows=200_000,
        n_features=12,
        n_workers=4,
        threshold=0.0,
    )

    print("\n" + EXERCISES)
