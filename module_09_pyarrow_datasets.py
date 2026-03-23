"""
Module 09 — PyArrow Datasets: Reading, Column Projection, and Conversion Costs
===============================================================================

Learning objectives
-------------------
1. Understand what a PyArrow Dataset is and how it differs from reading a single file.
2. Use column projection to avoid loading unnecessary data from disk.
3. Understand the cost of converting Arrow tables to pandas DataFrames.
4. Know which conversions are zero-copy and which are not.
5. Understand how PyArrow fits into a multiprocessing pipeline.

Run this module:
    python module_09_pyarrow_datasets.py
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False
    print("WARNING: pyarrow not installed. Install with: pip install pyarrow")
    print("         Some demos will be skipped.")


# ---------------------------------------------------------------------------
# 1. WHAT IS A PYARROW DATASET?
# ---------------------------------------------------------------------------
#
# pyarrow.dataset.Dataset represents a VIRTUAL VIEW over one or more data files
# on disk (Parquet, CSV, Arrow IPC, ORC, etc.).
#
# Key insight: a Dataset is a LAZY reference to data — nothing is read into
# memory until you actually scan it.
#
# Operations on a Dataset:
#   ds.to_table()                       → read ALL data into an Arrow Table
#   ds.to_table(columns=["a", "b"])     → read only columns a and b
#   ds.to_table(filter=expr)            → predicate pushdown
#   ds.to_batches()                     → streaming read as record batches
#
# Why use Dataset instead of reading files directly?
#   - Works over directories of partitioned files (e.g., Hive-style partitions)
#   - Consistent API regardless of format (Parquet, CSV, etc.)
#   - Supports predicate pushdown: filter BEFORE loading into RAM
#   - Column projection: load only the columns you need
#
# In multiprocessing pipelines, a Dataset is typically:
#   - Created in the parent process (fast, no I/O yet)
#   - The SCHEMA is read (column names, types)
#   - Workers scan specific columns as needed

# ---------------------------------------------------------------------------
# 2. CREATING SAMPLE PARQUET FILES FOR DEMOS
# ---------------------------------------------------------------------------

def create_sample_dataset(tmp_dir: str, n_rows: int = 100_000, n_files: int = 3) -> str:
    """Write sample Parquet files to tmp_dir and return the directory path."""
    np.random.seed(42)
    rows_per_file = n_rows // n_files

    for i in range(n_files):
        df = pd.DataFrame({
            "id":          np.arange(i * rows_per_file, (i + 1) * rows_per_file, dtype=np.int64),
            "feature_a":   np.random.randn(rows_per_file).astype(np.float64),
            "feature_b":   np.random.randn(rows_per_file).astype(np.float64),
            "feature_c":   np.random.randn(rows_per_file).astype(np.float64),
            "category":    np.random.choice(["X", "Y", "Z"], rows_per_file),
            "heavy_col_1": np.random.randn(rows_per_file).astype(np.float64),
            "heavy_col_2": np.random.randn(rows_per_file).astype(np.float64),
            "heavy_col_3": np.random.randn(rows_per_file).astype(np.float64),
        })
        path = os.path.join(tmp_dir, f"part_{i:03d}.parquet")
        df.to_parquet(path, index=False)

    return tmp_dir


# ---------------------------------------------------------------------------
# 3. COLUMN PROJECTION
# ---------------------------------------------------------------------------
#
# Column projection = reading only a SUBSET of columns from the file.
# Parquet stores columns in separate chunks, so this is very efficient:
# unrequested columns are never read from disk at all.
#
# This.to_table(columns=["a", "b"]) reads only those columns.
# Contrast with: read all, then filter → wastes I/O and RAM.

def demo_column_projection(data_dir: str) -> None:
    """Compare full table read vs column projection."""
    dataset = ds.dataset(data_dir, format="parquet")
    print(f"  Dataset schema: {dataset.schema.names}")
    print(f"  Files: {len(dataset.files)}")

    # Full scan
    t0 = time.perf_counter()
    full_table = dataset.to_table()
    t_full = time.perf_counter() - t0
    full_mb = full_table.nbytes / 1e6

    # Column projection
    t0 = time.perf_counter()
    proj_table = dataset.to_table(columns=["id", "feature_a"])
    t_proj = time.perf_counter() - t0
    proj_mb = proj_table.nbytes / 1e6

    print(f"\n  Full scan:        {full_mb:.1f} MB, {t_full*1000:.1f} ms")
    print(f"  Column projection:{proj_mb:.1f} MB, {t_proj*1000:.1f} ms")
    print(f"  Memory saved: {(1 - proj_mb/full_mb)*100:.0f}%")
    print(f"  Time saved:   {(1 - t_proj/t_full)*100:.0f}%")
    print(f"  Note: time savings are proportional to I/O, so larger files = bigger win")


# ---------------------------------------------------------------------------
# 4. ARROW → PANDAS CONVERSION COSTS
# ---------------------------------------------------------------------------
#
# table.to_pandas() converts an Arrow Table to a pandas DataFrame.
# The cost depends on the column types:
#
# ZERO-COPY (no data movement):
#   - Numeric types without nulls (int64, float64, etc.) that have a single
#     validity buffer with no nulls → Arrow buffer IS a numpy buffer.
#   - Arrow and NumPy share the same memory layout for these types.
#
# ALWAYS COPIES:
#   - String/dict columns (Arrow uses offset arrays; pandas uses object array)
#   - Nullable numeric types (Arrow uses a null bitmap; pandas needs NaN)
#   - List/struct types
#   - When the data spans multiple Arrow chunks
#
# table.to_pandas(split_blocks=True, self_destruct=True)
#   - self_destruct=True: DESTROYS the Arrow table as it converts, freeing
#     Arrow memory immediately → lower peak RAM.
#   - split_blocks=True: creates one numpy block per column → allows
#     partial conversion.

def demo_arrow_to_pandas_cost(data_dir: str) -> None:
    """Show conversion costs for different column types."""
    dataset = ds.dataset(data_dir, format="parquet")
    table = dataset.to_table()

    print(f"  Arrow table: {table.num_rows:,} rows × {table.num_columns} cols")
    print(f"  Arrow memory: {table.nbytes / 1e6:.1f} MB")

    # Standard conversion
    t0 = time.perf_counter()
    df_standard = table.to_pandas()
    t_standard = time.perf_counter() - t0

    # Conversion with self_destruct (can't reuse table afterward)
    table2 = dataset.to_table()  # re-read
    t0 = time.perf_counter()
    df_destruct = table2.to_pandas(self_destruct=True)
    t_destruct = time.perf_counter() - t0

    print(f"\n  to_pandas() (standard):      {t_standard*1000:.1f} ms")
    print(f"  to_pandas(self_destruct):    {t_destruct*1000:.1f} ms")
    print(f"  DataFrame memory: {df_standard.memory_usage(deep=True).sum()/1e6:.1f} MB")


# ---------------------------------------------------------------------------
# 5. ZERO-COPY INSPECTION
# ---------------------------------------------------------------------------
#
# We can check whether an Arrow array's buffer IS the numpy array's buffer.

def demo_zero_copy_check(data_dir: str) -> None:
    """Check which columns are zero-copy Arrow → numpy."""
    dataset = ds.dataset(data_dir, format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()

    print("  Zero-copy check (Arrow → numpy):")
    for col_name in table.schema.names:
        arrow_col = table.column(col_name)
        pd_arr = df[col_name]

        # Check: does the pandas column's underlying array share memory
        # with the Arrow column's buffer?
        try:
            np_arr = pd_arr.to_numpy()
            arrow_buf = arrow_col.buffers()[-1]  # data buffer (not validity/offsets)
            if arrow_buf is not None:
                # Check if numpy array starts within the arrow buffer
                np_ptr = np_arr.ctypes.data
                arrow_ptr = arrow_buf.address
                in_range = arrow_ptr <= np_ptr < arrow_ptr + arrow_buf.size
                status = "zero-copy ✓" if in_range else "copied"
            else:
                status = "null buffer"
        except Exception:
            status = "can't check"

        print(f"    {col_name:<15} dtype={str(df[col_name].dtype):<10} → {status}")


# ---------------------------------------------------------------------------
# 6. READING ONE COLUMN AT A TIME
# ---------------------------------------------------------------------------
#
# In the shared memory pipeline, a worker reads its feature column from disk
# (via the dataset) rather than from shared memory.
#
# Pattern:
#   table_col = data.to_table(columns=[col_name])
#   series = table_col.to_pandas()[col_name]
#   raw = np.ascontiguousarray(series.to_numpy())
#
# Why read one column at a time?
#   - Only that column occupies RAM in the worker.
#   - Other workers can simultaneously read different columns.
#   - No co-ordination needed (each worker reads its own column independently).

def demo_single_column_read(data_dir: str) -> None:
    """Show single-column read pattern."""
    dataset = ds.dataset(data_dir, format="parquet")

    cols_to_read = ["feature_a", "feature_b", "feature_c"]
    print("  Single-column reads:")
    for col in cols_to_read:
        t0 = time.perf_counter()
        table = dataset.to_table(columns=[col])
        series = table.to_pandas()[col]
        raw = np.ascontiguousarray(series.to_numpy())
        elapsed = time.perf_counter() - t0
        print(f"    {col}: shape={raw.shape}, dtype={raw.dtype}, "
              f"{raw.nbytes/1e6:.2f} MB, time={elapsed*1000:.1f} ms")


# ---------------------------------------------------------------------------
# 7. PYARROW IN A MULTIPROCESSING CONTEXT
# ---------------------------------------------------------------------------
#
# Can a pyarrow.dataset.Dataset be passed to workers via pickle?
# YES — datasets are picklable (they store file paths, not file handles).
# Workers can re-open the dataset and scan their assigned column independently.
#
# However, re-opening the dataset in each worker means:
#   - File I/O happens in the worker (parallelises I/O)
#   - No coordination needed between workers
#   - Each worker pays the scan cost for its column only
#
# This is the typical read-compute-return pattern in PyArrow-based pipelines.

def worker_read_and_summarise(
    dataset_path: str,
    col_name: str,
) -> dict:
    """Worker: read one column from disk and return summary stats."""
    worker_ds = ds.dataset(dataset_path, format="parquet")
    table = worker_ds.to_table(columns=[col_name])
    series = table.to_pandas()[col_name]
    raw = np.ascontiguousarray(series.to_numpy())

    return {
        "col": col_name,
        "mean": float(raw.mean()),
        "std":  float(raw.std()),
        "min":  float(raw.min()),
        "max":  float(raw.max()),
        "nbytes": raw.nbytes,
    }


def demo_parallel_dataset_reads(data_dir: str) -> None:
    """Parallel column reads using ProcessPoolExecutor."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    feature_cols = ["feature_a", "feature_b", "feature_c",
                    "heavy_col_1", "heavy_col_2", "heavy_col_3"]

    print(f"  Parallel reads of {len(feature_cols)} columns:")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(worker_read_and_summarise, data_dir, col): col
            for col in feature_cols
        }
        for future in as_completed(futures):
            r = future.result()
            print(f"    {r['col']:>12}: mean={r['mean']:+.4f}, "
                  f"std={r['std']:.4f}, {r['nbytes']/1e6:.1f} MB")
    elapsed = time.perf_counter() - t0
    print(f"  Wall time: {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# 8. COMBINED PATTERN: PYARROW + SHARED MEMORY
# ---------------------------------------------------------------------------
#
# For maximum efficiency, combine both approaches:
#   - "Always needed" columns (small, needed by every task) → shared memory
#   - Feature columns (one per task, large) → read per-worker from dataset
#
# This avoids:
#   - Re-reading always-needed columns from disk for every task.
#   - Keeping ALL feature columns in RAM simultaneously.
#
# Trade-off:
#   - Shared memory: faster (RAM), requires upfront allocation.
#   - Per-worker dataset read: slower (I/O), but no pre-allocation needed.
#   - Choose based on file size, I/O bandwidth, and RAM constraints.

def explain_combined_pattern():
    print("""
  Combined pattern: PyArrow Dataset + Shared Memory

  Always-needed columns (e.g. mask, weights, row IDs):
    → Load once in parent
    → Write to SharedMemory
    → Workers attach by name (zero IPC copy)

  Feature columns (one per task):
    → Pass only dataset path + column name to worker
    → Worker reads from disk independently
    → Worker discards after task completes

  This is the design used in production feature engineering pipelines
  where you need to run the same set of always-needed columns against
  many different feature columns in parallel.
  """)


# ---------------------------------------------------------------------------
# DEMO RUNNER
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODULE 09 — PyArrow Datasets and Conversion Costs")
    print("=" * 60)

    if not _HAS_PYARROW:
        print("\nPyArrow not installed. Install with: pip install pyarrow")
        print("Skipping all demos.")
        return

    # Create a temporary dataset
    tmp_dir = tempfile.mkdtemp(prefix="pyarrow_course_")
    try:
        print(f"\n  Creating sample dataset in {tmp_dir}...")
        create_sample_dataset(tmp_dir, n_rows=120_000, n_files=3)

        print("\n[1] Column projection (read only needed columns)")
        demo_column_projection(tmp_dir)

        print("\n[2] Arrow → pandas conversion costs")
        demo_arrow_to_pandas_cost(tmp_dir)

        print("\n[3] Zero-copy check")
        demo_zero_copy_check(tmp_dir)

        print("\n[4] Single-column read pattern")
        demo_single_column_read(tmp_dir)

        print("\n[5] Parallel column reads with ProcessPoolExecutor")
        demo_parallel_dataset_reads(tmp_dir)

        print("\n[6] Combined pattern: PyArrow + Shared Memory")
        explain_combined_pattern()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  Cleaned up {tmp_dir}")

    print("\n[7] Key takeaways")
    print("""
    - pyarrow.dataset.Dataset is a lazy reference — no I/O until scan.
    - Column projection: pass columns= to to_table() → reads only those columns.
      For Parquet, unrequested columns are NEVER read from disk.
    - Arrow → pandas is zero-copy for plain numeric columns without nulls.
      Strings, nullable types, and chunked columns always copy.
    - table.to_pandas(self_destruct=True) frees Arrow memory as it converts
      → lower peak RAM during conversion.
    - Datasets are picklable → safe to pass to workers.
    - Pattern: workers read their own column from the dataset independently.
      Parallelises I/O and computation simultaneously.
    - For always-needed columns: prefer shared memory (faster than re-reading).
    - For feature columns: read per-worker from dataset (avoids holding all in RAM).
    """)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
#
# Exercise 1 — Column projection benchmarking
# --------------------------------------------
# Create a Parquet file with 20 columns and 1M rows.
# Benchmark read time and memory for:
#   a) Full scan (all 20 columns)
#   b) 1 column
#   c) 5 columns
#   d) 10 columns
# Plot (or print) memory vs columns. Is it linear?
#
# Exercise 2 — Predicate pushdown
# ---------------------------------
# Use pyarrow.dataset.Expression to filter rows BEFORE loading into RAM:
#   import pyarrow.dataset as ds
#   expr = ds.field("category") == "X"
#   table = dataset.to_table(filter=expr)
# Compare: load all, then filter in pandas vs predicate pushdown.
# Measure memory and time for both.
#
# Exercise 3 — Chunked column concatenation
# -------------------------------------------
# When you read a multi-file dataset, each column is a ChunkedArray
# (multiple Arrow buffers, one per file).
# table.column("feature_a").num_chunks tells you how many chunks.
# Does to_pandas() on a chunked column return a view or a copy?
# Verify using np.shares_memory.
#
# Exercise 4 — Streaming batches
# --------------------------------
# Use dataset.to_batches(batch_size=10_000) to read the data in
# streaming fashion without loading everything into RAM at once.
# Write a worker that reads batches, computes mean per batch, and
# returns a grand mean.
# Compare peak memory vs loading the full table.
