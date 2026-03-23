"""
Solutions — Module 09: PyArrow Datasets
"""
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False
    print("pyarrow not installed — skipping solutions.")


def make_dataset(tmp_dir: str, n_rows: int = 500_000, n_cols: int = 20) -> str:
    np.random.seed(42)
    df = pd.DataFrame({
        f"col_{i:02d}": np.random.randn(n_rows).astype(np.float64)
        for i in range(n_cols)
    } | {"category": np.random.choice(["A", "B", "C"], n_rows)})
    pq.write_table(pa.Table.from_pandas(df), os.path.join(tmp_dir, "data.parquet"))
    return tmp_dir


# ---------------------------------------------------------------------------
# Exercise 1 — Column projection benchmarking
# ---------------------------------------------------------------------------

def exercise1(tmp_dir: str) -> None:
    print("Exercise 1 — Column projection scaling:")
    dataset = ds.dataset(tmp_dir, format="parquet")
    all_cols = [c for c in dataset.schema.names if c.startswith("col_")]

    for n_cols in [1, 5, 10, 20]:
        cols = all_cols[:n_cols]
        t0 = time.perf_counter()
        table = dataset.to_table(columns=cols)
        elapsed = time.perf_counter() - t0
        mb = table.nbytes / 1e6
        print(f"  {n_cols:>2} columns: {mb:>6.1f} MB, {elapsed*1000:.1f} ms")

    print("  Memory is linear in columns (each col is independent in Parquet).")
    print("  I/O time scales with actual bytes read — projection is very efficient.")


# ---------------------------------------------------------------------------
# Exercise 2 — Predicate pushdown
# ---------------------------------------------------------------------------

def exercise2(tmp_dir: str) -> None:
    print("\nExercise 2 — Predicate pushdown vs filter-after-load:")
    dataset = ds.dataset(tmp_dir, format="parquet")

    # (a) Load all, then filter in pandas
    t0 = time.perf_counter()
    full = dataset.to_table().to_pandas()
    filtered_pandas = full[full["category"] == "A"]
    t_pandas = time.perf_counter() - t0

    # (b) Predicate pushdown in Arrow
    expr = ds.field("category") == "A"
    t0 = time.perf_counter()
    filtered_arrow = dataset.to_table(filter=expr).to_pandas()
    t_arrow = time.perf_counter() - t0

    print(f"  Load-all then filter: {t_pandas*1000:.0f}ms, rows={len(filtered_pandas):,}")
    print(f"  Predicate pushdown:   {t_arrow*1000:.0f}ms, rows={len(filtered_arrow):,}")
    # For a single-file dataset, pushdown may not save much I/O
    # (Parquet page-level filtering still reads most of the file).
    # It's more impactful with partitioned datasets where whole files are skipped.
    print("  Note: single-file Parquet limits pushdown gains. Partitioned datasets")
    print("  show bigger wins because entire partition files can be skipped.")


# ---------------------------------------------------------------------------
# Exercise 3 — Chunked column zero-copy check
# ---------------------------------------------------------------------------

def exercise3(tmp_dir: str) -> None:
    print("\nExercise 3 — Chunked column zero-copy:")
    # Create a multi-file dataset (= chunked columns)
    multi_dir = tempfile.mkdtemp()
    try:
        np.random.seed(0)
        for i in range(3):
            df = pd.DataFrame({"val": np.random.randn(10_000)})
            pq.write_table(pa.Table.from_pandas(df), os.path.join(multi_dir, f"part_{i}.parquet"))

        dataset = ds.dataset(multi_dir, format="parquet")
        table = dataset.to_table()
        col = table.column("val")
        print(f"  num_chunks: {col.num_chunks}")

        df = table.to_pandas()
        np_arr = df["val"].to_numpy()

        # Check zero-copy: chunked arrays must be combined → always copies
        try:
            arrow_ptr = col.chunk(0).buffers()[-1].address
            np_ptr = np_arr.ctypes.data
            zero_copy = arrow_ptr == np_ptr
        except Exception:
            zero_copy = False

        print(f"  Chunked → pandas zero-copy: {zero_copy}")
        print("  Chunked columns ALWAYS copy when converting to pandas (numpy needs")
        print("  a single contiguous buffer, but Arrow uses separate chunk buffers).")
        print("  Use table.combine_chunks() first if zero-copy matters.")
    finally:
        shutil.rmtree(multi_dir)


# ---------------------------------------------------------------------------
# Exercise 4 — Streaming batches
# ---------------------------------------------------------------------------

def exercise4(tmp_dir: str) -> None:
    print("\nExercise 4 — Streaming batches vs full load:")
    dataset = ds.dataset(tmp_dir, format="parquet")
    BATCH_SIZE = 50_000

    # Streaming: compute mean per batch, then grand mean
    t0 = time.perf_counter()
    batch_means = []
    batch_sizes = []
    for batch in dataset.to_batches(batch_size=BATCH_SIZE, columns=["col_00"]):
        arr = batch.column("col_00").to_pydict()["col_00"]
        arr = np.array(arr)
        batch_means.append(arr.mean())
        batch_sizes.append(len(arr))

    grand_mean_streaming = np.average(batch_means, weights=batch_sizes)
    t_streaming = time.perf_counter() - t0

    # Full load
    t0 = time.perf_counter()
    full_arr = dataset.to_table(columns=["col_00"]).to_pandas()["col_00"].to_numpy()
    grand_mean_full = full_arr.mean()
    t_full = time.perf_counter() - t0

    print(f"  Streaming ({len(batch_means)} batches of {BATCH_SIZE:,}): "
          f"grand_mean={grand_mean_streaming:.6f}, time={t_streaming*1000:.0f}ms")
    print(f"  Full load: grand_mean={grand_mean_full:.6f}, time={t_full*1000:.0f}ms")
    print(f"  Results match: {abs(grand_mean_streaming - grand_mean_full) < 1e-10}")
    print("  Streaming peak RAM ≈ one batch; full load holds entire column in RAM.")
    print("  For datasets that don't fit in RAM, streaming is essential.")


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _HAS_PYARROW:
        print("Install pyarrow to run these solutions.")
    else:
        tmp = tempfile.mkdtemp()
        try:
            make_dataset(tmp, n_rows=500_000, n_cols=20)
            exercise1(tmp)
            exercise2(tmp)
            exercise3(tmp)
            exercise4(tmp)
        finally:
            shutil.rmtree(tmp)
