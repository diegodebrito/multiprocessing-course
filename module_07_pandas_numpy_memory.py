"""
Module 07 — Pandas ↔ NumPy Memory Behavior: Copies vs Views
=============================================================

Learning objectives
-------------------
1. Know when pandas returns a view vs a copy of the underlying data.
2. Understand Series.to_numpy() and np.asarray(series) semantics.
3. Know when pd.concat(..., copy=False) actually avoids a copy.
4. Understand index alignment issues when concatenating results.
5. Know how to write zero-copy extraction code and when to give up and copy.

Run this module:
    python module_07_pandas_numpy_memory.py
"""

import numpy as np
import pandas as pd
import sys


# ---------------------------------------------------------------------------
# 1. THE FUNDAMENTAL QUESTION: VIEW OR COPY?
# ---------------------------------------------------------------------------
#
# A VIEW shares the underlying buffer with the source.
# A COPY is an independent allocation with its own data.
#
# Views are cheap (just metadata), copies cost time and memory.
#
# The challenge with pandas:
#   - Pandas 1.x often returned views when convenient, copies otherwise.
#   - Pandas 2.0 introduced Copy-on-Write (CoW) semantics by default in
#     pandas 2.0+, which changes many of these rules.
#   - pandas 2.0+ CoW: EVERY operation that "looks" like a view is actually
#     treated as copy-on-write — you can READ without copying, but the moment
#     you WRITE, a copy is made.
#
# For our purposes (feeding data into shared memory), we care about:
#   - Getting a numpy array from a Series/DataFrame to write into shared memory
#   - Knowing whether this causes an extra copy or not

def check_shares_memory(a, b, label: str) -> None:
    """Print whether two arrays share the same underlying memory."""
    try:
        shares = np.shares_memory(a, b)
    except Exception:
        shares = False
    print(f"  {label:<50} shares_memory={shares}")


# ---------------------------------------------------------------------------
# 2. Series.to_numpy() vs np.asarray(series)
# ---------------------------------------------------------------------------
#
# Series.to_numpy(dtype=None, copy=False, na_value=...)
#   - Returns a numpy array.
#   - copy=False (default): tries to return a view if possible.
#     But "if possible" means: only if the Series has a single numpy block
#     of the right dtype and is contiguous.
#   - If dtype is specified and different from series.dtype, ALWAYS copies.
#   - If the Series has object dtype or nullable dtype (pd.Int64Dtype), copies.
#
# np.asarray(series)
#   - Calls series.__array__() under the hood.
#   - Similar behaviour: view if possible, copy if not.
#   - Does NOT respect the na_value parameter.
#
# Recommendation:
#   Use series.to_numpy() for clarity.
#   ALWAYS wrap with np.ascontiguousarray() before writing to shared memory
#   to ensure a contiguous, owned buffer.

def demo_series_to_numpy():
    """Investigate when to_numpy returns a view vs a copy."""
    print("  Series.to_numpy() — view vs copy:")

    # Case 1: Simple float64 series backed by a numpy array
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    s_simple = pd.Series(arr)
    result = s_simple.to_numpy()
    check_shares_memory(arr, result, "float64 series, no dtype change")

    # Case 2: Request different dtype — always copies
    result_cast = s_simple.to_numpy(dtype=np.float32)
    check_shares_memory(arr, result_cast, "float64 series, to float32")

    # Case 3: Object dtype — always copies (objects aren't raw bytes)
    s_obj = pd.Series(["a", "b", "c"])
    result_obj = s_obj.to_numpy()
    print(f"  object dtype to_numpy: dtype={result_obj.dtype}")

    # Case 4: Nullable integer dtype (pd.Int64Dtype) — copies to numpy int64
    s_nullable = pd.array([1, 2, 3], dtype="Int64")  # nullable integer
    s_nullable_series = pd.Series(s_nullable)
    result_nullable = s_nullable_series.to_numpy(dtype=np.int64, na_value=0)
    print(f"  nullable int64 to_numpy(int64): dtype={result_nullable.dtype}, values={result_nullable}")

    # Practical recommendation for shared memory pipelines:
    print()
    print("  Recommendation for shared memory pipeline:")
    print("    raw = np.ascontiguousarray(series.to_numpy())")
    print("    This guarantees: C-contiguous, owned buffer, correct dtype.")
    print("    It may copy, but that's the price of a safe write into shm.")


# ---------------------------------------------------------------------------
# 3. DETECTING COPIES: np.shares_memory and id()
# ---------------------------------------------------------------------------
#
# np.shares_memory(a, b) → True if a and b share any memory elements.
# This is the most reliable way to check.
#
# arr.base is not None → the array is a view; arr.base is the owner.
# arr.base is None     → the array owns its data (or is a scalar).
#
# id() on the array itself doesn't help (it's the metadata object, not the buffer).

def demo_detecting_copies():
    """Show how to detect copies using np.shares_memory and .base."""
    original = np.arange(10, dtype=np.float64)
    view = original[::2]          # stride view — non-contiguous
    copy = original.copy()
    ascontiguous_of_view = np.ascontiguousarray(view)  # forces a copy (different strides)

    print("  Detecting copies with shares_memory and .base:")
    for name, arr in [("view", view), ("copy", copy), ("ascontiguous_of_view", ascontiguous_of_view)]:
        shares = np.shares_memory(original, arr)
        has_base = arr.base is not None
        print(f"    {name:<25} shares_memory={shares}, has_base={has_base}")


# ---------------------------------------------------------------------------
# 4. DataFrame.to_numpy() AND COLUMN DTYPES
# ---------------------------------------------------------------------------
#
# DataFrame.to_numpy() returns a 2D numpy array.
# CRITICAL: if the DataFrame has mixed dtypes (e.g. float64 and int32 columns),
# numpy must find a common dtype — this ALWAYS causes a copy and dtype promotion.
#
# For shared memory pipelines: extract ONE COLUMN AT A TIME (as a Series)
# to avoid dtype promotion and to keep the data contiguous per column.
# This is exactly what the ColDescriptor pattern does.

def demo_dataframe_to_numpy():
    """Show how mixed dtypes force a copy in DataFrame.to_numpy()."""
    df_uniform = pd.DataFrame({
        "a": np.random.randn(5),
        "b": np.random.randn(5),
    })  # all float64

    df_mixed = pd.DataFrame({
        "a": np.random.randn(5).astype(np.float64),
        "b": np.arange(5, dtype=np.int32),
    })  # float64 + int32

    arr_uniform = df_uniform.to_numpy()
    arr_mixed = df_mixed.to_numpy()

    print("  DataFrame.to_numpy() dtype behaviour:")
    print(f"    uniform float64 df → {arr_uniform.dtype}  "
          f"(may share memory: {np.shares_memory(arr_uniform, df_uniform['a'].to_numpy())})")
    print(f"    mixed float64+int32 df → {arr_mixed.dtype}  "
          f"(promoted to float64 — always a copy)")
    print()
    print("  Implication: extract columns individually for shared memory:")
    print("    for col in df.columns:")
    print("        raw = np.ascontiguousarray(df[col].to_numpy())")
    print("        # write raw into its own SharedMemory segment")


# ---------------------------------------------------------------------------
# 5. pd.concat() AND THE copy= PARAMETER
# ---------------------------------------------------------------------------
#
# pd.concat(frames, copy=False) — the copy=False hint tells pandas to avoid
# copying data if possible.  In practice:
#   - pandas 1.x: honours copy=False for same-dtype frames (returns views).
#   - pandas 2.0+ CoW: copy=False is largely ignored because CoW defers copies
#     until mutation. The parameter is a hint, not a guarantee.
#
# When DOES pd.concat guarantee no copy?
#   Practically: NEVER reliably, because:
#   - Index alignment may require reindexing (always copies).
#   - dtype promotion requires a new buffer.
#   - CoW means "no copy until write" — appears copy-free but copies on mutation.
#
# Rule of thumb: treat pd.concat() output as having its own memory.
# If you need to write the result into shared memory, do:
#   raw = np.ascontiguousarray(result[col].to_numpy())

def demo_concat_copy():
    """Investigate memory sharing after pd.concat."""
    n = 5
    a = pd.DataFrame({"x": np.ones(n), "y": np.arange(n, dtype=np.float64)})
    b = pd.DataFrame({"x": np.ones(n) * 2, "y": np.arange(n, 2 * n, dtype=np.float64)})

    combined = pd.concat([a, b], ignore_index=True)
    combined_no_copy = pd.concat([a, b], ignore_index=True, copy=False)

    col_a = a["x"].to_numpy()
    col_combined = combined["x"].to_numpy()
    col_combined_nc = combined_no_copy["x"].to_numpy()

    print("  pd.concat() memory sharing:")
    check_shares_memory(col_a, col_combined,    "concat (copy=True) shares with source")
    check_shares_memory(col_a, col_combined_nc, "concat (copy=False) shares with source")
    print()
    pd_version = pd.__version__
    print(f"  (pandas {pd_version}: copy=False is a hint; CoW semantics may apply)")
    print("  Lesson: don't rely on concat sharing memory. Treat result as independent.")


# ---------------------------------------------------------------------------
# 6. INDEX ALIGNMENT ISSUES
# ---------------------------------------------------------------------------
#
# When workers return DataFrames, their index matters for concatenation.
# Worker A might return index [0, 1, 2] and worker B also [0, 1, 2] (default RangeIndex).
# pd.concat([df_a, df_b]) with overlapping indices doesn't error but produces
# a confusing result.
#
# SAFE PATTERN: workers return results with a KNOWN RangeIndex or
# explicit alignment key.  Use ignore_index=True in the final concat.

def demo_index_alignment():
    """Show the index alignment problem and the fix."""
    # Simulate worker results
    def process_chunk(chunk: pd.Series) -> pd.DataFrame:
        """Returns a DataFrame with default RangeIndex."""
        return pd.DataFrame({
            "original": chunk.values,
            "squared": (chunk.values ** 2),
        })

    # Two chunks from different parts of the original data
    data = pd.Series(np.arange(10, dtype=np.float64))
    chunk_a = data.iloc[:5]   # index 0..4
    chunk_b = data.iloc[5:]   # index 5..9

    result_a = process_chunk(chunk_a)  # has index 0..4 (original index)
    result_b = process_chunk(chunk_b)  # has index 5..9 (original index)

    # Problem: concat without reset — index is preserved from the chunk
    bad_concat = pd.concat([result_a, result_b])
    print(f"  Concat without ignore_index: index={bad_concat.index.tolist()}")

    # Fix: use ignore_index=True to get a clean 0..N RangeIndex
    good_concat = pd.concat([result_a, result_b], ignore_index=True)
    print(f"  Concat with ignore_index=True: index={good_concat.index.tolist()}")
    print(f"  Values align correctly: {good_concat['original'].tolist()}")


# ---------------------------------------------------------------------------
# 7. ZERO-COPY EXTRACTION SUMMARY
# ---------------------------------------------------------------------------
#
# When does pandas ACTUALLY give you a zero-copy numpy array?
#
# ✓ Zero-copy (view):
#   - series.to_numpy() on a single-block, non-nullable, C-contiguous series
#     of the SAME dtype as the underlying numpy array.
#   - np.asarray(series) — same conditions.
#
# ✗ Always copies:
#   - dtype change (to_numpy(dtype=float32) on a float64 series)
#   - Nullable extension types (Int64Dtype, Float64Dtype, etc.)
#   - object dtype
#   - After many pandas operations (groupby, resample, etc.) that create new blocks
#   - np.ascontiguousarray() on a non-contiguous array
#
# Practical rule for shared memory pipelines:
#   DON'T try to avoid copies at the extraction stage.
#   The ONE copy from Series → numpy array is acceptable.
#   The big saving comes from NOT copying that numpy array AGAIN via pickle.
#   Shared memory eliminates the per-worker IPC copy.

def summarise_copy_behaviour():
    """Print a reference table of copy vs view behaviours."""
    rows = [
        ("series.to_numpy() (same dtype, simple block)", "view if possible"),
        ("series.to_numpy(dtype=other)",                 "always copy"),
        ("series.to_numpy() on nullable Int64",          "always copy"),
        ("np.asarray(series)",                           "view if possible"),
        ("np.ascontiguousarray(arr) — already contiguous", "view"),
        ("np.ascontiguousarray(arr) — non-contiguous",   "copy"),
        ("df.to_numpy() — uniform dtype",                "may share"),
        ("df.to_numpy() — mixed dtype",                  "always copy"),
        ("pd.concat(..., copy=False)",                   "hint; not guaranteed"),
        ("df[col].to_numpy()",                           "view if possible"),
    ]
    print("  Copy vs View reference table:")
    for op, behaviour in rows:
        print(f"    {op:<50} → {behaviour}")


# ---------------------------------------------------------------------------
# DEMO RUNNER
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODULE 07 — Pandas ↔ NumPy Memory Behavior")
    print("=" * 60)

    print(f"\n  Running with pandas {pd.__version__}")

    print("\n[1] Series.to_numpy() — view vs copy")
    demo_series_to_numpy()

    print("\n[2] Detecting copies with shares_memory and .base")
    demo_detecting_copies()

    print("\n[3] DataFrame.to_numpy() and mixed dtypes")
    demo_dataframe_to_numpy()

    print("\n[4] pd.concat() and the copy= parameter")
    demo_concat_copy()

    print("\n[5] Index alignment issues")
    demo_index_alignment()

    print("\n[6] Copy vs View reference table")
    summarise_copy_behaviour()

    print("\n[7] Key takeaways")
    print("""
    - series.to_numpy() TRIES to return a view; success depends on dtype and backing.
    - np.ascontiguousarray() guarantees a C-contiguous owned buffer (copies if needed).
    - Mixed-dtype DataFrames → always copy in to_numpy(). Extract columns individually.
    - pd.concat(copy=False) is a hint, not a guarantee. Treat concat output as a copy.
    - Use ignore_index=True when concatenating worker results to avoid index confusion.

    RULE FOR SHARED MEMORY PIPELINES:
      The expensive copy to AVOID is the per-worker IPC pickle copy.
      The one copy from Series → np.ascontiguousarray() is acceptable overhead.
      Write to shared memory ONCE, pass only the name to workers.
    """)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
#
# Exercise 1 — Copy tracking
# ---------------------------
# Write a function copy_tracker(series) that:
#   1. Calls series.to_numpy()
#   2. Calls np.ascontiguousarray() on the result
#   3. Reports how many copies happened (0, 1, or 2)
#      using np.shares_memory comparisons at each step.
# Test with: simple float64, object dtype, non-contiguous slice, nullable int.
#
# Exercise 2 — Optimal DataFrame extraction
# -------------------------------------------
# Given a DataFrame with 20 columns of mixed dtypes (float64, int32, bool):
# Write extract_columns_to_shm(df) that:
#   - Extracts each column individually (not df.to_numpy())
#   - Writes each into its own SharedMemory segment
#   - Returns a list of ColDescriptor-like dicts {name, shm_name, dtype_str, shape}
#   - Correctly cleans up all segments in a finally block
# Compare memory usage vs passing the whole DataFrame via pickle.
#
# Exercise 3 — Index alignment bug hunt
# ----------------------------------------
# Create a parallel processing pipeline using ProcessPoolExecutor where:
#   - Input: DataFrame with a non-default index (e.g. string labels)
#   - Workers process chunks and return DataFrames with the original index
#   - Final result is concatenated
# Introduce the index alignment bug intentionally, observe incorrect output,
# then fix it with ignore_index=True and verify correctness.
#
# Exercise 4 — Pandas 2.0 CoW semantics
# ----------------------------------------
# (Requires pandas 2.0+)
# Create a DataFrame, take a slice (df.iloc[:5]), and modify the slice.
# Under CoW semantics, does the original DataFrame change?
# How does this affect the shared memory pattern — if you create a numpy view
# over shm.buf and then modify it via a pandas Series, does it write back to shm?
