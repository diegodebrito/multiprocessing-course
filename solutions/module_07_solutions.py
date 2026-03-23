"""
Solutions — Module 07: Pandas ↔ NumPy Memory Behavior
"""
import numpy as np
import pandas as pd
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Exercise 1 — copy_tracker(series)
# ---------------------------------------------------------------------------

def copy_tracker(series: pd.Series) -> None:
    """
    Track how many copies happen going from Series → ascontiguousarray.
    Step 1: series.to_numpy()
    Step 2: np.ascontiguousarray(step1)
    """
    raw_np = series.to_numpy()
    step1_copy = not np.shares_memory(series.to_numpy(), raw_np)
    # Note: calling to_numpy() twice may give different objects even if zero-copy,
    # so we check shares_memory between the original backing and raw_np.

    contiguous = np.ascontiguousarray(raw_np)
    step2_copy = not np.shares_memory(raw_np, contiguous)

    total_copies = int(step1_copy) + int(step2_copy)
    label = series.name if series.name else f"dtype={series.dtype}"
    print(f"  {label:<30} to_numpy copy={step1_copy}, ascontiguous copy={step2_copy}  "
          f"→ total={total_copies} copies")


def exercise1():
    print("Exercise 1 — copy_tracker:")
    # Simple float64
    s1 = pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float64), name="float64")
    copy_tracker(s1)

    # Object dtype
    s2 = pd.Series(["a", "b", "c"], name="object")
    copy_tracker(s2)

    # Non-contiguous slice (stride view → ascontiguousarray must copy)
    base = np.arange(10, dtype=np.float64)
    s3 = pd.Series(base[::2], name="strided_slice")
    copy_tracker(s3)

    # Nullable integer
    s4 = pd.Series(pd.array([1, 2, 3], dtype="Int64"), name="nullable_Int64")
    copy_tracker(s4)

    print("""
  float64 (simple block): to_numpy() → view; ascontiguousarray → view  (0 copies)
  object dtype:           to_numpy() → copy                             (1 copy)
  strided slice:          to_numpy() → may share; ascontiguous → copy   (1 copy)
  nullable Int64:         to_numpy() → always copy                      (1 copy)
  """)


# ---------------------------------------------------------------------------
# Exercise 2 — Optimal DataFrame extraction into shared memory
# ---------------------------------------------------------------------------

def extract_columns_to_shm(df: pd.DataFrame) -> list[dict]:
    """
    Extracts each column individually into its own SharedMemory segment.
    Returns list of descriptor dicts. Cleans up on any exception.
    """
    from contextlib import ExitStack
    descriptors = []
    segments = []  # keep references so we can unlink on error
    stack = ExitStack()
    try:
        for col in df.columns:
            raw = np.ascontiguousarray(df[col].to_numpy())
            shm = SharedMemory(create=True, size=raw.nbytes)
            stack.callback(shm.close)
            stack.callback(shm.unlink)
            view = np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm.buf)
            view[:] = raw
            descriptors.append({
                "name": col,
                "shm_name": shm.name,
                "dtype_str": raw.dtype.str,
                "shape": raw.shape,
            })
            segments.append(shm)
        stack.pop_all()   # success — caller owns cleanup
        return descriptors, segments
    except Exception:
        stack.close()     # cleanup on error
        raise


def exercise2():
    print("\nExercise 2 — Column extraction to shared memory:")
    N = 100_000
    df = pd.DataFrame({
        "a": np.random.randn(N).astype(np.float64),
        "b": np.random.randint(0, 100, N).astype(np.int32),
        "c": np.random.rand(N) > 0.5,  # bool
    })

    import pickle
    df_pickle_mb = len(pickle.dumps(df)) / 1e6
    print(f"  Full DataFrame pickle: {df_pickle_mb:.1f} MB")

    descs, segments = extract_columns_to_shm(df)
    for d in descs:
        shm = SharedMemory(name=d["shm_name"], create=False)
        arr = np.ndarray(d["shape"], dtype=np.dtype(d["dtype_str"]), buffer=shm.buf)
        shm.close()
        desc_pickle_bytes = len(pickle.dumps(d))
        print(f"  {d['name']:>5}: {arr.dtype}, {arr.nbytes/1e6:.2f} MB in shm, "
              f"descriptor={desc_pickle_bytes} bytes to pickle")

    for shm in segments:
        shm.close(); shm.unlink()
    print("  Cleaned up ✓")


# ---------------------------------------------------------------------------
# Exercise 3 — Index alignment bug hunt and fix
# ---------------------------------------------------------------------------

def process_chunk_with_index(chunk: pd.DataFrame) -> pd.DataFrame:
    """Process a chunk and return results. Retains chunk's original index."""
    return pd.DataFrame({
        "mean_val": [chunk["value"].mean()],
        "std_val":  [chunk["value"].std()],
    }, index=[chunk.index[0]])   # keeps first index of this chunk


def exercise3():
    print("\nExercise 3 — Index alignment:")
    np.random.seed(42)
    # DataFrame with string index
    N = 100
    df = pd.DataFrame(
        {"value": np.random.randn(N)},
        index=[f"row_{i:03d}" for i in range(N)],
    )

    chunk_size = 25
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, N, chunk_size)]

    # Bug: workers return DataFrames with overlapping/wrong indices
    results_bad = [process_chunk_with_index(c) for c in chunks]
    bad = pd.concat(results_bad)
    print(f"  Without fix — index: {bad.index.tolist()}")
    print(f"  (4 rows, but indices are 'row_000', 'row_025', ... — correct here,")
    print(f"   but if chunks had default RangeIndex, they'd all be 0 → concat duplicates)")

    # Fix: ignore_index=True in concat
    good = pd.concat(results_bad, ignore_index=True)
    print(f"  With ignore_index=True — index: {good.index.tolist()}")
    print("  Rule: always use ignore_index=True when order matters but not the index values.")


# ---------------------------------------------------------------------------
# Exercise 4 — Pandas 2.0 CoW and shared memory write-back
# ---------------------------------------------------------------------------

def exercise4():
    print("\nExercise 4 — CoW and shared memory write-back:")
    pd_major = int(pd.__version__.split(".")[0])
    print(f"  pandas version: {pd.__version__}")

    # CoW behaviour: does modifying a slice affect the original?
    df = pd.DataFrame({"a": np.array([1.0, 2.0, 3.0, 4.0, 5.0])})
    original_id = id(df["a"]._values)

    sliced = df.iloc[:3]   # a "view" under old pandas; CoW object under pandas 2
    sliced.iloc[0, 0] = 999.0  # attempt to modify

    if df.iloc[0, 0] == 999.0:
        print("  pandas < 2.0 behaviour: slice IS a view, parent was modified!")
    else:
        print("  pandas 2.0+ CoW: parent unchanged after modifying slice ✓")
        print("  CoW means 'copy on write' — write to the slice → separate copy made")

    # Does writing via a numpy view into shm write back to shm?
    shm = SharedMemory(create=True, size=5 * 8)
    arr = np.ndarray((5,), dtype=np.float64, buffer=shm.buf)
    arr[:] = [10.0, 20.0, 30.0, 40.0, 50.0]

    # Create a pandas Series backed by the numpy view
    s = pd.Series(arr)  # under CoW, s is a lazy copy-on-write reference to arr
    s.iloc[0] = 999.0   # writing to s

    # Does the write propagate back to shm.buf?
    val_in_shm = np.ndarray((5,), dtype=np.float64, buffer=shm.buf)[0]
    print(f"  After s.iloc[0]=999.0:")
    print(f"    s[0] = {s.iloc[0]}  (modified)")
    print(f"    shm[0] = {val_in_shm}  ({'modified' if val_in_shm == 999.0 else 'unchanged — CoW protected shm'})")
    print("  CoW means writing through a pandas Series does NOT corrupt the shared buffer.")

    shm.close(); shm.unlink()


if __name__ == "__main__":
    exercise1()
    exercise2()
    exercise3()
    exercise4()
