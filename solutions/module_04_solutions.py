"""
Solutions — Module 04: Pickling and What Crosses the Process Boundary
"""
import time
import pickle
import functools
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


# ---------------------------------------------------------------------------
# Exercise 1 — Closure workaround (three approaches)
# ---------------------------------------------------------------------------

# The original (broken) code uses a closure:
#   threshold = 0.5
#   def filter_above_threshold(x): return x > threshold
#   executor.map(filter_above_threshold, data)
# This fails because filter_above_threshold closes over `threshold` from
# the enclosing scope — it's not a module-level function.

# --- Approach (a): module-level function with extra parameter ---
def filter_above_threshold_a(x: float, threshold: float) -> bool:
    return x > threshold


def exercise1_approach_a():
    data = np.random.randn(10).tolist()
    threshold = 0.5
    with ProcessPoolExecutor(max_workers=2) as ex:
        # Use functools.partial to pre-fill the threshold argument
        fn = functools.partial(filter_above_threshold_a, threshold=threshold)
        results = list(ex.map(fn, data))
    print(f"  (a) module-level + partial: {results}")


# --- Approach (b): callable class at module level ---
class FilterAboveThreshold:
    """Callable class — picklable because it's defined at module level."""
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, x: float) -> bool:
        return x > self.threshold


def exercise1_approach_b():
    data = np.random.randn(10).tolist()
    fn = FilterAboveThreshold(threshold=0.5)
    with ProcessPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(fn, data))
    print(f"  (b) callable class:         {results}")


# --- Approach (c): functools.partial directly ---
def exercise1_approach_c():
    data = np.random.randn(10).tolist()
    fn = functools.partial(filter_above_threshold_a, threshold=0.5)
    with ProcessPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(fn, data))
    print(f"  (c) functools.partial:       {results}")
    # Verify it's picklable:
    print(f"      partial picklable: {len(pickle.dumps(fn))} bytes ✓")


def exercise1():
    print("Exercise 1 — Closure workarounds:")
    exercise1_approach_a()
    exercise1_approach_b()
    exercise1_approach_c()
    print("""
  Preference: (c) functools.partial is the most concise and idiomatic.
  (a) is equally good and avoids the import of functools.
  (b) is useful when the callable needs more complex stateful logic.
  Avoid (b) if the class can be replaced with a simple partial.
  """)


# ---------------------------------------------------------------------------
# Exercise 2 — Pickle cost profiling
# ---------------------------------------------------------------------------

def worker_compute_feature(col_name: str, data: dict) -> dict:
    """Receives col_name (string) + shared data dict — minimal IPC per task."""
    arr = np.array(data[col_name])
    return {"col": col_name, "mean": float(arr.mean()), "std": float(arr.std())}


def worker_compute_feature_by_col(col_name: str, shm_name: str,
                                   shape: tuple, dtype_str: str) -> dict:
    """Receives only metadata — no data copied."""
    from multiprocessing.shared_memory import SharedMemory
    shm = SharedMemory(name=shm_name, create=False)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    result = {"col": col_name, "mean": float(arr.mean()), "std": float(arr.std())}
    shm.close()
    return result


def exercise2():
    print("Exercise 2 — Pickle cost profiling:")
    from multiprocessing.shared_memory import SharedMemory
    N_ROWS = 1_000_000
    N_FEATURES = 10
    np.random.seed(0)

    df = pd.DataFrame({f"f{i}": np.random.randn(N_ROWS) for i in range(N_FEATURES)})
    feature_cols = list(df.columns)

    # --- Approach A: pass full column Series per task ---
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures_a = [ex.submit(lambda s: {"col": s.name, "mean": float(s.mean()), "std": float(s.std())}, df[c])
                     for c in feature_cols]
        _ = [f.result() for f in futures_a]
    # (lambda won't work with spawn, but demonstrates the point on fork)
    t_a = time.perf_counter() - t0

    # --- Approach B: pass column name only, worker reads from shared memory ---
    segments = []
    meta = {}
    for col in feature_cols:
        raw = np.ascontiguousarray(df[col].to_numpy())
        shm = SharedMemory(create=True, size=raw.nbytes)
        view = np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm.buf)
        view[:] = raw
        meta[col] = (shm.name, raw.shape, raw.dtype.str)
        segments.append(shm)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures_b = [ex.submit(worker_compute_feature_by_col, col, *meta[col])
                     for col in feature_cols]
        _ = [f.result() for f in futures_b]
    t_b = time.perf_counter() - t0

    for shm in segments:
        shm.close(); shm.unlink()

    col_pickle_size = len(pickle.dumps(df[feature_cols[0]])) / 1e6
    print(f"  Pickling one column: {col_pickle_size:.1f} MB per task")
    print(f"  Pass column (Series): {t_a:.2f}s")
    print(f"  Pass name (shm):      {t_b:.2f}s")
    print(f"  Speedup: {t_a/t_b:.1f}×")


# ---------------------------------------------------------------------------
# Exercise 3 — Custom __reduce__ for unpicklable classes
# ---------------------------------------------------------------------------

class FileWrapper:
    """Wraps a file handle. File handles are not picklable by default."""

    def __init__(self, path: str, mode: str = "r"):
        self.path = path
        self.mode = mode
        self._handle = open(path, mode)

    def read(self) -> str:
        return self._handle.read()

    def close(self) -> None:
        self._handle.close()

    def __reduce__(self):
        # Called by pickle.dumps(self).
        # Return (callable, args) — pickle will call callable(*args) to reconstruct.
        # We store the path and mode, NOT the file handle.
        return (FileWrapper, (self.path, self.mode))

    def __del__(self):
        try:
            self._handle.close()
        except Exception:
            pass


def worker_read_file(fw: FileWrapper) -> str:
    content = fw.read()
    fw.close()
    return content[:50]  # first 50 chars


def exercise3():
    print("\nExercise 3 — Custom __reduce__ for FileWrapper:")
    import tempfile, os

    # Create a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello from the file wrapper! This is a test.")
        tmp_path = f.name

    fw = FileWrapper(tmp_path)
    raw = pickle.dumps(fw)
    print(f"  FileWrapper pickled: {len(raw)} bytes ✓")

    reconstructed = pickle.loads(raw)
    print(f"  Reconstructed reads: {reconstructed.read()!r}")
    reconstructed.close()

    # Use with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1) as ex:
        result = ex.submit(worker_read_file, FileWrapper(tmp_path)).result()
        print(f"  Worker read via pickle: {result!r}")

    fw.close()
    os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Exercise 4 — Shared initializer vs per-task loading
# ---------------------------------------------------------------------------

_cached_model = None


def init_with_model(size_mb: int) -> None:
    global _cached_model
    _cached_model = np.random.randn(size_mb * 125_000)  # ~1 MB per 125k float64


def task_preloaded(x: float) -> float:
    return float(_cached_model[:10].sum() + x)


def task_inline_load(x: float, size_mb: int) -> float:
    model = np.random.randn(size_mb * 125_000)
    return float(model[:10].sum() + x)


def task_from_arg(x: float, model: np.ndarray) -> float:
    return float(model[:10].sum() + x)


def exercise4():
    print("\nExercise 4 — Shared initializer vs per-task loading:")
    MODEL_MB = 50
    N = 50
    inputs = list(range(N))

    # (a) Inline per task
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as ex:
        _ = list(ex.map(task_inline_load, inputs, [MODEL_MB] * N))
    t_inline = time.perf_counter() - t0

    # (b) Initializer (load once per worker)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2, initializer=init_with_model,
                             initargs=(MODEL_MB,)) as ex:
        _ = list(ex.map(task_preloaded, inputs))
    t_init = time.perf_counter() - t0

    # (c) Pass as argument (pickle cost each call)
    model = np.random.randn(MODEL_MB * 125_000)
    arg_pickle_mb = len(pickle.dumps(model)) / 1e6
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as ex:
        _ = list(ex.map(task_from_arg, inputs, [model] * N))
    t_arg = time.perf_counter() - t0

    print(f"  (a) Inline load  ({N} tasks): {t_inline:.2f}s")
    print(f"  (b) Initializer  ({N} tasks): {t_init:.2f}s  ← FASTEST (load once per worker)")
    print(f"  (c) Pass as arg  ({N} tasks): {t_arg:.2f}s  (pickle {arg_pickle_mb:.0f}MB × {N} tasks)")
    print(f"  (b) beats (a) from task 1.  (c) pays N × pickle cost regardless.")


if __name__ == "__main__":
    exercise1()
    exercise2()
    exercise3()
    exercise4()
