"""
Solutions — Module 03: ProcessPoolExecutor and Futures
"""
import time
import random
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Exercise 1 — Parallel feature computation with as_completed
# ---------------------------------------------------------------------------

def compute_stats(col_name: str, data_dict: dict) -> tuple:
    """Receives column name and a dict of {col: list} — minimal pickle cost."""
    import scipy.stats  # noqa: F401 — skew available only with scipy
    arr = np.array(data_dict[col_name])
    mean = float(arr.mean())
    std = float(arr.std())
    # Pearson skewness approximation (no scipy needed)
    skew = float(3 * (mean - float(np.median(arr))) / (std + 1e-12))
    return col_name, mean, std, skew


def exercise1():
    print("Exercise 1 — Parallel feature stats:")
    N_ROWS = 500_000
    N_COLS = 20
    np.random.seed(42)
    # Pass data as dict-of-lists (picklable, but still not ideal for large N)
    data = {f"col_{i}": np.random.randn(N_ROWS).tolist() for i in range(N_COLS)}

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(compute_stats, col, data): col for col in data}
        for future in as_completed(futures):
            col, mean, std, skew = future.result()
            print(f"  {col}: mean={mean:+.4f}, std={std:.4f}, skew={skew:+.4f}")
    print(f"  Total: {time.perf_counter()-t0:.2f}s")


# ---------------------------------------------------------------------------
# Exercise 2 — Exception isolation
# ---------------------------------------------------------------------------

def worker_maybe_fail(x: int) -> int:
    if x < 0:
        raise ValueError(f"Negative input not allowed: {x}")
    return x * x


def exercise2():
    print("\nExercise 2 — Exception isolation:")
    inputs = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
    successes = []
    failures = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(worker_maybe_fail, x): x for x in inputs}
        for future in as_completed(futures):
            x = futures[future]
            exc = future.exception()
            if exc:
                failures.append((x, str(exc)))
            else:
                successes.append((x, future.result()))

    print(f"  Successes: {sorted(successes)}")
    print(f"  Failures:  {failures}")
    print(f"  Main process never crashed ✓")


# ---------------------------------------------------------------------------
# Exercise 3 — Initializer for heavy resources
# ---------------------------------------------------------------------------

_heavy_model = None  # module-level storage for per-worker resource


def init_model(size_mb: int) -> None:
    global _heavy_model
    _heavy_model = np.random.randn(size_mb * 1024 * 1024 // 8)  # ~size_mb MB


def task_with_preloaded_model(x: float) -> float:
    return float(_heavy_model[:100].sum() + x)


def task_with_inline_model(x: float, size_mb: int) -> float:
    model = np.random.randn(size_mb * 1024 * 1024 // 8)  # loaded every call!
    return float(model[:100].sum() + x)


def exercise3():
    print("\nExercise 3 — Initializer vs inline model loading:")
    MODEL_MB = 50
    N_TASKS = 20
    inputs = list(range(N_TASKS))

    # (a) Model loaded per task
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as ex:
        results_a = list(ex.map(task_with_inline_model,
                                inputs, [MODEL_MB] * N_TASKS))
    t_inline = time.perf_counter() - t0

    # (b) Model loaded in initializer
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2, initializer=init_model,
                             initargs=(MODEL_MB,)) as ex:
        results_b = list(ex.map(task_with_preloaded_model, inputs))
    t_init = time.perf_counter() - t0

    print(f"  Inline model  ({N_TASKS} tasks): {t_inline:.2f}s")
    print(f"  Initializer   ({N_TASKS} tasks): {t_init:.2f}s")
    print(f"  Speedup: {t_inline/t_init:.1f}×")
    print(f"  (b) wins from task 1 because model is loaded ONCE per worker, not per task")


# ---------------------------------------------------------------------------
# Exercise 4 — map() vs as_completed() ordering
# ---------------------------------------------------------------------------

def timed_task(args: tuple) -> tuple:
    task_id, sleep_s = args
    time.sleep(sleep_s)
    return task_id, sleep_s


def exercise4():
    print("\nExercise 4 — map() vs as_completed() ordering:")
    import random
    random.seed(7)
    tasks = [(i, random.uniform(0.05, 0.4)) for i in range(8)]

    # map() — returns in SUBMISSION ORDER
    print("  map() results (always submission order):")
    with ProcessPoolExecutor(max_workers=4) as ex:
        for task_id, sleep_s in ex.map(timed_task, tasks):
            print(f"    task {task_id} (sleep={sleep_s:.2f}s)")

    # as_completed() — returns in COMPLETION ORDER
    print("  as_completed() results (completion order):")
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(timed_task, t): t[0] for t in tasks}
        for future in as_completed(futures):
            task_id, sleep_s = future.result()
            print(f"    task {task_id} (sleep={sleep_s:.2f}s)")

    print("""
  map() holds results in a queue and yields them in submission order,
  even if task 7 finishes before task 0.  This is simpler but means
  fast tasks at the end of the list can't be processed early.
  as_completed() yields tasks as they finish — better for pipelines
  where you want to process results progressively or show progress.
  """)


if __name__ == "__main__":
    exercise1()
    exercise2()
    exercise3()
    exercise4()
