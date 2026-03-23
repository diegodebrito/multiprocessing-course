"""
Solutions — Module 01: Process-Based Parallelism and the GIL
"""
import os
import time
import threading
import multiprocessing
from multiprocessing import Process
import numpy as np
import pandas as pd
import pickle


# ---------------------------------------------------------------------------
# Exercise 1 — Threads vs processes on NumPy work
# ---------------------------------------------------------------------------
#
# Key insight: NumPy's SVD releases the GIL for the LAPACK/BLAS C-level calls.
# This means threads CAN run in parallel for numpy-heavy work.
# Result: threads ≈ processes for numpy-heavy tasks (on this machine).

def numpy_task(n: int) -> None:
    arr = np.random.rand(n, n)
    np.linalg.svd(arr, full_matrices=False)


def benchmark_numpy_parallelism(n: int = 200, repeats: int = 4):
    print(f"numpy_task: SVD of ({n}×{n}) matrix, ×{repeats}")

    # Sequential
    t0 = time.perf_counter()
    for _ in range(repeats):
        numpy_task(n)
    t_seq = time.perf_counter() - t0

    # Threads
    threads = [threading.Thread(target=numpy_task, args=(n,)) for _ in range(repeats)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    t_thr = time.perf_counter() - t0

    # Processes
    procs = [Process(target=numpy_task, args=(n,)) for _ in range(repeats)]
    t0 = time.perf_counter()
    for p in procs: p.start()
    for p in procs: p.join()
    t_proc = time.perf_counter() - t0

    print(f"  Sequential: {t_seq:.2f}s")
    print(f"  Threads:    {t_thr:.2f}s  (speedup vs seq: {t_seq/t_thr:.1f}×)")
    print(f"  Processes:  {t_proc:.2f}s  (speedup vs seq: {t_seq/t_proc:.1f}×)")
    print("  Note: for numpy-heavy tasks, threads can be competitive with processes.")


# ---------------------------------------------------------------------------
# Exercise 2 — Measure IPC cost
# ---------------------------------------------------------------------------

def worker_return_shape(df: pd.DataFrame) -> tuple:
    return df.shape


def benchmark_ipc_cost():
    from concurrent.futures import ProcessPoolExecutor
    print("\nIPC round-trip cost (pickled DataFrame):")
    for rows in [10_000, 100_000, 1_000_000]:
        df = pd.DataFrame(np.random.randn(rows, 10), columns=[f"c{i}" for i in range(10)])
        pickle_size_mb = len(pickle.dumps(df)) / 1e6
        with ProcessPoolExecutor(max_workers=1) as ex:
            t0 = time.perf_counter()
            future = ex.submit(worker_return_shape, df)
            _ = future.result()
            elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  {rows:>9,} rows: pickle={pickle_size_mb:.1f}MB, round-trip={elapsed_ms:.0f}ms")


# ---------------------------------------------------------------------------
# Exercise 3 — Isolation mental model
# ---------------------------------------------------------------------------

def increment():
    global counter  # This refers to the CHILD's own copy of counter
    counter += 1
    print(f"  child counter = {counter}")


counter = 0


def exercise3():
    print("\nMemory isolation prediction:")
    print("  Expected: child prints 1, parent prints 0 (isolation!)")
    p = Process(target=increment)
    p.start()
    p.join()
    print(f"  parent counter = {counter}")


if __name__ == "__main__":
    benchmark_numpy_parallelism()
    benchmark_ipc_cost()
    exercise3()
