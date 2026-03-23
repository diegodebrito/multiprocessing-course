"""
Solutions — Module 02: Start Methods and Copy-on-Write
"""
import os
import sys
import gc
import time
import multiprocessing
from multiprocessing import Process

import numpy as np


# ---------------------------------------------------------------------------
# Exercise 1 — Observing COW with /proc
# ---------------------------------------------------------------------------
#
# Case (a): child reads arr.sum() → reads numpy buffer (raw C values),
#           no Python object header touched → very few pages copied.
# Case (b): child writes arr[:] = 0 → EVERY page is written → full COW copy.

def get_rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        return -1.0
    return -1.0


def child_read_only(arr: np.ndarray) -> None:
    _ = arr.sum()
    print(f"  [read-only child]  RSS = {get_rss_mb(os.getpid()):.0f} MB  (sum computed, pages shared)")


def child_write_all(arr: np.ndarray) -> None:
    arr[:] = 0          # writes every element → every page gets COW-copied
    print(f"  [write-all child]  RSS = {get_rss_mb(os.getpid()):.0f} MB  (pages copied due to write)")


def exercise1():
    if sys.platform != "linux":
        print("Exercise 1 requires Linux /proc — skipping.")
        return

    SIZE = 200_000_000 // 8  # 200 MB of float64
    arr = np.ones(SIZE, dtype=np.float64)
    parent_rss = get_rss_mb(os.getpid())
    print(f"Exercise 1 — COW observation with /proc:")
    print(f"  Parent RSS before fork: {parent_rss:.0f} MB  (array is {arr.nbytes/1e6:.0f} MB)")

    print("\n  Case (a): child only reads (arr.sum())")
    p = Process(target=child_read_only, args=(arr,))
    p.start(); p.join()

    print("\n  Case (b): child writes entire array (arr[:] = 0)")
    p = Process(target=child_write_all, args=(arr,))
    p.start(); p.join()

    print("""
  Expected:
    (a) child RSS ≈ parent RSS  (pages are still shared via COW)
    (b) child RSS ≈ parent RSS + array size  (all pages privately copied)
  Note: numpy reads don't touch Python object headers → minimal COW.
  """)


# ---------------------------------------------------------------------------
# Exercise 2 — Start method startup time
# ---------------------------------------------------------------------------

def noop(): pass


def exercise2():
    print("Exercise 2 — Start method startup time:")
    available = multiprocessing.get_all_start_methods()
    TRIALS = 5
    for method in available:
        ctx = multiprocessing.get_context(method)
        times = []
        for _ in range(TRIALS):
            p = ctx.Process(target=noop)
            t0 = time.perf_counter()
            p.start(); p.join()
            times.append(time.perf_counter() - t0)
        avg_ms = sum(times) / len(times) * 1000
        print(f"  {method:<12}: {avg_ms:.1f}ms avg over {TRIALS} trials")

    print("""
  Expected order: fork < forkserver ≈ spawn
  Why forkserver is not as fast as fork after the first worker:
    - forkserver forks FROM the clean server, not from the parent.
    - The fork itself is still fast (COW), but the server process must receive
      the function+args via IPC (pickled), which adds overhead vs raw fork.
    - forkserver startup (first worker) is slow (spawns the server process).
    - Subsequent workers fork from the server quickly, but still pay IPC cost.
  """)


# ---------------------------------------------------------------------------
# Exercise 3 — gc.freeze() and COW
# ---------------------------------------------------------------------------
#
# gc.freeze() moves all currently tracked objects to a permanent generation
# that the GC never collects — this also reduces refcount churn from GC cycles.
# It helps more for large dicts/lists of Python objects than for numpy arrays
# (numpy elements aren't Python objects, so there's no per-element refcount).

def child_iterate_list(data: list) -> None:
    total = sum(data)  # iterates all Python int objects — each read bumps refcount
    print(f"  [child] sum={total}, RSS={get_rss_mb(os.getpid()):.0f} MB")


def exercise3():
    if sys.platform != "linux":
        print("Exercise 3 requires Linux /proc — skipping.")
        return

    print("Exercise 3 — gc.freeze() and COW:")
    N = 5_000_000
    data = list(range(N))  # list of Python int objects

    # Without gc.freeze()
    gc.collect()
    parent_rss = get_rss_mb(os.getpid())
    print(f"\n  Without gc.freeze() (parent RSS: {parent_rss:.0f} MB):")
    p = Process(target=child_iterate_list, args=(data,))
    p.start(); p.join()

    # With gc.freeze()
    gc.freeze()
    gc.collect()
    parent_rss = get_rss_mb(os.getpid())
    print(f"\n  With gc.freeze() (parent RSS: {parent_rss:.0f} MB):")
    p = Process(target=child_iterate_list, args=(data,))
    p.start(); p.join()
    gc.unfreeze()

    print("""
  gc.freeze() reduces refcount churn from GC cycles (cyclic objects).
  For a plain list of ints, the benefit is small because:
    - Each list element is a Python int → reading it bumps its refcount → COW.
    - gc.freeze() prevents the GC from visiting these objects (reduces churn),
      but doesn't eliminate the per-access refcount write.
  For numpy arrays: no per-element refcount → COW-friendly regardless of gc.freeze().
  """)


if __name__ == "__main__":
    exercise1()
    print()
    exercise2()
    print()
    exercise3()
