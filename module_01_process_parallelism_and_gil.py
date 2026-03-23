"""
Module 01 — Process-Based Parallelism and the GIL
==================================================

Learning objectives
-------------------
1. Understand what the GIL is and why it exists.
2. Distinguish CPU-bound from I/O-bound work and know which benefits from processes.
3. Understand process overhead vs thread overhead.
4. Know when to reach for multiprocessing vs threading vs asyncio.

Run this module:
    python module_01_process_parallelism_and_gil.py
"""

# ---------------------------------------------------------------------------
# 1. THE GLOBAL INTERPRETER LOCK (GIL)
# ---------------------------------------------------------------------------
#
# CPython (the standard Python interpreter) has a mutex called the GIL.
# Only ONE thread may execute Python bytecode at a time — even on a
# multi-core machine.
#
# Why does the GIL exist?
#   CPython's memory management (reference counting) is NOT thread-safe at
#   the C level.  Instead of adding fine-grained locks around every object,
#   the designers added one coarse lock around the whole interpreter.  This
#   keeps the implementation simple and safe.
#
# What the GIL does NOT prevent:
#   - Parallel I/O: a thread that is blocking on a syscall (read, recv, …)
#     releases the GIL so other threads can run.
#   - C extensions that explicitly release the GIL (NumPy does this for many
#     array operations, as does pandas for some paths).
#
# Key insight: if your Python code is CPU-bound (pure computation, not waiting
# for I/O), adding threads will NOT make it run faster on multiple cores.
# The GIL ensures only one thread runs at a time.

import os
import time
import threading
import multiprocessing
from multiprocessing import Process


# ---------------------------------------------------------------------------
# 2. A CONCRETE DEMONSTRATION
# ---------------------------------------------------------------------------
#
# We'll run a CPU-bound task:
#   - single-threaded (baseline)
#   - with N threads (should NOT be faster than baseline for CPU-bound work)
#   - with N processes (SHOULD be faster)

def cpu_bound_task(n: int) -> int:
    """Count down from n — pure Python CPU work, no I/O."""
    total = 0
    for i in range(n):
        total += i * i
    return total


def run_single(n: int, repeats: int) -> float:
    """Run the task `repeats` times sequentially. Returns elapsed seconds."""
    start = time.perf_counter()
    for _ in range(repeats):
        cpu_bound_task(n)
    return time.perf_counter() - start


def run_threaded(n: int, repeats: int) -> float:
    """
    Run the task `repeats` times using threads.

    For CPU-bound work this should be about the same as (or slower than)
    single-threaded because the GIL serialises execution.  Thread overhead
    (creation, context switching, GIL contention) can even make it slower.
    """
    threads = [threading.Thread(target=cpu_bound_task, args=(n,)) for _ in range(repeats)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


def run_multiprocess(n: int, repeats: int) -> float:
    """
    Run the task `repeats` times using processes.

    Each process has its OWN interpreter and its OWN GIL, so they truly run
    in parallel on separate CPU cores.
    """
    processes = [Process(target=cpu_bound_task, args=(n,)) for _ in range(repeats)]
    start = time.perf_counter()
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# 3. PROCESS OVERHEAD
# ---------------------------------------------------------------------------
#
# Spawning a new OS process is far heavier than spawning a thread:
#
#   Thread creation:  ~10–50 µs   (shared memory space, just a new stack)
#   Process creation: ~1–50 ms    (new memory space, file descriptors, …)
#
# On Linux with `fork` start method (the default), forking is faster than
# spawning from scratch because the OS uses copy-on-write to clone memory.
# But the cost is still orders of magnitude more than creating a thread.
#
# The practical implication: multiprocessing only pays off when the work done
# per task is large enough to amortise the process-creation and IPC overhead.
# For very short tasks (< ~10 ms), use a pool rather than creating processes
# per task.

def measure_process_creation_overhead():
    """
    Measure how long it takes to start and join a do-nothing process.
    This is the baseline overhead you're paying regardless of task size.
    """
    def noop():
        pass

    trials = 5
    times = []
    for _ in range(trials):
        p = Process(target=noop)
        t0 = time.perf_counter()
        p.start()
        p.join()
        times.append(time.perf_counter() - t0)

    avg_ms = (sum(times) / len(times)) * 1000
    print(f"  Average process start+join overhead: {avg_ms:.1f} ms")


# ---------------------------------------------------------------------------
# 4. WHEN TO USE WHAT
# ---------------------------------------------------------------------------
#
# ┌─────────────────────┬──────────────┬────────────────────────────────────┐
# │ Workload type       │ Best tool    │ Why                                │
# ├─────────────────────┼──────────────┼────────────────────────────────────┤
# │ CPU-bound Python    │ processes    │ Each process has its own GIL       │
# │ CPU-bound NumPy     │ threads *or* │ NumPy releases the GIL during      │
# │                     │ processes    │ C-level array ops                  │
# │ I/O-bound           │ threads or   │ Threads release GIL while waiting  │
# │ (network, disk)     │ asyncio      │                                    │
# │ Mixed CPU+I/O       │ processes    │ Safest default; avoids GIL issues  │
# └─────────────────────┴──────────────┴────────────────────────────────────┘
#
# For data scientists:
#   - Feature engineering loops (pure Python logic per column/row): processes
#   - Reading many Parquet files in parallel: threads or asyncio often fine
#   - ML model training on a single process: processes (model.fit() keeps GIL)
#   - NumPy-heavy transforms: depends on whether NumPy ops release the GIL


# ---------------------------------------------------------------------------
# 5. PROCESSES AND MEMORY ISOLATION
# ---------------------------------------------------------------------------
#
# Unlike threads, processes do NOT share memory by default.
# Each process gets its own copy of:
#   - The Python interpreter
#   - All imported modules
#   - All variables and data structures
#
# This isolation is both the strength and the challenge:
#   + No race conditions on shared data
#   + No need for most locking
#   - You can't just pass a large DataFrame to a worker; it must be serialised
#     (pickled) and sent over IPC, which copies it
#   - Results must be serialised and sent back
#
# This is exactly why later modules introduce shared memory — it's the way to
# let processes READ a large dataset without copying it.

def demonstrate_isolation():
    """Show that child processes have their own memory: modifying a variable
    in the child does NOT affect the parent."""
    shared_list = [1, 2, 3]

    def mutate(lst):
        lst.append(99)
        print(f"  Child sees: {lst}  (pid {os.getpid()})")

    p = Process(target=mutate, args=(shared_list,))
    p.start()
    p.join()
    # The child appended 99, but the parent's list is unchanged.
    print(f"  Parent sees: {shared_list}  (pid {os.getpid()})")
    print("  -> Processes have isolated memory. Mutation in child is invisible to parent.")


# ---------------------------------------------------------------------------
# DEMO RUNNER
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODULE 01 — Process-Based Parallelism and the GIL")
    print("=" * 60)

    # --- GIL demo ---
    print("\n[1] GIL demo: CPU-bound task, single vs threads vs processes")
    N = 2_000_000
    REPEATS = 4
    cpu_count = os.cpu_count()
    print(f"    CPU cores available: {cpu_count}")
    print(f"    Task: sum of squares from 0..{N:,}  ×{REPEATS}")

    t_single = run_single(N, REPEATS)
    print(f"    Single-threaded : {t_single:.2f}s")

    t_threaded = run_threaded(N, REPEATS)
    print(f"    {REPEATS} threads      : {t_threaded:.2f}s  "
          f"({'SLOWER' if t_threaded > t_single else 'faster'} — GIL effect)")

    # For multiprocessing, only run if there's more than 1 core
    if cpu_count and cpu_count > 1:
        t_mp = run_multiprocess(N, REPEATS)
        speedup = t_single / t_mp
        print(f"    {REPEATS} processes    : {t_mp:.2f}s  (≈{speedup:.1f}× speedup)")
    else:
        print("    (Skipping multiprocessing demo: only 1 CPU core detected)")

    # --- Overhead demo ---
    print("\n[2] Process creation overhead")
    measure_process_creation_overhead()

    # --- Isolation demo ---
    print("\n[3] Memory isolation between parent and child")
    demonstrate_isolation()

    print("\n[4] Key takeaways")
    print("""
    - The GIL means Python threads cannot run CPU-bound code in parallel.
    - Processes each have their own GIL → true CPU parallelism.
    - Process creation is expensive (~ms); pools amortise that cost.
    - Process memory is isolated by default → no accidental sharing,
      but data must be explicitly transferred (serialised) across the boundary.
    - The next modules show how to deal with that transfer cost efficiently.
    """)


if __name__ == "__main__":
    # Guard is mandatory for multiprocessing on spawn-based platforms.
    # On Linux/fork it's optional, but always include it — it's good practice
    # and required for code that needs to run on macOS/Windows too.
    main()


# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
#
# Exercise 1 — Threads vs processes on NumPy work
# ------------------------------------------------
# Write a function `numpy_task(n)` that creates a random (n, n) float64 array
# and computes `np.linalg.svd(arr)`.  Run it 4 times:
#   a) sequentially
#   b) with 4 threads
#   c) with 4 processes
# Compare the timings.  You may find that threads are competitive here because
# numpy's SVD releases the GIL.  Does the result surprise you?
#
# Exercise 2 — Measure IPC cost
# -----------------------------
# Write a worker function that receives a pandas DataFrame (100k rows × 10 cols)
# as an argument and immediately returns its shape.
# Use ProcessPoolExecutor.submit() to run it.
# Time just the round-trip (submit → result).
# Then repeat with a DataFrame of 1M rows × 10 cols.
# How does the round-trip time scale?  (Hint: it's proportional to pickle size.)
#
# Exercise 3 — Isolation vs sharing mental model
# -----------------------------------------------
# Without running any code, predict what the following snippet prints:
#
#   import multiprocessing
#   counter = 0
#   def increment():
#       global counter
#       counter += 1
#       print(f"child counter = {counter}")
#   p = multiprocessing.Process(target=increment)
#   p.start(); p.join()
#   print(f"parent counter = {counter}")
#
# Then run it and verify your prediction.
