"""
Module 03 — ProcessPoolExecutor and Futures
============================================

Learning objectives
-------------------
1. Understand the ProcessPoolExecutor API: submit(), map(), shutdown().
2. Understand Future objects and how to collect results.
3. Use as_completed() for early results and progress tracking.
4. Handle exceptions raised inside workers.
5. Understand worker initialisation (initializer / initargs).
6. Know when to use map() vs submit() + as_completed().

Run this module:
    python module_03_process_pool_executor.py
"""

import os
import time
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, Future


# ---------------------------------------------------------------------------
# 1. WHAT IS ProcessPoolExecutor?
# ---------------------------------------------------------------------------
#
# ProcessPoolExecutor is the high-level interface to a pool of worker processes.
# It lives in `concurrent.futures` (Python 3.2+) and wraps the lower-level
# `multiprocessing.Pool` API with a cleaner, more Pythonic interface.
#
# A pool works like this:
#
#   Parent process
#   ├─ submits tasks → serialises (function, args) via pickle → IPC queue
#   ├─ Worker 0 ─────── picks up task ─────────────────────────── puts result in result queue
#   ├─ Worker 1 ─────── picks up task ─────────────────────────── puts result in result queue
#   └─ ...collects results from result queue → deserialises → returns to caller
#
# The pool is persistent: workers are started once (not per task), which
# amortises the process-creation overhead across many tasks.


# ---------------------------------------------------------------------------
# 2. BASIC USAGE: submit() and Future.result()
# ---------------------------------------------------------------------------
#
# submit(fn, *args, **kwargs) → Future
#   Schedules `fn(*args, **kwargs)` to run in a worker.
#   Returns a Future immediately (non-blocking).
#
# future.result(timeout=None) → value
#   Blocks until the worker finishes and returns the result.
#   If the worker raised an exception, future.result() RE-RAISES it.
#   This is a critical gotcha: exceptions are NOT silently swallowed.

def square(x: float) -> float:
    """A simple CPU-bound task."""
    return x * x


def demo_basic_submit():
    print("  Basic submit() and future.result():")
    with ProcessPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(square, 4)
        f2 = executor.submit(square, 7)
        # Both tasks are now running (or queued) in worker processes.
        # We haven't blocked yet.
        print(f"    f1 done? {f1.done()}")  # Likely False — still running
        r1 = f1.result()  # Blocks until f1 completes
        r2 = f2.result()
        print(f"    square(4) = {r1},  square(7) = {r2}")
    # The `with` block calls executor.shutdown(wait=True) on exit,
    # which blocks until all submitted futures are done and workers exit.


# ---------------------------------------------------------------------------
# 3. MAP: SIMPLER FOR UNIFORM WORKLOADS
# ---------------------------------------------------------------------------
#
# executor.map(fn, iterable) is like the built-in map() but runs in parallel.
# It returns results IN ORDER (same order as the input iterable).
# It blocks until ALL results are available (or raises on any exception).
#
# Compared to submit() + gather, map() is more concise but less flexible:
#   - You can't get "first finished" ordering.
#   - All exceptions bubble on the same call to next() on the iterator.
#   - No individual Future access for fine-grained cancellation.

def slow_square(x: float) -> float:
    """Simulates work that takes variable time."""
    time.sleep(random.uniform(0.05, 0.2))
    return x * x


def demo_map():
    print("\n  executor.map():")
    inputs = list(range(1, 9))
    with ProcessPoolExecutor(max_workers=4) as executor:
        t0 = time.perf_counter()
        results = list(executor.map(slow_square, inputs))
        elapsed = time.perf_counter() - t0
    print(f"    Inputs:  {inputs}")
    print(f"    Results: {results}")
    print(f"    Time: {elapsed:.2f}s  (sequential would be {sum(0.125 for _ in inputs):.2f}s avg)")


# ---------------------------------------------------------------------------
# 4. as_completed(): PROCESS RESULTS AS THEY FINISH
# ---------------------------------------------------------------------------
#
# as_completed(futures) yields Future objects as they complete.
# Order: first-finished, not submission order.
#
# Use this when:
#   - You want to process/log results as soon as they're ready.
#   - Tasks have heterogeneous runtimes and you don't want fast tasks to
#     wait for slow ones.
#   - You want early stopping: process the first N results then cancel the rest.

def variable_work(task_id: int) -> tuple[int, float]:
    """Returns (task_id, result) after a random sleep."""
    sleep_time = random.uniform(0.05, 0.3)
    time.sleep(sleep_time)
    return task_id, sleep_time * 100


def demo_as_completed():
    print("\n  as_completed() — results arrive in completion order:")
    task_ids = list(range(8))
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(variable_work, tid): tid for tid in task_ids}
        for future in as_completed(futures):
            tid, result = future.result()
            print(f"    task {tid} finished → result={result:.1f}")


# ---------------------------------------------------------------------------
# 5. EXCEPTION HANDLING
# ---------------------------------------------------------------------------
#
# If a worker raises an exception, the Future captures it.
# future.result() re-raises it in the calling process.
# future.exception() returns the exception without re-raising.
#
# This means: don't forget to call .result() or .exception() — otherwise
# exceptions are silently swallowed and you'll wonder why your pipeline
# seems to complete but nothing happened.

def sometimes_fails(x: int) -> int:
    if x % 3 == 0:
        raise ValueError(f"x={x} is divisible by 3 — task failed!")
    return x * 10


def demo_exception_handling():
    print("\n  Exception handling:")
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(sometimes_fails, i): i for i in range(6)}
        for future in as_completed(futures):
            i = futures[future]
            if future.exception() is not None:
                print(f"    task({i}) FAILED: {future.exception()}")
            else:
                print(f"    task({i}) succeeded: {future.result()}")


# ---------------------------------------------------------------------------
# 6. WORKER INITIALISATION: initializer and initargs
# ---------------------------------------------------------------------------
#
# Sometimes workers need to do setup once (load a model, open a DB connection,
# set up logging) before processing any tasks.
#
# executor = ProcessPoolExecutor(max_workers=4,
#                                initializer=setup_fn,
#                                initargs=(arg1, arg2))
#
# setup_fn(arg1, arg2) runs ONCE per worker process when it starts.
# You typically store results in a module-level (or global) variable so
# the task function can access it.
#
# This is valuable because:
#   - Loading a heavy resource per task is expensive.
#   - The initialiser runs ONCE per worker, not once per task.
#   - On fork, the initialiser is still called (the fork happens first,
#     then the initialiser runs in the child).

_worker_id = None          # Module-level: will be set by the initialiser


def worker_init(worker_name: str) -> None:
    """Initialiser: runs once per worker process at startup."""
    global _worker_id
    _worker_id = f"{worker_name}-{os.getpid()}"
    # In a real pipeline: load ML model, open DB connection, etc.


def task_using_worker_state(x: int) -> str:
    """Task that uses worker-level state set by the initialiser."""
    # _worker_id was set once in worker_init; this task just reads it.
    return f"[{_worker_id}] result={x * x}"


def demo_initializer():
    print("\n  Worker initializer (runs once per worker):")
    with ProcessPoolExecutor(
        max_workers=2,
        initializer=worker_init,
        initargs=("worker",),
    ) as executor:
        futures = [executor.submit(task_using_worker_state, i) for i in range(6)]
        for f in futures:
            print(f"    {f.result()}")


# ---------------------------------------------------------------------------
# 7. CHUNKSIZE: BATCHING SMALL TASKS
# ---------------------------------------------------------------------------
#
# executor.map(fn, iterable, chunksize=N)
#
# By default, each task is sent to a worker individually (IPC overhead per task).
# With chunksize=N, tasks are grouped into batches of N and sent together.
# This reduces the number of IPC round-trips for many small tasks.
#
# Rule of thumb: use chunksize when you have many thousands of cheap tasks.
# For expensive tasks (seconds each), chunksize=1 is fine.

def trivial_task(x: int) -> int:
    return x + 1


def demo_chunksize():
    print("\n  chunksize comparison for many cheap tasks:")
    N = 10_000
    inputs = range(N)

    with ProcessPoolExecutor(max_workers=4) as executor:
        t0 = time.perf_counter()
        _ = list(executor.map(trivial_task, inputs, chunksize=1))
        t_chunk1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = list(executor.map(trivial_task, inputs, chunksize=100))
        t_chunk100 = time.perf_counter() - t0

    print(f"    chunksize=1  : {t_chunk1:.3f}s")
    print(f"    chunksize=100: {t_chunk100:.3f}s")
    print(f"    Speedup from batching: {t_chunk1/t_chunk100:.1f}×")


# ---------------------------------------------------------------------------
# 8. SHUTDOWN AND CONTEXT MANAGER
# ---------------------------------------------------------------------------
#
# Always use ProcessPoolExecutor as a context manager (with statement).
# This calls executor.shutdown(wait=True) on exit, which:
#   1. Sends a sentinel to each worker to exit cleanly.
#   2. Waits for all submitted futures to complete.
#   3. Joins all worker processes.
#
# If you don't use a context manager:
#   - Workers may become orphaned (running after the parent exits).
#   - Futures may not be collected.
#
# executor.shutdown(wait=False) exits immediately without waiting,
# leaving futures running. Only do this if you have another mechanism
# to wait for completion.

def demo_context_manager_safety():
    print("\n  Context manager guarantees clean shutdown:")
    with ProcessPoolExecutor(max_workers=2) as executor:
        f = executor.submit(square, 9)
        # On __exit__, executor waits for f to complete before closing.
    # Here: all workers have exited. Safe to proceed.
    # f.result() is still accessible after shutdown (result is cached).
    print(f"    square(9) = {f.result()}  (accessible after shutdown)")


# ---------------------------------------------------------------------------
# DEMO RUNNER
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODULE 03 — ProcessPoolExecutor and Futures")
    print("=" * 60)

    print("\n[1] Basic submit() and result()")
    demo_basic_submit()

    print("\n[2] map() — ordered results")
    demo_map()

    print("\n[3] as_completed() — first-finished ordering")
    demo_as_completed()

    print("\n[4] Exception handling with futures")
    demo_exception_handling()

    print("\n[5] Worker initializer")
    demo_initializer()

    print("\n[6] chunksize for cheap tasks")
    demo_chunksize()

    print("\n[7] Context manager and shutdown")
    demo_context_manager_safety()

    print("\n[8] Key takeaways")
    print("""
    - ProcessPoolExecutor maintains a persistent pool — workers start once.
    - submit() is non-blocking; future.result() blocks until complete.
    - future.result() RE-RAISES worker exceptions in the parent — don't ignore futures.
    - as_completed() yields futures in finish order, not submission order.
    - Use initializer/initargs to set up per-worker resources once.
    - Use chunksize for many cheap tasks to reduce IPC overhead.
    - Always use the context manager (with) for clean shutdown.
    - CRITICAL: everything passed to a worker (function + args) must be picklable.
      This is the subject of the next module.
    """)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
#
# Exercise 1 — Parallel feature computation
# ------------------------------------------
# You have a DataFrame with 20 columns and 500k rows.
# Write a function `compute_stats(col_name, data_dict)` that returns
# (col_name, mean, std, skew) for the named column.
# Use ProcessPoolExecutor to compute stats for all 20 columns in parallel.
# Collect results with as_completed() and print as they arrive.
#
# Exercise 2 — Exception isolation
# ----------------------------------
# Write a worker that raises an exception if its input is negative.
# Submit 10 tasks: half with positive, half with negative inputs.
# Using as_completed(), collect the successful results and log the failures,
# WITHOUT crashing the main process.
# Verify that successful tasks still return valid results even when some fail.
#
# Exercise 3 — Initializer for heavy resources
# ---------------------------------------------
# Simulate loading a "heavy model" (just a large numpy array, ~100 MB) in the
# worker initializer.  Write the task function to use this pre-loaded model.
# Compare the total runtime when:
#   a) The model is loaded inside the task function (loaded per call).
#   b) The model is loaded in the initializer (loaded once per worker).
# How many tasks does it take before approach (b) wins?
#
# Exercise 4 — Understanding map() vs as_completed() ordering
# ------------------------------------------------------------
# Submit 8 tasks with random sleep times (0.05–0.5s each).
# Collect results with map() and record the order they were processed.
# Then do the same with submit() + as_completed().
# Show that map() returns in submission order while as_completed() returns
# in completion order.  Why does this matter for pipelines?
