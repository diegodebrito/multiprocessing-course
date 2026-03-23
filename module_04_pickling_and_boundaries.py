"""
Module 04 — Pickling and What Crosses the Process Boundary
===========================================================

Learning objectives
-------------------
1. Understand what pickle is and how it serialises Python objects.
2. Know what IS and ISN'T picklable — and why.
3. Understand why module-level functions are required for worker tasks.
4. Quantify the cost of pickling large DataFrames and arrays.
5. Know strategies to minimise the pickle overhead in worker pipelines.

Run this module:
    python module_04_pickling_and_boundaries.py
"""

import os
import io
import sys
import time
import pickle
import pickletools
import textwrap
import numpy as np
import pandas as pd
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor


# ---------------------------------------------------------------------------
# 1. WHAT IS PICKLE?
# ---------------------------------------------------------------------------
#
# Pickle is Python's built-in serialisation protocol.
# It converts a Python object into a byte stream (serialise) and back (deserialise).
#
#   bytes  = pickle.dumps(obj)   → serialise
#   obj    = pickle.loads(bytes) → deserialise
#
# When ProcessPoolExecutor sends a task to a worker, it pickles:
#   1. The target function
#   2. All positional arguments
#   3. All keyword arguments
#
# When the worker sends the result back, it pickles the return value.
#
# Key implication: EVERYTHING that crosses the process boundary MUST be picklable.
# If it isn't, you get a PicklingError at submit() time, not at execution time.


# ---------------------------------------------------------------------------
# 2. WHAT IS PICKLABLE?
# ---------------------------------------------------------------------------
#
# Generally picklable:
#   ✓ Numbers (int, float, complex)
#   ✓ Strings, bytes, bytearrays
#   ✓ Tuples, lists, dicts, sets containing picklable items
#   ✓ None, True, False
#   ✓ Module-level functions (by reference to their fully-qualified name)
#   ✓ Module-level classes and their instances (if __reduce__ or __getstate__ work)
#   ✓ numpy arrays
#   ✓ pandas DataFrames and Series
#
# Generally NOT picklable:
#   ✗ Lambda functions (anonymous, no importable name)
#   ✗ Functions defined inside other functions (closures)
#   ✗ File handles, sockets, database connections
#   ✗ Threading/multiprocessing locks, events (sometimes)
#   ✗ Generator objects (iterators)
#   ✗ Some C extension objects (depends on implementation)


def demonstrate_picklability():
    """Try to pickle various objects and show what works and what doesn't."""
    import socket

    test_cases = [
        ("int 42",            42),
        ("float 3.14",        3.14),
        ("list [1,2,3]",      [1, 2, 3]),
        ("numpy array",       np.array([1, 2, 3], dtype=np.float64)),
        ("pandas Series",     pd.Series([1, 2, 3])),
        ("pandas DataFrame",  pd.DataFrame({"a": [1, 2], "b": [3, 4]})),
        ("module-level fn",   square),     # defined at module top level below
        ("lambda",            lambda x: x),
        ("generator",         (x for x in range(3))),
        ("open file",         None),       # We'll skip this; just explain
    ]

    print("  Object picklability test:")
    for name, obj in test_cases:
        if obj is None:
            print(f"    {'file handle':<25} → ✗  (not shown; file handles aren't picklable)")
            continue
        try:
            data = pickle.dumps(obj)
            _ = pickle.loads(data)
            print(f"    {name:<25} → ✓  ({len(data)} bytes)")
        except Exception as e:
            print(f"    {name:<25} → ✗  {type(e).__name__}: {e}")


# Module-level function — picklable because it has a stable import path.
def square(x: float) -> float:
    return x * x


# ---------------------------------------------------------------------------
# 3. WHY LAMBDAS AND CLOSURES FAIL
# ---------------------------------------------------------------------------
#
# Pickle serialises a function by storing its MODULE and QUALNAME, not its code.
# When deserialising, it imports the module and looks up the name.
#
# A lambda has no stable name in any module:
#   >>> import pickle
#   >>> pickle.dumps(lambda x: x)
#   AttributeError: Can't pickle local object '<lambda>'
#
# A closure (function defined inside another function) is in a local scope
# that can't be re-imported:
#   >>> def make_fn():
#   ...     def inner(x): return x
#   ...     return inner
#   >>> pickle.dumps(make_fn())  → AttributeError: local object 'make_fn.<locals>.inner'
#
# FIX: define the worker function at module top level.

def demonstrate_module_level_requirement():
    """Show why module-level functions are required for workers."""
    # This closure can't cross the process boundary:
    multiplier = 3
    def closure_fn(x):  # noqa: E306
        return x * multiplier  # captures `multiplier` from enclosing scope

    try:
        pickle.dumps(closure_fn)
        print("  closure_fn pickled (unexpected)")
    except (pickle.PicklingError, AttributeError) as e:
        print(f"  closure_fn ✗  {type(e).__name__}")

    # Workaround 1: use a module-level function with an extra parameter
    # (shown as `_process_one_feature(col_name, shm_name, ...)` in real code)
    print("  square (module-level) ✓  pickled in", len(pickle.dumps(square)), "bytes")

    # Workaround 2: functools.partial wraps a module-level function with preset args
    import functools
    times3 = functools.partial(square)  # trivial example; real use: preset config args
    data = pickle.dumps(times3)
    print(f"  functools.partial(square) ✓  pickled in {len(data)} bytes")


# ---------------------------------------------------------------------------
# 4. THE COST OF PICKLING LARGE DATA STRUCTURES
# ---------------------------------------------------------------------------
#
# Pickling is not free.  For large DataFrames, the cost can dominate the
# actual work you're trying to parallelize.
#
# Let's measure pickle round-trip times for different DataFrame sizes.

def measure_pickle_cost(df: pd.DataFrame, label: str) -> None:
    """Measure time and size to pickle and unpickle a DataFrame."""
    t0 = time.perf_counter()
    raw = pickle.dumps(df, protocol=5)
    t_dump = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pickle.loads(raw)
    t_load = time.perf_counter() - t0

    mb = len(raw) / 1e6
    print(f"    {label:<30}  "
          f"size={mb:.1f} MB  "
          f"dumps={t_dump*1000:.1f}ms  "
          f"loads={t_load*1000:.1f}ms")


def demo_pickle_cost():
    """Compare pickle overhead for DataFrames of increasing size."""
    print("  Pickle cost for DataFrames (protocol=5):")
    for rows in [10_000, 100_000, 1_000_000]:
        df = pd.DataFrame({
            "a": np.random.randn(rows),
            "b": np.random.randn(rows),
            "c": np.arange(rows, dtype=np.int64),
        })
        mb_actual = df.memory_usage(deep=True).sum() / 1e6
        measure_pickle_cost(df, f"{rows:>9,} rows ({mb_actual:.0f} MB)")


# ---------------------------------------------------------------------------
# 5. PROTOCOL 5 AND BUFFER PROTOCOL
# ---------------------------------------------------------------------------
#
# pickle.dumps(obj, protocol=5) — Python 3.8+
#
# Protocol 5 introduced "out-of-band" buffers via the buffer_callback mechanism.
# For numpy arrays (which implement the buffer protocol), protocol 5 can avoid
# an extra copy during serialisation by handing the buffer directly to the
# OS pipe rather than copying into the pickle byte stream.
#
# ProcessPoolExecutor uses protocol 5 by default since Python 3.8.
# This is why pickling large numpy arrays is faster than you might expect.

def show_protocol_versions():
    arr = np.ones((1_000_000,), dtype=np.float64)
    print(f"  numpy array (8 MB), pickle protocol comparison:")
    for proto in [2, 4, 5]:
        t0 = time.perf_counter()
        for _ in range(10):
            raw = pickle.dumps(arr, protocol=proto)
        elapsed = (time.perf_counter() - t0) / 10 * 1000
        print(f"    protocol={proto}: {len(raw)/1e6:.2f} MB, "
              f"avg dumps time: {elapsed:.2f}ms")


# ---------------------------------------------------------------------------
# 6. STRATEGIES TO MINIMISE PICKLE OVERHEAD
# ---------------------------------------------------------------------------
#
# Strategy 1: Pass IDENTIFIERS, not data
# ---------------------------------------
# Instead of:  executor.submit(process_row, df.iloc[i])   ← pickles a row
# Do:          executor.submit(process_row, i)            ← pickles an int
# The worker reconstructs the row from a shared resource (file, shared memory, …)
#
# Strategy 2: Use shared memory for large read-only arrays
# --------------------------------------------------------
# Write the big array into a shared memory segment ONCE.
# Pass just the NAME (a short string) to workers.
# Workers attach by name (no copy) and work on their slice.
# This is the subject of modules 05 and 08.
#
# Strategy 3: Use initializer + initargs for large static data
# ------------------------------------------------------------
# If every worker needs the same large data (e.g. a lookup table):
#   Pass it via initargs when constructing the pool.
#   It's pickled ONCE per worker at initialisation, not once per task.
#
# Strategy 4: Use memory-mapped files (np.memmap)
# ------------------------------------------------
# Workers can memory-map the same file independently — each attaches to the
# OS page cache, which may share physical pages.
#
# Strategy 5: Return small results
# ---------------------------------
# Even if the input is large, design your worker to return a small result
# (e.g. a dict of scalars, not a full DataFrame).
# The return value is also pickled and sent back through the IPC pipe.

# Example of Strategy 1:
def process_row_by_index(row_idx: int) -> float:
    """Worker that receives a row INDEX instead of the actual data.
    In a real system, it would look up the row in a shared resource."""
    # Simulate work
    return row_idx * 0.1


def demo_strategy_pass_index():
    """Demonstrate passing indices instead of data to workers."""
    print("  Strategy: pass indices, not data")
    N = 100
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_row_by_index, range(N)))
    print(f"    Processed {N} rows by passing index (pickled {sys.getsizeof(0)} bytes each)")
    print(f"    Compare to pickling a 1000-row Series: "
          f"~{len(pickle.dumps(pd.Series(range(1000))))} bytes each")


# ---------------------------------------------------------------------------
# 7. INSPECTING THE PICKLE STREAM
# ---------------------------------------------------------------------------
#
# pickletools.dis() disassembles the pickle byte stream to show what's in it.
# Useful for understanding why something is large or can't be pickled.

def demo_pickle_inspection():
    """Show the pickle bytecode for a simple function and a small object."""
    print("  Pickle stream for square (module-level function):")
    stream = io.StringIO()
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = stream
    pickletools.dis(pickle.dumps(square))
    _sys.stdout = old_stdout
    output = stream.getvalue()
    # Print just first few lines
    lines = output.strip().split('\n')
    for line in lines[:8]:
        print(f"    {line}")
    if len(lines) > 8:
        print(f"    ... ({len(lines)} lines total)")


# ---------------------------------------------------------------------------
# DEMO RUNNER
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODULE 04 — Pickling and What Crosses the Process Boundary")
    print("=" * 60)

    print("\n[1] What is and isn't picklable?")
    demonstrate_picklability()

    print("\n[2] Why module-level functions are required")
    demonstrate_module_level_requirement()

    print("\n[3] Pickle cost for large DataFrames")
    demo_pickle_cost()

    print("\n[4] Pickle protocol versions")
    show_protocol_versions()

    print("\n[5] Strategy: pass indices, not data")
    demo_strategy_pass_index()

    print("\n[6] Pickle stream inspection")
    demo_pickle_inspection()

    print("\n[7] Key takeaways")
    print("""
    - Everything crossing the process boundary MUST be picklable.
    - Pickle serialises functions by MODULE + QUALNAME — no lambdas or closures.
    - Module-level functions work because they can be re-imported by the child.
    - Large DataFrames are picklable but expensive (time + memory).
    - Protocol 5 uses the buffer protocol for numpy arrays (faster for large arrays).
    - The key design pattern: pass small identifiers (names, indices) to workers;
      workers access large data via shared memory, memory-mapped files, or
      initializer-loaded resources.
    - Return small results from workers to avoid expensive return-value pickling.
    """)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
#
# Exercise 1 — Closure workaround
# ---------------------------------
# You have this code that doesn't work:
#
#   threshold = 0.5
#   def filter_above_threshold(x): return x > threshold
#   executor.map(filter_above_threshold, data)
#
# Rewrite it so it works with ProcessPoolExecutor using:
#   a) A module-level function with an extra parameter
#   b) A class with __call__ at module level
#   c) functools.partial
# Which approach do you prefer? Why?
#
# Exercise 2 — Pickle cost profiling
# ------------------------------------
# Build a pipeline that processes 10 "features" from a DataFrame with 1M rows.
# Profile time spent in: (a) pickling inputs, (b) actual worker computation,
# (c) pickling results.
# Then redesign it to pass only column names and measure the improvement.
#
# Exercise 3 — Custom __reduce__ for unpicklable classes
# -------------------------------------------------------
# Create a class that holds a file handle (which is normally unpicklable).
# Implement __reduce__ to make it picklable by storing the filename instead
# of the handle, and reopening the file on reconstruction.
# (This pattern is useful for database connections, for example.)
#
# Exercise 4 — Shared initializer vs per-task loading
# -----------------------------------------------------
# Worker function: apply a "feature transformer" (simulate with a 50 MB numpy array).
# Compare:
#   a) Create the array inside the task function (every call)
#   b) Create the array in the initializer and store as a global
#   c) Pass the array as an argument (pickle cost)
# Time 100 tasks with each approach. What's the optimal approach?
