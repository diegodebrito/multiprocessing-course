"""
Module 02 — Start Methods: fork, spawn, forkserver & Copy-on-Write
===================================================================

Learning objectives
-------------------
1. Understand what the OS does during fork(), spawn, and forkserver.
2. Understand copy-on-write (COW) semantics and how they affect memory use.
3. Know which start method Linux uses by default and why it matters.
4. Understand why COW breaks down and when you actually pay the copy cost.
5. Know the implications for passing large datasets to worker processes.

Run this module:
    python module_02_start_methods_and_cow.py
"""

import os
import sys
import time
import ctypes
import struct
import multiprocessing
from multiprocessing import Process
import numpy as np


# ---------------------------------------------------------------------------
# 1. THE THREE START METHODS
# ---------------------------------------------------------------------------
#
# When you create a new child process, Python/the OS can use one of three
# mechanisms to initialise that child.  The choice determines:
#   - How fast the child starts
#   - What state the child inherits
#   - What must be pickled/serialised to pass data to the child
#
# ┌──────────────┬───────────────────────────────────────────────────────────┐
# │ Method       │ What happens                                              │
# ├──────────────┼───────────────────────────────────────────────────────────┤
# │ fork         │ OS clones the parent process via the fork() syscall.      │
# │ (Linux       │ The child gets a complete copy of the parent's address    │
# │  default)    │ space, file descriptors, and Python state — almost        │
# │              │ instantaneously thanks to copy-on-write (COW).            │
# │              │ No pickling needed for existing objects.                  │
# ├──────────────┼───────────────────────────────────────────────────────────┤
# │ spawn        │ A fresh Python interpreter is launched.                   │
# │ (macOS/Win   │ The child starts with nothing — it imports __main__       │
# │  default)    │ from scratch and then unpickles the target function and   │
# │              │ its arguments.  Everything passed to the child must be    │
# │              │ picklable.  Much slower to start.                         │
# ├──────────────┼───────────────────────────────────────────────────────────┤
# │ forkserver   │ A dedicated "server" process is spawned once.  After      │
# │              │ that, new workers are forked FROM the server (not from    │
# │              │ the parent).  The server is clean, so there's no risk of  │
# │              │ inheriting bad state from a long-running parent.          │
# └──────────────┴───────────────────────────────────────────────────────────┘
#
# TL;DR for Linux:
#   - Default is fork — fast, COW, no pickling required for globals.
#   - If you explicitly need cross-platform compatibility, use spawn.
#   - forkserver is a niche choice for long-running parent processes that
#     accumulate state you don't want children to inherit.


# ---------------------------------------------------------------------------
# 2. COPY-ON-WRITE (COW) — THE KEY LINUX CONCEPT
# ---------------------------------------------------------------------------
#
# After fork(), parent and child share the SAME physical memory pages.
# The OS marks all pages as "read-only, shared".
#
# As long as neither process WRITES to a page:
#   → Both processes read from the same physical RAM.  No copy.
#
# The moment either process WRITES to a page:
#   → The OS detects the write, creates a private copy of that page for the
#     writing process, and then allows the write.  The other process still
#     reads the original.
#
# This is why fork() is so cheap:
#   - Forking a process with a 4 GB in-memory dataset doesn't immediately use
#     8 GB of RAM.  If the children only read that dataset, they share the
#     same physical pages throughout their lifetime.
#
# Where COW BREAKS DOWN for Python:
#   - Python uses reference counting for garbage collection.
#   - Every time you READ an object, Python increments/decrements its
#     refcount — which is stored in the object itself.
#   - Reading an object WRITES to its refcount field → triggers a COW copy.
#   - This means that simply iterating over a large dictionary can cause
#     nearly every page to be privately copied, defeating COW benefits.
#
# Mitigations:
#   - Use Python 3.8+ with gc.freeze() to prevent GC tracking of long-lived
#     objects (reduces refcount churn).
#   - Use numpy arrays backed by shared memory — NumPy arrays don't have
#     per-element Python refcounts; accessing elements doesn't trigger COW.

def explain_cow_with_refcounts():
    """
    Demonstrates that reading Python objects causes COW copies via refcount
    updates, while accessing raw numpy arrays does not touch the object headers.
    """
    print("  COW and refcounts:")
    print("  - Python list of ints: each element is a Python int object.")
    print("    Reading list[i] increments int.__refcount__ → write → COW copy.")
    print("  - numpy array: elements are raw C values in a contiguous buffer.")
    print("    Reading arr[i] does NOT update any Python object header.")
    print("  → For read-only worker access, numpy arrays are much more COW-friendly.")


# ---------------------------------------------------------------------------
# 3. DEMONSTRATING START METHODS
# ---------------------------------------------------------------------------

def child_sees_parent_var(label: str, value: int) -> None:
    """Worker that prints a variable inherited from the parent."""
    # This variable was set in the parent before fork().
    # A fork child sees it; a spawn child does NOT (starts fresh).
    print(f"  [{label}] child pid={os.getpid()}, parent_global={value}")


PARENT_GLOBAL = 42  # This exists at module level before any child is created


def demo_fork_inherits_globals():
    """
    With fork, the child inherits module-level state from the parent.
    No pickling required — the variable just 'exists' in the cloned address space.
    """
    ctx = multiprocessing.get_context("fork")
    p = ctx.Process(target=child_sees_parent_var, args=("fork", PARENT_GLOBAL))
    p.start()
    p.join()


def demo_spawn_starts_fresh():
    """
    With spawn, the child imports __main__ fresh.  It WILL see PARENT_GLOBAL
    because importing the module sets it — but it cannot see in-memory state
    that was set AFTER import (e.g. variables mutated at runtime).
    """
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=child_sees_parent_var, args=("spawn", PARENT_GLOBAL))
    p.start()
    p.join()


# ---------------------------------------------------------------------------
# 4. PRACTICAL IMPLICATIONS FOR DATA PIPELINES
# ---------------------------------------------------------------------------
#
# SCENARIO: You load a 2 GB feature matrix into memory before launching workers.
#
# With FORK (Linux default):
#   - Workers start in microseconds (fork is cheap).
#   - Workers initially share the same physical RAM pages as the parent.
#   - IF workers only READ the matrix → near-zero additional RAM.
#   - IF workers modify the matrix → COW kicks in; modified pages are copied.
#   - Caveat: Python refcount churn still causes some COW even for reads.
#     Using numpy arrays mitigates this significantly.
#
# With SPAWN (macOS/Windows default, or explicit):
#   - Workers start from scratch — they have NO access to the parent's data.
#   - To share the 2 GB matrix you MUST either:
#       a) Pickle it and send it through the IPC pipe (very slow, 2× RAM).
#       b) Put it in shared memory BEFORE spawning workers (the correct approach).
#   - This is why the shared memory pattern (module 05) is so valuable on spawn.
#
# RECOMMENDATION for production:
#   - On Linux, fork is fine for read-mostly data sharing.
#   - For cross-platform code, design around spawn — use shared memory.
#   - Never mutate shared data structures from worker processes without locks.


# ---------------------------------------------------------------------------
# 5. HOW TO SET THE START METHOD
# ---------------------------------------------------------------------------

def show_current_start_method():
    """Show how to inspect and set the start method."""
    current = multiprocessing.get_start_method()
    print(f"  Current start method: {current!r}")
    print("  Available methods:", multiprocessing.get_all_start_methods())
    print()
    print("  To change the start method for your whole program:")
    print("    multiprocessing.set_start_method('spawn')  # call once, in if __name__=='__main__'")
    print()
    print("  To use a specific method for one pool/process without changing the global:")
    print("    ctx = multiprocessing.get_context('spawn')")
    print("    ctx.Process(target=...).start()")
    print("    ctx.Pool(4)")


# ---------------------------------------------------------------------------
# 6. VISUALISING MEMORY USAGE WITH COW
# ---------------------------------------------------------------------------
#
# We can observe COW by watching /proc/<pid>/status (VmRSS) before and after
# the child reads a large numpy array.

def get_rss_mb(pid: int) -> float:
    """Read resident set size (RSS) in MB from /proc on Linux."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024
    except FileNotFoundError:
        return -1.0
    return -1.0


def cow_memory_demo():
    """
    Parent allocates a large numpy array.
    Child reads it (doesn't modify).
    We compare RSS of parent before and after fork.

    On a real Linux system you'd see that child's RSS is much smaller than
    the array size because COW keeps pages shared.
    (In containers or when the kernel merges pages, numbers may vary.)
    """
    SIZE = 50_000_000  # 50M float64 = 400 MB
    arr = np.ones(SIZE, dtype=np.float64)

    parent_rss_before = get_rss_mb(os.getpid())

    def child_read_array():
        # Just access a few elements — not modifying
        total = arr[0] + arr[-1]  # triggers COW for refcount, but only a few pages
        child_rss = get_rss_mb(os.getpid())
        print(f"  Child RSS after reading 2 elements: {child_rss:.0f} MB")
        print(f"  (Array is {SIZE * 8 / 1e6:.0f} MB; child shares most pages via COW)")

    p = Process(target=child_read_array)
    p.start()
    p.join()

    parent_rss_after = get_rss_mb(os.getpid())
    print(f"  Parent RSS before fork: {parent_rss_before:.0f} MB")
    print(f"  Parent RSS after fork+join: {parent_rss_after:.0f} MB")
    print(f"  Parent RSS delta: {parent_rss_after - parent_rss_before:+.0f} MB")


# ---------------------------------------------------------------------------
# 7. THE SPAWN CONSTRAINT AND PICKLING PREVIEW
# ---------------------------------------------------------------------------
#
# With spawn, the child has to RECONSTRUCT the target function and arguments.
# It does this by:
#   1. Importing the module where the function is defined.
#   2. Unpickling the arguments.
#
# This means:
#   - Functions must be importable (defined at module top level, not inside
#     another function or as lambdas).
#   - All arguments must be picklable.
#
# We'll cover pickling in depth in module 04.  For now, keep in mind:
# fork sidesteps these constraints because no pickling is needed — the child
# already has everything from the cloned address space.

def demo_spawn_requires_picklable_args():
    """
    Show that spawn fails if the argument is not picklable.
    We'll use a lambda, which is not picklable.
    """
    import pickle
    fn = lambda x: x * 2  # noqa: E731
    try:
        pickle.dumps(fn)
        print("  Lambda pickled successfully (unexpected?)")
    except (pickle.PicklingError, AttributeError) as e:
        print(f"  Lambda cannot be pickled: {type(e).__name__}")
        print("  → This is why spawn workers need module-level named functions.")


# ---------------------------------------------------------------------------
# DEMO RUNNER
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODULE 02 — Start Methods and Copy-on-Write")
    print("=" * 60)

    print("\n[1] Current Python start method")
    show_current_start_method()

    print("\n[2] COW and Python refcount interaction")
    explain_cow_with_refcounts()

    print("\n[3] fork — child inherits parent globals")
    demo_fork_inherits_globals()

    if sys.platform != "win32":
        print("\n[4] Memory COW demo (Linux /proc)")
        cow_memory_demo()
    else:
        print("\n[4] Memory COW demo (skipped — Linux only)")

    print("\n[5] spawn — requires picklable arguments")
    demo_spawn_requires_picklable_args()

    if "fork" in multiprocessing.get_all_start_methods():
        print("\n[6] spawn — child starts fresh (slower)")
        # NOTE: on Linux this spawns a whole new Python, takes ~0.5–2s
        t0 = time.perf_counter()
        demo_spawn_starts_fresh()
        elapsed = time.perf_counter() - t0
        print(f"  spawn start time: {elapsed:.2f}s (vs fork which is <0.01s)")

    print("\n[7] Key takeaways")
    print("""
    - fork: child is a clone of parent (COW). Fast. No pickling needed.
      Linux default. Risk: inheriting unwanted state (open file handles, locks).
    - spawn: child starts fresh. Must pickle everything. Slow. macOS/Win default.
      Cross-platform safe. Forces you to be explicit about data transfer.
    - forkserver: fork from a clean server process. Niche use.

    - COW lets fork'd children share the parent's large read-only datasets
      almost for free — as long as workers don't write back to them.
    - Python object reads cause refcount writes → COW copies of touched pages.
      NumPy arrays avoid per-element refcount → more COW-friendly.
    - For spawn-based environments, shared memory is the solution for large
      datasets (covered in module 05).
    """)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
#
# Exercise 1 — Observing COW with /proc
# --------------------------------------
# Allocate a 200 MB numpy array in the parent.
# Fork a child that:
#   a) only reads arr.sum()  — print child RSS
#   b) writes arr[:] = 0    — print child RSS
# Compare the child RSS in cases (a) and (b).
# Explain the difference in terms of COW.
#
# Exercise 2 — Start method startup time
# ----------------------------------------
# Measure how long it takes to start-and-join a do-nothing Process using
# each available start method on your system.
# Expected order: fork < forkserver ≈ spawn.
# Why is forkserver not as fast as fork after the first worker?
#
# Exercise 3 — gc.freeze() and COW
# ----------------------------------
# import gc; gc.freeze() prevents the GC from tracking the frozen objects,
# which reduces the refcount churn that breaks COW.
# Allocate a large list of Python ints (NOT numpy), call gc.freeze(), then
# fork a child that iterates the list.
# Compare RSS before/after gc.freeze() to see if it reduces COW copies.
# (Hint: it helps more for lists/dicts than for numpy because numpy bypasses
# Python's per-element refcounts regardless.)
