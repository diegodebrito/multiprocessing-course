"""
Module 10 — Context Managers and Resource Safety
=================================================

Learning objectives
-------------------
1. Understand __enter__ / __exit__ and the @contextmanager decorator.
2. Write context managers that guarantee cleanup even on exceptions.
3. Apply context managers to shared memory lifecycle management.
4. Understand the "frozen config" pattern (temporarily changing __class__).
5. Know how to handle cleanup in parent vs worker processes.

Run this module:
    python module_10_context_managers_resource_safety.py
"""

import os
import sys
import time
import contextlib
from contextlib import contextmanager
from typing import Iterator, Optional, Any
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. CONTEXT MANAGERS: THE BASICS
# ---------------------------------------------------------------------------
#
# A context manager controls entry and exit from a `with` block.
# It guarantees that __exit__ is called no matter how the block exits:
#   - Normal completion
#   - Exception raised inside the block
#   - return statement inside the block
#   - break / continue
#
# Two ways to write context managers:
#   a) Class-based: implement __enter__ and __exit__
#   b) Generator-based: use @contextmanager decorator

# --- Class-based ---

class Timer:
    """Simple context manager that times a block of code."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self  # returned as `as` target

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self._start
        # Return False (or None) to let exceptions propagate.
        # Return True to SUPPRESS exceptions.
        return False  # don't suppress exceptions

    def __repr__(self):
        return f"Timer(elapsed={self.elapsed:.4f}s)"


# --- Generator-based ---

@contextmanager
def timed_block(label: str) -> Iterator[None]:
    """Contextmanager version of Timer."""
    t0 = time.perf_counter()
    try:
        yield  # body of the `with` block runs here
    finally:
        elapsed = time.perf_counter() - t0
        print(f"  [{label}] elapsed: {elapsed*1000:.2f}ms")


def demo_basic_context_managers():
    """Show both styles of context managers."""
    # Class-based
    with Timer() as t:
        _ = sum(range(1_000_000))
    print(f"  Timer: {t}")

    # Generator-based
    with timed_block("sum 1M"):
        _ = sum(range(1_000_000))


# ---------------------------------------------------------------------------
# 2. EXCEPTION SAFETY — WHY try/finally MATTERS
# ---------------------------------------------------------------------------
#
# Without a context manager, cleanup is easy to miss:
#
#   shm = SharedMemory(create=True, size=1000)
#   arr = np.ndarray((125,), dtype=np.float64, buffer=shm.buf)
#   result = process(arr)   ← if this raises, shm is NEVER cleaned up!
#   shm.close()
#   shm.unlink()
#
# With a context manager (or try/finally), cleanup always runs:
#
#   shm = SharedMemory(create=True, size=1000)
#   try:
#       arr = np.ndarray((125,), dtype=np.float64, buffer=shm.buf)
#       result = process(arr)
#   finally:
#       shm.close()     ← ALWAYS runs
#       shm.unlink()    ← ALWAYS runs
#
# Better: wrap in a context manager.

@contextmanager
def managed_shared_memory(size: int, name: Optional[str] = None) -> Iterator[SharedMemory]:
    """
    Context manager that creates a SharedMemory segment and guarantees cleanup.

    Usage:
        with managed_shared_memory(1000) as shm:
            arr = np.ndarray((125,), dtype=np.float64, buffer=shm.buf)
            # use arr
        # shm is closed and unlinked here, even if an exception occurred
    """
    shm = SharedMemory(create=True, size=size, name=name)
    try:
        yield shm
    finally:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass


def demo_exception_safety():
    """Show that cleanup runs even when an exception is raised."""
    print("  Exception safety with context manager:")

    try:
        with managed_shared_memory(1024) as shm:
            name = shm.name
            print(f"    Segment created: {name}")
            raise RuntimeError("Simulated failure during processing!")
    except RuntimeError as e:
        print(f"    Exception caught: {e}")

    # Verify that the segment was cleaned up
    try:
        SharedMemory(name=name, create=False)
        print("    BUG: segment still exists after exception!")
    except FileNotFoundError:
        print(f"    Segment '{name}' was cleaned up despite exception ✓")


# ---------------------------------------------------------------------------
# 3. NESTED CONTEXT MANAGERS
# ---------------------------------------------------------------------------
#
# When managing multiple resources, nest context managers or use contextlib.ExitStack.
# ExitStack is especially useful when you don't know the number of resources
# at compile time (e.g., one segment per column in a DataFrame).

def demo_exit_stack():
    """Use ExitStack to manage a dynamic number of shared memory segments."""
    columns = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4.0, 5.0, 6.0])}

    print("  ExitStack for dynamic resource management:")
    segments = {}

    with contextlib.ExitStack() as stack:
        for col_name, data in columns.items():
            shm = stack.enter_context(managed_shared_memory(data.nbytes))
            view = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
            view[:] = data
            segments[col_name] = (shm.name, data.shape, data.dtype.str)
            print(f"    Segment for '{col_name}': {shm.name}")

        print(f"    Inside stack: {len(segments)} segments active")
        # All segments are live here

    # After the with block: all segments are automatically closed and unlinked
    print("  After ExitStack exit: all segments cleaned up ✓")
    for col_name, (name, shape, dtype_str) in segments.items():
        try:
            SharedMemory(name=name, create=False)
            print(f"  BUG: {col_name} segment still exists!")
        except FileNotFoundError:
            pass
    print("  Confirmed: all segments gone ✓")


# ---------------------------------------------------------------------------
# 4. CONTEXT MANAGER FOR WORKER ATTACHMENT
# ---------------------------------------------------------------------------
#
# Workers need a slightly different context manager: they ATTACH (don't create)
# and they CLOSE (don't unlink).

@contextmanager
def attach_shared_memory(name: str) -> Iterator[SharedMemory]:
    """
    Context manager for workers: attach to existing segment, close on exit.
    NEVER unlinks — the parent owns the lifecycle.
    """
    shm = SharedMemory(name=name, create=False)
    try:
        yield shm
    finally:
        shm.close()  # detach only; no unlink


def demo_worker_attachment():
    """Show the attach pattern for workers."""
    # Create a segment in the parent
    with managed_shared_memory(64) as shm:
        view = np.ndarray((8,), dtype=np.float64, buffer=shm.buf)
        view[:] = [1, 2, 3, 4, 5, 6, 7, 8]
        name = shm.name

        # Simulate a worker context
        with attach_shared_memory(name) as worker_shm:
            arr = np.ndarray((8,), dtype=np.float64, buffer=worker_shm.buf)
            total = arr.sum()
            print(f"  Worker read: sum={total} (expected 36.0)")
        # worker_shm.close() was called, but name is still valid

        # Parent's `with` block keeps the segment alive
        still_live = np.ndarray((8,), dtype=np.float64, buffer=shm.buf)
        print(f"  Parent still reads: {still_live[:3]}")

    # Here: parent's with block exits → shm.close() + shm.unlink()
    print("  Segment cleaned up by parent context manager ✓")


# ---------------------------------------------------------------------------
# 5. THE FROZEN CONFIG PATTERN
# ---------------------------------------------------------------------------
#
# Problem: You have a mutable configuration object that workers read from.
# You want to prevent accidental mutation DURING parallel execution.
#
# One unusual pattern (seen in production code): temporarily change the
# object's __class__ to a read-only version.
#
# This is unusual but occasionally useful when:
#   - You can't change the class hierarchy.
#   - The read-only version raises on setattr.
#   - You want to "freeze" the object for the duration of a block.
#
# More common and idiomatic alternatives:
#   - dataclasses.dataclass(frozen=True) — immutable dataclass
#   - Return a copy with copy.deepcopy()
#   - Use a NamedTuple
#   - Use __slots__ and remove the mutation methods

@dataclass
class PipelineConfig:
    """Mutable configuration for a feature pipeline."""
    n_workers: int = 4
    batch_size: int = 1000
    output_dir: str = "/tmp/output"

    def update_output_dir(self, new_dir: str) -> None:
        self.output_dir = new_dir


class FrozenPipelineConfig(PipelineConfig):
    """
    Read-only version of PipelineConfig.
    Raises AttributeError on any setattr, preventing mutation.
    """
    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"PipelineConfig is frozen during parallel execution. "
            f"Cannot set '{name}' = {value!r}"
        )


@contextmanager
def frozen_config(config: PipelineConfig) -> Iterator[PipelineConfig]:
    """
    Temporarily freeze a PipelineConfig by changing its __class__.
    This is the unusual __class__ swap pattern.

    How it works:
      - config.__class__ = FrozenPipelineConfig  → activates frozen behaviour
      - code inside the with block can't mutate config
      - config.__class__ = PipelineConfig  → restores mutability
    """
    original_class = config.__class__
    config.__class__ = FrozenPipelineConfig
    try:
        yield config
    finally:
        config.__class__ = original_class


def demo_frozen_config():
    """Show the frozen config pattern."""
    config = PipelineConfig(n_workers=4, batch_size=500)
    print(f"  Before freeze: {config}")

    with frozen_config(config) as frozen:
        print(f"  Inside frozen block: {frozen}")
        print(f"  Read n_workers: {frozen.n_workers}  ← reads work fine")

        try:
            frozen.n_workers = 99  # should raise
        except AttributeError as e:
            print(f"  Mutation blocked: {e}")

    # After the context manager, config is mutable again
    config.n_workers = 8
    print(f"  After freeze: {config}  (mutation works again)")


# ---------------------------------------------------------------------------
# 6. PREFERRED ALTERNATIVES TO __CLASS__ SWAP
# ---------------------------------------------------------------------------
#
# The __class__ swap is unusual and surprising to readers.
# Here are cleaner alternatives:

from dataclasses import dataclass as _dc

@_dc(frozen=True)
class ImmutableConfig:
    """Use frozen=True for a truly immutable dataclass."""
    n_workers: int = 4
    batch_size: int = 1000
    output_dir: str = "/tmp/output"

    def with_output_dir(self, new_dir: str) -> "ImmutableConfig":
        """Return a new config with the updated field (functional update)."""
        return ImmutableConfig(
            n_workers=self.n_workers,
            batch_size=self.batch_size,
            output_dir=new_dir,
        )


def demo_immutable_config():
    """Show frozen dataclass as preferred alternative."""
    config = ImmutableConfig(n_workers=4, batch_size=500)
    print(f"  Frozen dataclass: {config}")

    # Functional update (returns new object, original unchanged)
    new_config = config.with_output_dir("/data/output")
    print(f"  Updated: {new_config}")
    print(f"  Original unchanged: {config.output_dir}")

    # Mutation is blocked at class level (no __class__ swap needed)
    try:
        config.n_workers = 8  # type: ignore
    except Exception as e:
        print(f"  Mutation blocked by frozen dataclass: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 7. BUILDING A COMPLETE RESOURCE-SAFE PIPELINE
# ---------------------------------------------------------------------------
#
# Putting it all together: a pipeline that:
#   - Manages shared memory for multiple columns
#   - Guarantees cleanup on exception
#   - Freezes config during parallel execution
#   - Uses attach_shared_memory in workers (no unlink)

from module_08_memory_design_patterns import ColDescriptor, write_series_to_shm


def worker_safe_compute(desc: ColDescriptor, config_dict: dict) -> dict:
    """
    Worker that uses the safe attach pattern.
    Receives config as a plain dict (fully picklable).
    """
    with attach_shared_memory(desc.shm_name) as shm:
        arr = np.ndarray(desc.shape, dtype=np.dtype(desc.dtype_str), buffer=shm.buf)
        # arr is read-only for safety
        result = {
            "col": desc.col_name,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n_workers": config_dict["n_workers"],
        }
    # shm.close() was called by attach_shared_memory.__exit__
    return result


def demo_complete_safe_pipeline():
    """End-to-end resource-safe pipeline."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    config = PipelineConfig(n_workers=2, batch_size=1000)
    N = 50_000
    df = pd.DataFrame({
        "feature_x": np.random.randn(N),
        "feature_y": np.random.randn(N),
        "feature_z": np.random.randn(N),
    })

    print("  Complete safe pipeline:")
    with contextlib.ExitStack() as stack:
        descs = []
        for col in df.columns:
            shm, desc = write_series_to_shm(df[col], col_name=col)
            stack.callback(shm.close)
            stack.callback(shm.unlink)
            descs.append(desc)

        with frozen_config(config) as frozen:
            config_dict = {"n_workers": frozen.n_workers, "batch_size": frozen.batch_size}

            with ProcessPoolExecutor(max_workers=frozen.n_workers) as executor:
                futures = {executor.submit(worker_safe_compute, d, config_dict): d.col_name
                           for d in descs}
                for future in as_completed(futures):
                    r = future.result()
                    print(f"    {r['col']}: mean={r['mean']:+.4f}, std={r['std']:.4f}")

    print("  All resources cleaned up by ExitStack ✓")


# ---------------------------------------------------------------------------
# DEMO RUNNER
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODULE 10 — Context Managers and Resource Safety")
    print("=" * 60)

    print("\n[1] Basic context managers (class vs @contextmanager)")
    demo_basic_context_managers()

    print("\n[2] Exception safety")
    demo_exception_safety()

    print("\n[3] ExitStack for dynamic resource management")
    demo_exit_stack()

    print("\n[4] Worker attachment context manager")
    demo_worker_attachment()

    print("\n[5] The frozen config pattern (__class__ swap)")
    demo_frozen_config()

    print("\n[6] Preferred alternative: frozen dataclass")
    demo_immutable_config()

    print("\n[7] Complete resource-safe pipeline")
    demo_complete_safe_pipeline()

    print("\n[8] Key takeaways")
    print("""
    CONTEXT MANAGER RULES FOR SHARED MEMORY:
      - Parent (creator): use context manager that calls close() + unlink() on exit.
      - Worker (reader):  use context manager that calls ONLY close() on exit.
      - NEVER unlink() in a worker.

    EXITSTACK:
      - Use contextlib.ExitStack when you don't know the number of resources
        at compile time (e.g., one segment per column).
      - stack.enter_context(cm) registers a context manager.
      - stack.callback(fn) registers an arbitrary cleanup function.

    FROZEN CONFIG (__class__ swap):
      - Unusual pattern: temporarily changes an object's class to prevent mutation.
      - Prefer: frozen dataclasses, NamedTuples, or separate config objects.
      - Understand it so you can read it in existing code, but don't write it new.

    try/finally vs context manager:
      - Context managers are try/finally, but reusable and composable.
      - For shared memory: ALWAYS use a context manager (don't rely on GC).
      - Python doesn't guarantee __del__ is called (especially in CPython
        with cycles, or in PyPy/Jython). Don't rely on __del__ for cleanup.
    """)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
#
# Exercise 1 — Nested context manager with cleanup verification
# --------------------------------------------------------------
# Write a context manager NSharedArrays(n, shape, dtype) that:
#   - Creates n shared memory segments, each holding an array of given shape/dtype
#   - Yields a list of (shm, view) tuples
#   - On exit: closes and unlinks ALL segments, even if one fails
# Test that if an exception is raised in the middle of creation (simulate with
# a mock that fails on the 3rd segment), the first 2 segments are still cleaned up.
#
# Exercise 2 — Reentrant resource management
# --------------------------------------------
# Write a SharedPool context manager that:
#   - Accepts a list of ColDescriptors
#   - Provides a method get_array(col_name) → np.ndarray (attaches on demand)
#   - On __exit__, closes all open handles
# Use it in a worker that reads two columns and computes their correlation.
#
# Exercise 3 — Frozen config comparison
# ----------------------------------------
# Implement the same config-freezing behaviour using three different approaches:
#   a) __class__ swap (as shown)
#   b) frozen dataclass + functional update
#   c) A proxy object that intercepts setattr
# Compare: which is most readable? Which is safest? Which handles subclasses?
#
# Exercise 4 — Cleanup verification test
# ----------------------------------------
# Write a test that:
#   1. Counts /dev/shm entries before the pipeline runs.
#   2. Runs the demo_complete_safe_pipeline (even with simulated exceptions).
#   3. Counts /dev/shm entries after.
#   4. Asserts the counts are equal (no leaked segments).
# Run this in a loop 10 times with random exception injection to verify robustness.
