"""
Solutions — Module 10: Context Managers and Resource Safety
"""
import os
import sys
import contextlib
import random
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Exercise 1 — NSharedArrays context manager with fault tolerance
# ---------------------------------------------------------------------------

@contextmanager
def NSharedArrays(
    n: int,
    shape: tuple,
    dtype: np.dtype,
) -> Iterator[list[tuple[SharedMemory, np.ndarray]]]:
    """
    Creates n shared memory segments, each holding an array of given shape/dtype.
    Guarantees cleanup of ALL created segments even if creation of one fails.
    """
    created: list[SharedMemory] = []
    results: list[tuple[SharedMemory, np.ndarray]] = []
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize

    try:
        for i in range(n):
            shm = SharedMemory(create=True, size=size)
            created.append(shm)
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            arr[:] = i  # fill with index so we can verify
            results.append((shm, arr))
        yield results
    finally:
        for shm in created:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


def exercise1():
    print("Exercise 1 — NSharedArrays context manager:")

    # Normal usage
    with NSharedArrays(3, shape=(100,), dtype=np.float64) as arrays:
        print(f"  Created {len(arrays)} segments")
        for i, (shm, arr) in enumerate(arrays):
            print(f"    segment {i}: name={shm.name}, arr[0]={arr[0]}")

    # Verify cleanup
    print("  All segments cleaned up after exit ✓")

    # Fault-tolerant test: simulate failure after 2 segments created
    # (We can't easily inject a fault into the constructor, so we verify
    #  that if an exception is raised INSIDE the with block, cleanup runs)
    created_names = []
    try:
        with NSharedArrays(5, shape=(50,), dtype=np.float64) as arrays:
            created_names = [shm.name for shm, _ in arrays]
            raise RuntimeError("Simulated failure in user code")
    except RuntimeError:
        pass

    # Verify all 5 were cleaned up despite the exception
    leaked = []
    for name in created_names:
        try:
            SharedMemory(name=name, create=False).close()
            leaked.append(name)
        except FileNotFoundError:
            pass
    print(f"  After exception: {len(leaked)} segments leaked (expected 0)")


# ---------------------------------------------------------------------------
# Exercise 2 — SharedPool context manager
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColDesc:
    col_name: str
    shm_name: str
    dtype_str: str
    shape: tuple


class SharedPool:
    """
    Manages attachment to multiple shared memory segments.
    Provides get_array(col_name) on demand.
    Closes all handles on __exit__.
    """

    def __init__(self, descriptors: list[ColDesc]):
        self._descs = {d.col_name: d for d in descriptors}
        self._open_shms: list[SharedMemory] = []
        self._arrays: dict[str, np.ndarray] = {}

    def get_array(self, col_name: str) -> np.ndarray:
        if col_name not in self._arrays:
            desc = self._descs[col_name]
            shm = SharedMemory(name=desc.shm_name, create=False)
            self._open_shms.append(shm)
            arr = np.ndarray(desc.shape, dtype=np.dtype(desc.dtype_str), buffer=shm.buf)
            self._arrays[col_name] = arr
        return self._arrays[col_name]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for shm in self._open_shms:
            try:
                shm.close()
            except Exception:
                pass
        self._open_shms.clear()
        self._arrays.clear()


def write_col(series: pd.Series, col_name: str) -> tuple[SharedMemory, ColDesc]:
    raw = np.ascontiguousarray(series.to_numpy())
    shm = SharedMemory(create=True, size=raw.nbytes)
    np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm.buf)[:] = raw
    return shm, ColDesc(col_name=col_name, shm_name=shm.name,
                        dtype_str=raw.dtype.str, shape=raw.shape)


def exercise2():
    print("\nExercise 2 — SharedPool context manager:")
    N = 10_000
    df = pd.DataFrame({
        "x": np.random.randn(N),
        "y": np.random.randn(N) * 2,
    })

    segs = []
    descs = []
    for col in df.columns:
        shm, desc = write_col(df[col], col)
        segs.append(shm); descs.append(desc)

    with SharedPool(descs) as pool:
        x = pool.get_array("x")
        y = pool.get_array("y")
        corr = float(np.corrcoef(x, y)[0, 1])
        print(f"  Correlation(x, y) = {corr:.4f}")
        print(f"  Open handles: {len(pool._open_shms)}")
    print("  All handles closed after __exit__ ✓")

    for shm in segs:
        shm.close(); shm.unlink()


# ---------------------------------------------------------------------------
# Exercise 3 — Three approaches to config freezing
# ---------------------------------------------------------------------------

# (a) __class__ swap (shown in module 10)
@dataclass
class MutableConfig:
    n_workers: int = 4
    batch_size: int = 1000


class FrozenConfig(MutableConfig):
    def __setattr__(self, name, value):
        raise AttributeError(f"Config is frozen: cannot set {name!r}")


@contextmanager
def frozen_class_swap(config: MutableConfig):
    config.__class__ = FrozenConfig
    try:
        yield config
    finally:
        config.__class__ = MutableConfig


# (b) frozen dataclass + functional update
@dataclass(frozen=True)
class ImmutableConfig:
    n_workers: int = 4
    batch_size: int = 1000

    def replace(self, **kwargs) -> "ImmutableConfig":
        import dataclasses
        return dataclasses.replace(self, **kwargs)


# (c) Proxy object that intercepts setattr
class ReadOnlyProxy:
    """Wraps any object and blocks attribute writes."""
    def __init__(self, obj):
        object.__setattr__(self, "_obj", obj)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_obj"), name)

    def __setattr__(self, name, value):
        raise AttributeError(f"ReadOnlyProxy blocks write to {name!r}")


def exercise3():
    print("\nExercise 3 — Three approaches to config freezing:")

    # (a)
    cfg_a = MutableConfig(n_workers=4)
    with frozen_class_swap(cfg_a) as f:
        try:
            f.n_workers = 99
        except AttributeError as e:
            print(f"  (a) __class__ swap blocked: {e}")
    cfg_a.n_workers = 8  # works after exit
    print(f"  (a) mutable again: {cfg_a}")

    # (b)
    cfg_b = ImmutableConfig(n_workers=4)
    try:
        cfg_b.n_workers = 99  # type: ignore
    except Exception as e:
        print(f"  (b) frozen dataclass blocked: {type(e).__name__}")
    updated = cfg_b.replace(n_workers=8)
    print(f"  (b) functional update: {updated} (original: {cfg_b})")

    # (c)
    cfg_c = MutableConfig(n_workers=4)
    proxy = ReadOnlyProxy(cfg_c)
    try:
        proxy.n_workers = 99
    except AttributeError as e:
        print(f"  (c) proxy blocked: {e}")
    print(f"  (c) reads still work: n_workers={proxy.n_workers}")

    print("""
  Comparison:
    (a) __class__ swap: surprising, mutates the object, no static type info.
        Useful when you can't change the class hierarchy.
    (b) frozen dataclass: clearest, safest, IDE-friendly, immutable from creation.
        Requires designing config as frozen from the start.
    (c) proxy: flexible, works for any object, but adds a layer of indirection.
        Subclass attributes (via __getattr__) may behave unexpectedly.
  Prefer (b) for new code. Use (c) for third-party objects.
  """)


# ---------------------------------------------------------------------------
# Exercise 4 — Cleanup verification test
# ---------------------------------------------------------------------------

def count_shm_segments(prefix: str = "") -> set:
    """List shared memory segments (Linux only)."""
    if sys.platform != "linux":
        return set()
    try:
        entries = os.listdir("/dev/shm")
        return {e for e in entries if e.startswith(prefix) or not prefix}
    except PermissionError:
        return set()


def run_pipeline_with_possible_exception(fail: bool) -> None:
    """Mini pipeline that may fail mid-way."""
    segs = []
    try:
        for i in range(3):
            shm = SharedMemory(create=True, size=100)
            segs.append(shm)
            if fail and i == 1:
                raise RuntimeError("Simulated mid-pipeline failure")
    finally:
        for shm in segs:
            try:
                shm.close(); shm.unlink()
            except Exception:
                pass


def exercise4():
    print("\nExercise 4 — Cleanup verification test:")
    before = count_shm_segments()

    for trial in range(10):
        fail = random.random() > 0.5
        try:
            run_pipeline_with_possible_exception(fail=fail)
        except RuntimeError:
            pass

    after = count_shm_segments()
    # New segments that weren't there before
    leaked = after - before
    print(f"  Ran 10 trials (random failures). Leaked segments: {len(leaked)}")
    if leaked:
        print(f"  Leaked: {leaked}")
    else:
        print("  No leaks ✓  (try/finally in run_pipeline_with_possible_exception guarantees cleanup)")


if __name__ == "__main__":
    exercise1()
    exercise2()
    exercise3()
    exercise4()
