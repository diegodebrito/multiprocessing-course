"""
Solutions — Module 06: NumPy Arrays as Views over Raw Buffers
"""
import numpy as np
import pandas as pd
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Exercise 1 — Strides and offset views
# ---------------------------------------------------------------------------

def worker_stride_check(shm_name: str, shape: tuple, dtype_str: str) -> None:
    shm = SharedMemory(name=shm_name, create=False)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)

    col5 = arr[:, 5]   # column 5
    row5 = arr[5, :]   # row 5

    print(f"  arr shape={arr.shape}, strides={arr.strides}")
    print(f"  arr[:,5]  strides={col5.strides}, C_CONTIGUOUS={col5.flags['C_CONTIGUOUS']}")
    print(f"  arr[5,:]  strides={row5.strides}, C_CONTIGUOUS={row5.flags['C_CONTIGUOUS']}")
    print("""
  Explanation:
    arr is (1000, 50) C-contiguous:
      strides = (50*8, 8) = (400, 8)  — 400 bytes per row, 8 bytes per element

    arr[:,5] = column 5: every element is 400 bytes apart (one row stride).
      strides = (400,)   → NOT C-contiguous (elements are not adjacent in memory)

    arr[5,:] = row 5: every element is 8 bytes apart (adjacent in memory).
      strides = (8,)     → C-contiguous (this is a row, already contiguous)
  """)
    shm.close()


def exercise1():
    print("Exercise 1 — Strides and offset views:")
    shape = (1000, 50)
    dtype = np.float64
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize

    shm = SharedMemory(create=True, size=size)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr[:] = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    p = Process(target=worker_stride_check, args=(shm.name, shape, dtype.str if hasattr(dtype, 'str') else np.dtype(dtype).str))
    p.start(); p.join()

    shm.close(); shm.unlink()


# ---------------------------------------------------------------------------
# Exercise 2 — ColDescriptor with round-trip
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColDescriptor:
    col_name: str
    shm_name: str
    dtype_str: str
    shape: tuple

    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * np.dtype(self.dtype_str).itemsize


def write_series_to_shm(series: pd.Series, col_name: str) -> tuple[SharedMemory, ColDescriptor]:
    raw = np.ascontiguousarray(series.to_numpy())
    shm = SharedMemory(create=True, size=raw.nbytes)
    view = np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm.buf)
    view[:] = raw
    return shm, ColDescriptor(
        col_name=col_name,
        shm_name=shm.name,
        dtype_str=raw.dtype.str,
        shape=raw.shape,
    )


def read_series_from_shm(desc: ColDescriptor) -> np.ndarray:
    shm = SharedMemory(name=desc.shm_name, create=False)
    arr = np.ndarray(desc.shape, dtype=np.dtype(desc.dtype_str), buffer=shm.buf)
    result = arr.copy()  # copy to local memory
    shm.close()
    return result


def exercise2():
    print("\nExercise 2 — ColDescriptor round-trip:")
    for dtype in [np.float64, np.float32, np.int32, np.bool_]:
        s = pd.Series(np.array([1, 2, 3, 4, 5], dtype=dtype))
        shm, desc = write_series_to_shm(s, col_name=f"col_{np.dtype(dtype).name}")
        recovered = read_series_from_shm(desc)
        match = np.array_equal(s.to_numpy(), recovered)
        print(f"  dtype={np.dtype(dtype).name:<8}: round-trip match={match}, "
              f"desc picklable={len(__import__('pickle').dumps(desc))} bytes")
        shm.close(); shm.unlink()


# ---------------------------------------------------------------------------
# Exercise 3 — Multiple dtypes in one segment
# ---------------------------------------------------------------------------

def worker_read_two_arrays(shm_name: str, n_float: int, n_int: int) -> tuple:
    shm = SharedMemory(name=shm_name, create=False)
    float_arr = np.ndarray((n_float,), dtype=np.float64, buffer=shm.buf, offset=0)
    int_arr = np.ndarray((n_int,), dtype=np.int32, buffer=shm.buf, offset=n_float * 8)
    result = float(float_arr.sum()), int(int_arr.sum())
    shm.close()
    return result


def exercise3():
    print("\nExercise 3 — Multiple dtypes in one segment:")
    N_F, N_I = 100_000, 50_000
    # Compute total size; int32 starts at offset N_F * 8 (already 8-byte aligned)
    total_size = N_F * 8 + N_I * 4

    shm = SharedMemory(create=True, size=total_size)
    float_arr = np.ndarray((N_F,), dtype=np.float64, buffer=shm.buf, offset=0)
    int_arr = np.ndarray((N_I,), dtype=np.int32, buffer=shm.buf, offset=N_F * 8)
    float_arr[:] = np.arange(N_F, dtype=np.float64)
    int_arr[:] = np.arange(N_I, dtype=np.int32)

    sum_f, sum_i = worker_read_two_arrays(shm.name, N_F, N_I)
    print(f"  float64 sum: {sum_f}  expected: {N_F*(N_F-1)/2}")
    print(f"  int32 sum:   {sum_i}  expected: {N_I*(N_I-1)/2}")
    print("  Separate segments are usually preferred: simpler bookkeeping,")
    print("  no manual offset calculations, easier to add/remove columns.")
    shm.close(); shm.unlink()


# ---------------------------------------------------------------------------
# Exercise 4 — Read-only views
# ---------------------------------------------------------------------------

def worker_try_write_readonly(shm_name: str, shape: tuple, dtype_str: str) -> str:
    shm = SharedMemory(name=shm_name, create=False)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    arr.flags["WRITEABLE"] = False  # mark read-only

    try:
        arr[0] = 999.0
        shm.close()
        return "ERROR: write succeeded — should not happen!"
    except ValueError as e:
        shm.close()
        return f"Write blocked as expected: {e}"


def exercise4():
    print("\nExercise 4 — Read-only views in workers:")
    N = 100
    shm = SharedMemory(create=True, size=N * 8)
    arr = np.ndarray((N,), dtype=np.float64, buffer=shm.buf)
    arr[:] = np.arange(N, dtype=np.float64)

    p = Process(target=lambda: print(f"  {worker_try_write_readonly(shm.name, (N,), '<f8')}"))
    # (lambda won't work with spawn, but fine for fork demo)
    # Use a proper module-level-style approach:
    result = worker_try_write_readonly(shm.name, (N,), arr.dtype.str)
    print(f"  {result}")

    print("""
  Why read-only views in workers:
    - Workers that SHOULD only read data get an immediate error if they
      accidentally write (programming mistake caught early).
    - Prevents silent data corruption of the shared segment.
    - Does NOT prevent other processes from writing through their own views.
    - Set with: arr.flags['WRITEABLE'] = False  (affects this view only)
  """)
    shm.close(); shm.unlink()


if __name__ == "__main__":
    exercise1()
    exercise2()
    exercise3()
    exercise4()
