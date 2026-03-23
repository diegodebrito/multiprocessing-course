"""
Module 06 — NumPy Arrays as Views over Raw Buffers
===================================================

Learning objectives
-------------------
1. Understand that a numpy array is just metadata + a pointer to a buffer.
2. Create numpy arrays backed by arbitrary raw memory (buffer= parameter).
3. Understand memory layout: C-contiguous vs Fortran-contiguous vs non-contiguous.
4. Know when np.ascontiguousarray() is needed and why.
5. Understand dtype.str and how to reconstruct an array from metadata alone.
6. Apply this to shared memory: create array views over shm.buf.

Run this module:
    python module_06_numpy_buffer_views.py
"""

import sys
import ctypes
import struct
import numpy as np
from multiprocessing.shared_memory import SharedMemory


# ---------------------------------------------------------------------------
# 1. A NUMPY ARRAY IS METADATA + BUFFER POINTER
# ---------------------------------------------------------------------------
#
# A numpy array object stores:
#   - A POINTER to the underlying data buffer (a C pointer to raw bytes)
#   - shape    : tuple of dimension sizes
#   - dtype    : data type (float64, int32, etc.)
#   - strides  : bytes to step in each dimension
#   - flags    : C_CONTIGUOUS, F_CONTIGUOUS, WRITEABLE, OWNDATA, etc.
#
# The OWNDATA flag tells you whether the array "owns" its buffer:
#   - np.array([1,2,3]) → OWNDATA=True  (numpy allocated the buffer)
#   - arr[::2]          → OWNDATA=False (view of a slice of arr's buffer)
#   - np.ndarray(..., buffer=some_buf) → OWNDATA=False (buffer owned externally)
#
# This design means creating a "view" is essentially free:
# you're just creating a new metadata object that points to existing bytes.

def inspect_array_metadata(arr: np.ndarray, label: str) -> None:
    """Print key metadata fields of a numpy array."""
    print(f"  {label}:")
    print(f"    shape={arr.shape}, dtype={arr.dtype}, strides={arr.strides}")
    print(f"    itemsize={arr.itemsize}, nbytes={arr.nbytes}")
    print(f"    OWNDATA={arr.flags['OWNDATA']}, C_CONTIGUOUS={arr.flags['C_CONTIGUOUS']}")
    print(f"    data ptr: {arr.ctypes.data:#x}")


def demo_array_metadata():
    """Inspect the metadata of various array types."""
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    inspect_array_metadata(base, "base array (owns data)")

    view = base[::2]  # every other element — non-contiguous view
    inspect_array_metadata(view, "view (stride=16, non-contiguous)")

    reshaped = base.reshape(2, 3)
    inspect_array_metadata(reshaped, "reshaped (2D view, C-contiguous)")


# ---------------------------------------------------------------------------
# 2. CREATING ARRAYS WITH EXTERNAL BUFFERS (buffer= parameter)
# ---------------------------------------------------------------------------
#
# np.ndarray(shape, dtype=dtype, buffer=buf)
#
# This creates an array that VIEWS the given buffer — no data is copied.
# The buffer must be a bytes-like object that supports the buffer protocol:
#   - bytes / bytearray
#   - mmap objects
#   - ctypes arrays
#   - memoryview (like shm.buf)
#
# CRITICAL: the buffer must have at least `shape * itemsize` bytes.
# CRITICAL: the array does NOT own the buffer — you must ensure the buffer
#           stays alive as long as the array is in use.

def demo_array_from_raw_bytes():
    """Create a numpy array from a raw bytearray — no data copy."""
    # Allocate 24 bytes (room for 3 float64 values)
    raw = bytearray(24)

    # Write 3 float64 values into the raw bytes
    struct.pack_into("3d", raw, 0, 1.1, 2.2, 3.3)

    # Create a numpy view — ZERO COPY
    arr = np.ndarray((3,), dtype=np.float64, buffer=raw)
    print(f"  Array from bytearray: {arr}")
    print(f"  OWNDATA={arr.flags['OWNDATA']}  (False → views the bytearray)")

    # Modify through the array — changes the underlying bytes!
    arr[1] = 99.9
    print(f"  After arr[1]=99.9, raw bytes: {struct.unpack_from('3d', raw, 0)}")


def demo_array_from_memoryview():
    """Create a numpy array from a memoryview (like shm.buf)."""
    data = bytearray(40)
    mv = memoryview(data)  # same type as shm.buf

    arr = np.ndarray((5,), dtype=np.float64, buffer=mv)
    arr[:] = [10.0, 20.0, 30.0, 40.0, 50.0]

    print(f"  Array from memoryview: {arr}")
    # Read back through the raw bytes to confirm it's the same memory
    readback = np.frombuffer(data, dtype=np.float64)
    print(f"  Read back via frombuffer: {readback}")
    print(f"  Same memory? data ptr equal: {arr.ctypes.data == readback.ctypes.data}")


# ---------------------------------------------------------------------------
# 3. NUMPY ARRAY OVER SHARED MEMORY — THE KEY PATTERN
# ---------------------------------------------------------------------------
#
# This is the core of the shared memory pattern:
#
#   PARENT:
#     shm = SharedMemory(create=True, size=N * dtype.itemsize)
#     arr = np.ndarray((N,), dtype=dtype, buffer=shm.buf)
#     arr[:] = source_data   ← writes into shared memory
#
#   WORKER (receives shm.name, N, dtype as plain picklable values):
#     shm = SharedMemory(name=shm_name, create=False)
#     arr = np.ndarray((N,), dtype=dtype, buffer=shm.buf)
#     # arr is a zero-copy view of the shared memory
#     result = arr.sum()     ← reads from shared memory
#     shm.close()

def demo_array_over_shared_memory():
    """Full cycle: write array to shm, read in 'another process' context."""
    N = 5
    dtype = np.float64

    # --- Simulate parent ---
    size = N * np.dtype(dtype).itemsize
    shm = SharedMemory(create=True, size=size)

    # Write via numpy view
    arr_write = np.ndarray((N,), dtype=dtype, buffer=shm.buf)
    arr_write[:] = [1.0, 4.0, 9.0, 16.0, 25.0]
    print(f"  Parent wrote: {arr_write}")

    # --- Simulate worker (same process for demo, but same API as subprocess) ---
    shm_worker = SharedMemory(name=shm.name, create=False)
    arr_read = np.ndarray((N,), dtype=dtype, buffer=shm_worker.buf)
    print(f"  Worker reads: {arr_read}")
    print(f"  Same memory? {arr_write.ctypes.data == arr_read.ctypes.data}")
    shm_worker.close()

    # --- Parent cleans up ---
    shm.close()
    shm.unlink()


# ---------------------------------------------------------------------------
# 4. MEMORY LAYOUT: C-CONTIGUOUS VS FORTRAN-CONTIGUOUS
# ---------------------------------------------------------------------------
#
# For shared memory to work correctly, the data must be laid out predictably.
# The two common layouts:
#
# C-contiguous (row-major): rows are contiguous in memory.
#   arr[i, j] is at offset (i * ncols + j) * itemsize
#   This is the numpy default (order='C').
#
# Fortran-contiguous (column-major): columns are contiguous.
#   arr[i, j] is at offset (i + j * nrows) * itemsize
#   This is what BLAS/LAPACK prefer; also how Fortran arrays work.
#
# For shared memory, use C-contiguous arrays (the default).
# If you have an array that came from somewhere else and might be
# F-contiguous or non-contiguous (e.g. a transposed array or a slice),
# use np.ascontiguousarray() BEFORE writing it into shared memory.

def demo_contiguity():
    """Show how transposing creates a non-contiguous array."""
    original = np.arange(12, dtype=np.float64).reshape(3, 4)
    print(f"  Original (3×4): C_CONTIGUOUS={original.flags['C_CONTIGUOUS']}")
    print(f"  Strides: {original.strides}  (4*8=32 bytes per row, 8 bytes per col)")

    transposed = original.T  # Returns a VIEW with swapped strides
    print(f"  Transposed (4×3): C_CONTIGUOUS={transposed.flags['C_CONTIGUOUS']}")
    print(f"  Strides: {transposed.strides}  (non-standard)")

    # Problem: if you try to write a transposed array into shared memory,
    # the bytes won't be in the right order for a reader who expects C layout.

    contiguous_transposed = np.ascontiguousarray(transposed)
    print(f"  np.ascontiguousarray(transposed): C_CONTIGUOUS={contiguous_transposed.flags['C_CONTIGUOUS']}")
    print(f"  Strides: {contiguous_transposed.strides}  (back to standard)")


def demo_write_contiguous_to_shm():
    """Show that writing a non-contiguous array to shm requires ascontiguousarray."""
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    transposed = arr.T  # non-contiguous

    # WRONG: writing non-contiguous data into shared memory
    # The bytes will be in wrong order for the reader
    shm = SharedMemory(create=True, size=transposed.nbytes)
    # This WORKS (no error), but the data layout is wrong for a reader
    # expecting a C-contiguous (3×4) array:
    view_wrong = np.ndarray(transposed.shape, dtype=transposed.dtype, buffer=shm.buf)
    view_wrong[:] = transposed  # numpy copies WITH correct layout, actually fine here
    # (numpy's assignment handles non-contiguous assignment correctly,
    # but when you later reconstruct with the same shape, it still works.)
    shm.close()
    shm.unlink()

    # CORRECT: explicitly make contiguous before writing
    cont = np.ascontiguousarray(transposed)
    shm2 = SharedMemory(create=True, size=cont.nbytes)
    view = np.ndarray(cont.shape, dtype=cont.dtype, buffer=shm2.buf)
    view[:] = cont
    print(f"  Wrote C-contiguous (4×3) array to shared memory: {bytes(shm2.buf[:8])[:4]}...")
    shm2.close()
    shm2.unlink()

    # Key point: always use np.ascontiguousarray() on the SOURCE before writing
    # into shared memory. This ensures the layout is predictable for readers.


# ---------------------------------------------------------------------------
# 5. DTYPE SERIALISATION AND ARRAY RECONSTRUCTION
# ---------------------------------------------------------------------------
#
# To reconstruct a numpy array from shared memory, a worker needs:
#   1. shm_name  → to attach to the segment
#   2. dtype     → to interpret the bytes correctly
#   3. shape     → to know the dimensions
#
# These three pieces of information are what you store in a "column descriptor"
# struct and pass to workers.
#
# dtype serialisation:
#   dtype.str   → platform-independent string like '<f8' (little-endian float64)
#   np.dtype(dtype_str) → reconstructs the dtype
#
# shape serialisation:
#   just a tuple of ints — picklable, tiny

def demo_dtype_serialisation():
    """Show dtype.str serialisation and reconstruction."""
    for dt in [np.float64, np.float32, np.int32, np.int64, np.bool_]:
        dtype = np.dtype(dt)
        dtype_str = dtype.str
        reconstructed = np.dtype(dtype_str)
        match = dtype == reconstructed
        print(f"  dtype={dtype.name:>8}, str={dtype_str!r:>6}  → "
              f"reconstructed={reconstructed.name:>8}, match={match}")


def demo_full_reconstruction():
    """Show how to reconstruct an array from just (name, dtype_str, shape)."""
    # Metadata that would be passed to a worker:
    shape = (3, 4)
    dtype_str = np.dtype(np.float64).str  # '<f8'
    N = int(np.prod(shape))

    # Parent: write data
    shm = SharedMemory(create=True, size=N * np.dtype(dtype_str).itemsize)
    write_arr = np.ndarray(shape, dtype=dtype_str, buffer=shm.buf)
    write_arr[:] = np.arange(12, dtype=np.float64).reshape(shape)
    shm_name = shm.name
    shm.close()  # detach but don't unlink

    # Worker: reconstruct from (shm_name, dtype_str, shape) — all picklable
    shm_w = SharedMemory(name=shm_name, create=False)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm_w.buf)
    print(f"  Worker reconstructed array:")
    print(f"  {arr}")
    print(f"  dtype={arr.dtype}, shape={arr.shape}, C_CONTIGUOUS={arr.flags['C_CONTIGUOUS']}")
    shm_w.close()

    # Cleanup
    shm_cleanup = SharedMemory(name=shm_name, create=False)
    shm_cleanup.close()
    shm_cleanup.unlink()


# ---------------------------------------------------------------------------
# 6. WRITING PANDAS SERIES INTO SHARED MEMORY
# ---------------------------------------------------------------------------
#
# A pandas Series backed by numpy:
#   series.to_numpy() → may return a view or a copy (see module 07)
#   np.ascontiguousarray(series.to_numpy()) → guaranteed C-contiguous copy
#
# Safe pattern for writing a Series into shared memory:
#   raw = np.ascontiguousarray(series.to_numpy(dtype=series.dtype))
#   size = raw.nbytes
#   shm = SharedMemory(create=True, size=size)
#   view = np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm.buf)
#   view[:] = raw   ← copies into shared memory
#   # Now shm.buf holds the data; raw can be garbage-collected

try:
    import pandas as pd

    def demo_series_to_shm():
        """Write a pandas Series to shared memory and read it back."""
        s = pd.Series(np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64))

        # Step 1: get a C-contiguous numpy array (may copy)
        raw = np.ascontiguousarray(s.to_numpy())

        # Step 2: create shared memory and write
        shm = SharedMemory(create=True, size=raw.nbytes)
        view = np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm.buf)
        view[:] = raw
        print(f"  Wrote Series to shm '{shm.name}': {view}")

        # Step 3: worker reads it back
        shm_r = SharedMemory(name=shm.name, create=False)
        arr = np.ndarray(raw.shape, dtype=raw.dtype, buffer=shm_r.buf)
        print(f"  Worker reads: {arr}")
        shm_r.close()

        shm.close()
        shm.unlink()

    _has_pandas = True
except ImportError:
    _has_pandas = False


# ---------------------------------------------------------------------------
# DEMO RUNNER
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODULE 06 — NumPy Arrays as Views over Raw Buffers")
    print("=" * 60)

    print("\n[1] Array metadata (shape, dtype, strides, OWNDATA)")
    demo_array_metadata()

    print("\n[2] Array from raw bytearray (buffer= parameter)")
    demo_array_from_raw_bytes()

    print("\n[3] Array from memoryview")
    demo_array_from_memoryview()

    print("\n[4] Array over shared memory")
    demo_array_over_shared_memory()

    print("\n[5] Memory layout: contiguity and ascontiguousarray")
    demo_contiguity()
    demo_write_contiguous_to_shm()

    print("\n[6] dtype serialisation")
    demo_dtype_serialisation()

    print("\n[7] Full reconstruction from (name, dtype_str, shape)")
    demo_full_reconstruction()

    if _has_pandas:
        print("\n[8] Series to shared memory")
        demo_series_to_shm()

    print("\n[9] Key takeaways")
    print("""
    MENTAL MODEL:
      numpy array = (pointer to buffer) + (shape, dtype, strides, flags)
      No buffer copy when creating a view (OWNDATA=False).

    SHARED MEMORY RECIPE:
      1. Compute size = np.prod(shape) * dtype.itemsize
      2. shm = SharedMemory(create=True, size=size)
      3. view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
      4. view[:] = source_data          ← writes data into shm
      5. Pass (shm.name, dtype.str, shape) to workers (all picklable!)
      6. Workers: attach by name, np.ndarray(shape, dtype, buffer=shm.buf)

    CONTIGUITY RULE:
      Always np.ascontiguousarray() BEFORE writing into shared memory.
      Slices and transposes may be non-contiguous; the reader won't know.

    DTYPE STRING:
      dtype.str (e.g. '<f8') is a compact, portable dtype serialiser.
      np.dtype('<f8') reconstructs it.
    """)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
#
# Exercise 1 — Strides and offset views
# ----------------------------------------
# Create a 2D shared memory segment that holds a (1000, 50) float64 matrix.
# In a worker, reconstruct the array and:
#   a) Access just column 5 using arr[:, 5] — is it contiguous?
#   b) Access just row 5 using arr[5, :] — is it contiguous?
# Explain the strides in each case.
#
# Exercise 2 — dtype roundtrip
# ------------------------------
# Write a helper class ColDescriptor with fields:
#   shm_name: str, dtype_str: str, shape: tuple, col_name: str
# Make it picklable (it should be automatically since it has no unpicklable fields).
# Write a function write_series_to_shm(series, col_name) → ColDescriptor.
# Write a function read_series_from_shm(desc: ColDescriptor) → np.ndarray.
# Test that the round-trip preserves values exactly.
#
# Exercise 3 — Multiple dtypes in one segment
# ---------------------------------------------
# Store a float64 array (N,) AND an int32 array (N,) in a single shared memory
# segment.  Compute offsets carefully (align int32 to a 4-byte boundary after
# the float64 section).  In a worker, reconstruct both arrays.
# Why might you prefer separate segments over one large segment?
#
# Exercise 4 — Read-only views
# -----------------------------
# Create a shared memory segment with some data.
# In a worker, create a numpy array view over it but mark it read-only:
#   arr.flags['WRITEABLE'] = False
# Try to modify arr[0] — what happens?
# Why might you want read-only views in workers that should only read data?
