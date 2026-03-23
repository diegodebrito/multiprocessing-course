"""
Module 05 — OS-Level Shared Memory: multiprocessing.shared_memory
==================================================================

Learning objectives
-------------------
1. Understand what an OS shared memory segment is.
2. Create, attach to, and destroy shared memory segments.
3. Understand the CRITICAL distinction between close() and unlink().
4. Understand lifetime rules: who should call close() and who calls unlink().
5. Know how to verify shared memory segments exist (/dev/shm on Linux).
6. Understand the platform gotchas (Windows vs Linux).

Run this module:
    python module_05_shared_memory.py
"""

import os
import sys
import time
import struct
import ctypes
import numpy as np
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory


# ---------------------------------------------------------------------------
# 1. WHAT IS A SHARED MEMORY SEGMENT?
# ---------------------------------------------------------------------------
#
# A shared memory segment is a region of RAM managed by the OS kernel that
# multiple processes can map into their own address spaces simultaneously.
#
# Key properties:
#   - It's identified by a NAME (a string), not a Python object reference.
#   - It lives in the kernel's memory, not in any process's heap.
#   - Multiple processes can attach to it by name → they all see the SAME bytes.
#   - When one process writes to it, all other attached processes see the change.
#   - It persists until explicitly unlinked, even if all processes detach.
#
# On Linux:
#   - Shared memory segments appear as files in /dev/shm/.
#   - You can list them: ls -la /dev/shm/
#   - Typical names: /psm_<random>, or custom names like /my_feature_matrix
#
# On macOS:
#   - Similar POSIX shared memory, but appears differently in the filesystem.
#
# On Windows:
#   - Named file mappings in the kernel's namespace.
#   - Some lifetime rules differ (see section 5).
#
# Python's multiprocessing.SharedMemory wraps the POSIX shm_open / mmap calls.


# ---------------------------------------------------------------------------
# 2. CREATING A SHARED MEMORY SEGMENT
# ---------------------------------------------------------------------------
#
# SharedMemory(create=True, size=N)
#   - Creates a new segment of N bytes.
#   - The OS assigns a name if you don't specify one; or you can pass name="myname".
#   - Returns a SharedMemory object. Key attributes:
#       .name   → the OS-level name (string, e.g. "psm_abc123")
#       .buf    → a memoryview of the raw bytes
#       .size   → number of bytes
#
# The name is what you pass to worker processes — they use it to attach.

def demo_create_shared_memory():
    """Create a shared memory segment and inspect it."""
    print("  Creating a shared memory segment (100 bytes):")
    shm = SharedMemory(create=True, size=100)
    print(f"    Name:  {shm.name}")
    print(f"    Size:  {shm.size} bytes")
    print(f"    buf:   {type(shm.buf)}")

    # Write some bytes into the buffer
    shm.buf[0:4] = b"ABCD"
    print(f"    First 4 bytes after write: {bytes(shm.buf[0:4])}")

    # On Linux, the segment appears in /dev/shm/
    if sys.platform == "linux":
        shm_path = f"/dev/shm/{shm.name}"
        exists = os.path.exists(shm_path)
        print(f"    /dev/shm/{shm.name} exists: {exists}")

    # Clean up properly (see section 4 for the full story)
    shm.close()
    shm.unlink()
    print(f"    After close()+unlink(): segment destroyed")


# ---------------------------------------------------------------------------
# 3. ATTACHING TO AN EXISTING SEGMENT
# ---------------------------------------------------------------------------
#
# SharedMemory(name="the_name", create=False)
#   - Attaches THIS PROCESS to an already-existing segment.
#   - Does NOT create a new one.
#   - Both processes now see the same bytes.
#
# This is the pattern for sharing data between parent and worker:
#   Parent: create → write data → pass name to worker
#   Worker: attach by name → read/write data → close()
#   Parent: wait for workers → close() + unlink()

def worker_read_shared(shm_name: str, size: int) -> None:
    """Worker that attaches to an existing shared memory segment and reads it."""
    shm = SharedMemory(name=shm_name, create=False)
    value = struct.unpack_from("d", shm.buf, offset=0)[0]  # read a float64 at offset 0
    print(f"  Worker (pid={os.getpid()}) read value from shared memory: {value}")
    shm.close()  # detach THIS PROCESS'S handle; does NOT destroy the segment


def worker_write_shared(shm_name: str, value: float) -> None:
    """Worker that attaches and writes to shared memory."""
    shm = SharedMemory(name=shm_name, create=False)
    struct.pack_into("d", shm.buf, 0, value)  # write a float64 at offset 0
    print(f"  Worker (pid={os.getpid()}) wrote {value} to shared memory")
    shm.close()


def demo_attach_and_share():
    """Parent creates shared memory, worker reads and writes, parent verifies."""
    print("  Parent creates, worker modifies, parent reads back:")
    SIZE = 64

    # Parent creates the segment
    shm = SharedMemory(create=True, size=SIZE)

    # Write initial value
    struct.pack_into("d", shm.buf, 0, 3.14159)
    print(f"  Parent wrote: 3.14159")

    # Spawn worker that reads
    p_read = Process(target=worker_read_shared, args=(shm.name, SIZE))
    p_read.start()
    p_read.join()

    # Spawn worker that writes a new value
    p_write = Process(target=worker_write_shared, args=(shm.name, 2.71828))
    p_write.start()
    p_write.join()

    # Parent reads back the modified value
    value = struct.unpack_from("d", shm.buf, offset=0)[0]
    print(f"  Parent reads after worker write: {value}")  # Should be 2.71828

    # Clean up
    shm.close()
    shm.unlink()


# ---------------------------------------------------------------------------
# 4. THE CRITICAL DISTINCTION: close() vs unlink()
# ---------------------------------------------------------------------------
#
# This is the most common source of bugs with shared memory. Understand it well.
#
#   shm.close()
#   ────────────
#   Detaches THIS PROCESS'S memory-mapped handle to the segment.
#   After close(), THIS process can no longer access shm.buf.
#   The segment itself STILL EXISTS — other processes with open handles
#   can still read/write it.
#   Analogy: like closing a file descriptor. The file still exists.
#
#   shm.unlink()
#   ────────────
#   Removes the OS-level name for the segment.
#   After unlink(), NEW processes cannot attach by name.
#   The actual memory is freed once ALL open handles are closed.
#   Analogy: like deleting a file. The data persists until all fd's are closed.
#
# Correct pattern:
#   - PARENT:  create → use → close() → unlink()    (owner: creates and destroys)
#   - WORKER:  attach → use → close()               (borrower: just detaches)
#
# WRONG patterns:
#   - Worker calls unlink(): the parent and other workers lose access.
#   - Nobody calls unlink(): segment leaks in /dev/shm — persists after all
#     processes exit! (On Linux, until reboot or manual cleanup.)
#   - Nobody calls close(): handle leak in that process.

def demo_close_vs_unlink():
    """Illustrate the difference between close() and unlink()."""
    print("  close() vs unlink() semantics:")

    shm = SharedMemory(create=True, size=10)
    name = shm.name
    shm.buf[0] = 42

    print(f"  Created segment '{name}'")

    # close() just detaches this process's handle
    shm.close()
    print(f"  After close(): this process can no longer access buf")

    # But another process (or this one re-attaching) can still see it!
    shm2 = SharedMemory(name=name, create=False)
    print(f"  Re-attached after close(): buf[0] = {shm2.buf[0]}  ← segment still exists")
    shm2.close()

    # unlink() removes the name — now nobody can attach by name
    shm3 = SharedMemory(name=name, create=False)  # attach before unlink
    shm3.unlink()  # remove the name
    shm3.close()   # close this handle too

    try:
        shm4 = SharedMemory(name=name, create=False)
        print("  Attached after unlink() — should NOT happen")
        shm4.close()
    except FileNotFoundError:
        print(f"  After unlink(): can't attach by name '{name}' (FileNotFoundError) ✓")


# ---------------------------------------------------------------------------
# 5. LIFETIME RULES AND WHO CALLS WHAT
# ---------------------------------------------------------------------------
#
# The lifetime of a shared memory segment is controlled by:
#   1. The NAME (unlink removes the name → no new attachments)
#   2. The reference count of open handles (last close() frees memory)
#
# Recommended ownership model:
#
#   class SharedDataContext:
#       def __enter__(self):
#           self.shm = SharedMemory(create=True, size=...)
#           return self
#
#       def __exit__(self, *_):
#           self.shm.close()    # detach parent's handle
#           self.shm.unlink()   # remove name so no new attaches
#
# Worker pattern:
#   def worker(shm_name):
#       shm = SharedMemory(name=shm_name, create=False)
#       try:
#           # use shm.buf
#           ...
#       finally:
#           shm.close()   # ALWAYS close in worker; NEVER unlink in worker
#
# On Windows:
#   - The OS holds the segment open as long as any handle is open.
#   - shm.unlink() in the parent while workers still have handles open → error.
#   - Fix: workers must close() their handles BEFORE the parent calls unlink().
#   - That's why in real production code, workers often copy data to a local
#     numpy array immediately and close the shm handle before doing any work.

def worker_linux_close_pattern(shm_name: str, n: int) -> float:
    """
    Best-practice worker: attach, copy needed data, close immediately.
    This pattern works on both Linux and Windows.
    """
    shm = SharedMemory(name=shm_name, create=False)
    # Copy data locally (so we can close the handle right away)
    arr_view = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
    local_copy = arr_view.copy()     # ← local numpy array, no longer tied to shm
    shm.close()                      # ← close handle immediately after copy

    # Do all processing on local_copy — no shm handle held
    result = float(local_copy.sum())
    return result


def demo_worker_copy_pattern():
    """Show the copy-then-close pattern in action."""
    print("  Worker copy-then-close pattern:")
    N = 1000
    shm = SharedMemory(create=True, size=N * 8)  # 8 bytes per float64
    arr = np.ndarray((N,), dtype=np.float64, buffer=shm.buf)
    arr[:] = np.arange(N, dtype=np.float64)

    p = Process(target=worker_linux_close_pattern, args=(shm.name, N))
    p.start()
    p.join()
    print(f"  Worker sum = {worker_linux_close_pattern(shm.name, N)}")  # verify locally
    shm.close()
    shm.unlink()


# ---------------------------------------------------------------------------
# 6. SHARED MEMORY SEGMENT SIZE AND ALIGNMENT
# ---------------------------------------------------------------------------
#
# The size must be calculated carefully:
#   - For a numpy array: shape product × dtype.itemsize
#   - Alignment: numpy expects data to be appropriately aligned.
#     For dtype=float64 (8 bytes), start offset should be 0 mod 8.
#     SharedMemory always starts at a page boundary, so offset 0 is fine.
#
# For multiple arrays in one segment:
#   - Lay them out sequentially with proper padding.
#   - Use struct.calcsize() or numpy strides to compute offsets.

def compute_shm_size_for_array(shape: tuple, dtype: np.dtype) -> int:
    """Calculate the exact number of bytes needed for a numpy array in shared memory."""
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


def demo_size_calculation():
    """Show how to correctly size a shared memory segment for numpy arrays."""
    shapes_and_dtypes = [
        ((1_000_000,), np.float64),
        ((500, 200), np.float32),
        ((100, 100, 100), np.int32),
    ]
    print("  Shared memory size calculations:")
    for shape, dtype in shapes_and_dtypes:
        size = compute_shm_size_for_array(shape, dtype)
        print(f"    shape={shape}, dtype={np.dtype(dtype)} → {size:,} bytes = {size/1e6:.2f} MB")


# ---------------------------------------------------------------------------
# 7. VERIFYING AND INSPECTING SHARED MEMORY (LINUX)
# ---------------------------------------------------------------------------

def demo_shm_inspection():
    """Show how to inspect shared memory segments on Linux."""
    if sys.platform != "linux":
        print("  (Linux only — skipping /dev/shm inspection)")
        return

    print("  Creating 3 segments and listing /dev/shm/:")
    segments = []
    for i in range(3):
        shm = SharedMemory(create=True, size=1024 * (i + 1))
        segments.append(shm)
        print(f"    Created: /dev/shm/{shm.name}  ({shm.size} bytes)")

    # List /dev/shm/ to see them
    shm_files = [f for f in os.listdir("/dev/shm") if f in {s.name for s in segments}]
    print(f"  Visible in /dev/shm/: {shm_files}")

    # Clean up
    for shm in segments:
        shm.close()
        shm.unlink()

    shm_files_after = [f for f in os.listdir("/dev/shm") if f in {s.name for s in segments}]
    print(f"  After unlink(): visible in /dev/shm/: {shm_files_after}  (empty ✓)")


# ---------------------------------------------------------------------------
# DEMO RUNNER
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODULE 05 — OS-Level Shared Memory")
    print("=" * 60)

    print("\n[1] Creating a shared memory segment")
    demo_create_shared_memory()

    print("\n[2] Attaching and sharing between processes")
    demo_attach_and_share()

    print("\n[3] close() vs unlink() — the critical distinction")
    demo_close_vs_unlink()

    print("\n[4] Worker copy-then-close best practice")
    demo_worker_copy_pattern()

    print("\n[5] Size calculations for numpy arrays")
    demo_size_calculation()

    print("\n[6] Inspecting /dev/shm on Linux")
    demo_shm_inspection()

    print("\n[7] Key takeaways")
    print("""
    MENTAL MODEL:
      shared memory segment = a named file in kernel memory
      close()   = close my file descriptor (file still exists)
      unlink()  = delete the file (file freed once all FDs closed)

    OWNERSHIP RULE:
      - CREATOR calls both close() AND unlink() (after all workers are done).
      - WORKERS call ONLY close() — NEVER unlink().

    COMMON BUGS:
      1. Worker calls unlink() → parent loses access → crash.
      2. Nobody calls unlink() → segment leaks in /dev/shm across reboots.
      3. Nobody calls close() → file descriptor leak in that process.
      4. (Windows) Parent calls unlink() while worker handles still open → error.
         Fix: workers copy data and close their handles before parent unlinks.

    SIZE CALCULATION:
      size = np.prod(shape) * np.dtype(dtype).itemsize
    """)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
#
# Exercise 1 — Leak detection
# ----------------------------
# Write a script that creates 5 shared memory segments but "forgets" to unlink
# one of them.  List /dev/shm before and after, and show which segment leaked.
# Then write a cleanup helper that unlinks all your named segments.
#
# Exercise 2 — Two-process round-trip via shared memory
# -------------------------------------------------------
# Parent writes an array [1, 2, 3, ..., 1000] into shared memory.
# Child reads it, computes the sum, and writes the result into a different
# offset in the same shared memory segment.
# Parent reads the result back.
# (Hint: use a multiprocessing.Event to signal when the child is done.)
#
# Exercise 3 — Multiple arrays in one segment
# ---------------------------------------------
# Store two numpy arrays in a single shared memory segment:
#   arr_a: shape (100_000,), dtype=float64 → starts at offset 0
#   arr_b: shape (50_000,), dtype=int32   → starts at offset len(arr_a) * 8
# In a worker, attach by name, reconstruct BOTH arrays using the known offsets,
# and return their sums.
# (This simulates the multi-column layout in a real shared DataFrame.)
#
# Exercise 4 — Lifetime quiz
# ---------------------------
# Without running code, trace through this pseudocode and identify the bugs:
#
#   shm = SharedMemory(create=True, size=1000)
#   def worker(name):
#       s = SharedMemory(name=name, create=False)
#       s.unlink()           # ← bug?
#       result = s.buf[0]
#       s.close()
#       return result
#   p = Process(target=worker, args=(shm.name,))
#   p.start(); p.join()
#   shm.unlink()             # ← bug?
#   shm.close()              # ← bug?
