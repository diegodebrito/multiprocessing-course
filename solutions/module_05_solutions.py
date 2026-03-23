"""
Solutions — Module 05: OS-Level Shared Memory
"""
import os
import sys
import struct
import multiprocessing
from multiprocessing import Process, Event
from multiprocessing.shared_memory import SharedMemory
import numpy as np


# ---------------------------------------------------------------------------
# Exercise 1 — Leak detection
# ---------------------------------------------------------------------------

def demo_leak_and_cleanup():
    """Create 5 segments, "forget" one, then detect and clean up."""
    created = []
    for i in range(5):
        shm = SharedMemory(create=True, size=100)
        created.append(shm)

    # "Forget" to unlink segment 2 (simulate a bug)
    leaked_name = created[2].name
    for i, shm in enumerate(created):
        shm.close()
        if i != 2:
            shm.unlink()

    print(f"Leaked segment: {leaked_name}")
    if sys.platform == "linux":
        exists = os.path.exists(f"/dev/shm/{leaked_name}")
        print(f"  /dev/shm/{leaked_name} exists: {exists}")

    # Cleanup helper
    def cleanup_segment(name: str) -> None:
        try:
            shm = SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
            print(f"  Cleaned up leaked segment: {name}")
        except FileNotFoundError:
            print(f"  Segment {name} already gone")

    cleanup_segment(leaked_name)


# ---------------------------------------------------------------------------
# Exercise 2 — Two-process round-trip via shared memory
# ---------------------------------------------------------------------------

ARRAY_SIZE = 1000
RESULT_OFFSET = ARRAY_SIZE * 8  # result float64 stored after the input array
SHM_SIZE = ARRAY_SIZE * 8 + 8   # input array + one float64 for result


def child_sum_and_write(shm_name: str, ready_event: multiprocessing.Event) -> None:
    shm = SharedMemory(name=shm_name, create=False)
    arr = np.ndarray((ARRAY_SIZE,), dtype=np.float64, buffer=shm.buf)
    total = float(arr.sum())
    # Write result at the offset after the input array
    struct.pack_into("d", shm.buf, RESULT_OFFSET, total)
    shm.close()
    ready_event.set()  # signal the parent


def exercise2():
    print("\nTwo-process round-trip via shared memory:")
    shm = SharedMemory(create=True, size=SHM_SIZE)
    arr = np.ndarray((ARRAY_SIZE,), dtype=np.float64, buffer=shm.buf)
    arr[:] = np.arange(1, ARRAY_SIZE + 1, dtype=np.float64)

    ready = multiprocessing.Event()
    p = Process(target=child_sum_and_write, args=(shm.name, ready))
    p.start()
    ready.wait()  # block until child signals
    p.join()

    result = struct.unpack_from("d", shm.buf, RESULT_OFFSET)[0]
    expected = np.arange(1, ARRAY_SIZE + 1).sum()
    print(f"  Result: {result}  (expected {expected})")
    shm.close()
    shm.unlink()


# ---------------------------------------------------------------------------
# Exercise 3 — Multiple arrays in one segment
# ---------------------------------------------------------------------------

def worker_two_arrays(shm_name: str, n_float: int, n_int: int) -> tuple[float, int]:
    shm = SharedMemory(name=shm_name, create=False)
    # float64 array starts at offset 0
    arr_f = np.ndarray((n_float,), dtype=np.float64, buffer=shm.buf, offset=0)
    # int32 array starts right after (padded to 4-byte boundary automatically)
    offset_int = n_float * 8
    arr_i = np.ndarray((n_int,), dtype=np.int32, buffer=shm.buf, offset=offset_int)
    result = float(arr_f.sum()), int(arr_i.sum())
    shm.close()
    return result


def exercise3():
    print("\nMultiple arrays in one segment:")
    N_F = 100_000   # float64
    N_I = 50_000    # int32
    size = N_F * 8 + N_I * 4  # total bytes

    shm = SharedMemory(create=True, size=size)
    arr_f = np.ndarray((N_F,), dtype=np.float64, buffer=shm.buf, offset=0)
    arr_i = np.ndarray((N_I,), dtype=np.int32, buffer=shm.buf, offset=N_F * 8)
    arr_f[:] = np.arange(N_F, dtype=np.float64)
    arr_i[:] = np.arange(N_I, dtype=np.int32)

    p = Process(target=worker_two_arrays, args=(shm.name, N_F, N_I))
    p.start()
    p.join()

    sum_f, sum_i = worker_two_arrays(shm.name, N_F, N_I)
    expected_f = N_F * (N_F - 1) / 2
    expected_i = N_I * (N_I - 1) / 2
    print(f"  float64 sum: {sum_f} (expected {expected_f})")
    print(f"  int32 sum:   {sum_i} (expected {expected_i})")
    print("  Separate segments are simpler and avoid manual offset management,")
    print("  but one large segment has slightly lower OS overhead.")
    shm.close()
    shm.unlink()


# ---------------------------------------------------------------------------
# Exercise 4 — Lifetime quiz answers
# ---------------------------------------------------------------------------

def exercise4_answers():
    print("\nLifetime quiz — bugs identified:")
    print("""
  1. Worker calls s.unlink() BEFORE the parent reads or other workers use it.
     → Bug: parent will get FileNotFoundError on next access via name.
     → Fix: NEVER unlink in a worker.

  2. Worker calls s.close() AFTER unlink() — this is fine (close detaches the handle).
     But the order matters: unlink removes the name; close() releases the handle.
     The actual segment memory is freed only after ALL handles are closed.

  3. After join(), shm.unlink() — too late if the worker already called unlink().
     If worker did unlink (bug #1), this will raise FileNotFoundError.
     → Fix: only the parent (creator) calls unlink, once.

  4. shm.close() is called AFTER unlink() in the parent — correct order is:
     shm.close() first, then shm.unlink().
     Actually either order works for the parent's handle, but close-then-unlink
     is the conventional pattern.

  Corrected pattern:
    # worker:
    s = SharedMemory(name=name, create=False)
    result = s.buf[0]
    s.close()     # ← only close, never unlink
    return result

    # parent (after join):
    shm.close()   # detach parent's handle
    shm.unlink()  # remove the name (segment freed once all handles close)
    """)


if __name__ == "__main__":
    demo_leak_and_cleanup()
    exercise2()
    exercise3()
    exercise4_answers()
