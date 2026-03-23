# Python Multiprocessing & Shared Memory — Self-Contained Course

**Target audience:** Senior data scientists who write production Python and understand
pandas/NumPy, but haven't worked with `multiprocessing.shared_memory` or raw buffer
management.

**Environment:** Linux, Python 3.11+

---

## Course Structure

Work through the modules in order. Each module is a standalone Python file you can
run directly (`python module_XX_*.py`). Read the source carefully — the explanations
are in the code, not in separate docs.

| Module | File | Core Topic |
|--------|------|------------|
| 01 | `module_01_process_parallelism_and_gil.py` | The GIL, CPU-bound work, why processes |
| 02 | `module_02_start_methods_and_cow.py` | fork / spawn / forkserver + copy-on-write |
| 03 | `module_03_process_pool_executor.py` | ProcessPoolExecutor, Futures, as_completed |
| 04 | `module_04_pickling_and_boundaries.py` | Pickle, what crosses the process boundary |
| 05 | `module_05_shared_memory.py` | OS shared memory segments, close vs unlink |
| 06 | `module_06_numpy_buffer_views.py` | NumPy arrays as views over raw buffers |
| 07 | `module_07_pandas_numpy_memory.py` | Pandas ↔ NumPy: copies vs views |
| 08 | `module_08_memory_design_patterns.py` | ColDescriptor pattern, shared DataFrame |
| 09 | `module_09_pyarrow_datasets.py` | PyArrow datasets, column projection, conversion costs |
| 10 | `module_10_context_managers_resource_safety.py` | Context managers, resource cleanup, frozen config |
| 11 | `module_11_capstone.py` | Full pipeline: shared memory + ProcessPoolExecutor + PyArrow |

---

## Setup

```bash
pip install -r requirements.txt
```

Then run any module:

```bash
python module_01_process_parallelism_and_gil.py
```

---

## How to Use This Course

Each module follows the same structure:

1. **Concept sections** — richly commented code that explains the "why" alongside the "how"
2. **Runnable demos** — call `python module_XX.py` to see everything execute and print output
3. **Exercises** — at the bottom of each file, clearly marked. Try them before peeking at `solutions/`

The course is designed so that running each file top-to-bottom also runs all the demos,
so you see real output and can experiment by modifying the code.

---

## Key Mental Models to Build

By the end you should be able to answer:

- Why does a `ProcessPoolExecutor` worker need to re-attach to shared memory by *name* rather than
  receiving the object directly?
- What is the difference between `shm.close()` and `shm.unlink()`, and who calls each?
- When does `np.ndarray(shape, dtype, buffer=shm.buf)` copy data, and when is it a zero-copy view?
- Why is `np.ascontiguousarray()` necessary before writing into a shared buffer?
- When does `series.to_numpy()` return a view vs. a copy?
- What does copy-on-write mean for a forked worker process that reads a large in-memory DataFrame?
