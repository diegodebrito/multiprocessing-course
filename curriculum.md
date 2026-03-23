# Python Multiprocessing & Shared Memory Curriculum

To understand this module “really well”, you mainly need a solid mental model of (1) Python process-based parallelism and serialization rules, and (2) how OS shared memory works and how NumPy/Pandas sit on top of raw buffers.

Here are the most relevant topics to study, in roughly the order that will make the code click.

-----

## 1) Python multiprocessing fundamentals (processes, not threads)

- **Why processes bypass the GIL** and when that helps.
- **Process lifecycle** and overhead: starting processes, memory costs, IPC costs.
- **Start methods**: `fork` vs `spawn` vs `forkserver`
- Linux commonly uses **fork** (copy-on-write semantics).
- Windows/macOS default to **spawn** (child starts fresh; everything passed must be pickled).
- Key doc topics: `multiprocessing` module overview; start methods.

## 2) `concurrent.futures.ProcessPoolExecutor` specifically

- How `ProcessPoolExecutor` works: task submission, worker processes, and result collection.
- `Future` objects, `as_completed`, exception propagation (`future.result()` re-raises).
- The crucial constraint: **arguments and return values must be picklable** (because they are sent to subprocesses).

## 3) Pickling / serialization and “what crosses the process boundary”

This module is optimized around minimizing what gets pickled per task. Study:

- Python **pickle** basics (what is picklable, what isn’t, performance implications).
- Why module-level functions are used for workers (`_process_one_feature` must be picklable).
- Why passing a whole `DataFrame` to each worker is expensive (copies/serialization).

## 4) Shared memory at the OS level (`multiprocessing.shared_memory`)

This is the core of the “confusing” section. You want to understand:

- What a **shared memory segment** is (named OS-managed block of bytes).
- `SharedMemory(create=True, size=...)` vs `SharedMemory(name=..., create=False)`.
- **Lifetime rules**:
- `close()` = detach this process’s handle
- `unlink()` = remove the segment name (actual destruction happens once all handles close)
- Platform gotcha reflected in the code:
- On **Windows**, if workers keep handles open, `unlink()` can fail. That’s why workers copy then `close()` immediately.

## 5) NumPy arrays as views over raw buffers

The shared memory blocks become usable via NumPy:

- `np.ndarray(shape, dtype, buffer=shm.buf)` creates an array **view** over existing bytes.
- Contiguity: `np.ascontiguousarray(...)` ensures predictable layout before writing to shared memory.
- `dtype.str`, `shape`, and reconstructing arrays from metadata.

This explains why `ColDescriptor` stores:

- **shm_name** (how to attach)
- **dtype** and **shape** (how to interpret bytes)
- **col_name** (how to rebuild the DataFrame)

## 6) Pandas <-> NumPy memory behavior

Because the code moves between pandas Series/DataFrames and NumPy buffers:

- `Series.to_numpy()` vs `np.asarray(series)` and when copies happen.
- `pd.concat(..., copy=False)` and what that does/doesn’t guarantee.
- Index alignment issues (why they return a DataFrame with a default RangeIndex and ensure concatenation alignment).

## 7) Copy-on-write and memory usage modeling (why this design exists)

To appreciate the design, understand:

- **Copy-on-write with fork**: read-only sharing until mutation.
- With **spawn**, everything must be explicitly transferred—shared memory avoids repeatedly copying the big “always needed” columns.
- Peak memory reasoning: `always_needed` shared + per-worker one feature column.

## 8) PyArrow dataset reading and conversion costs (secondary but relevant)

This code repeatedly does:

- `data.to_table(columns=[c]).to_pandas()[c]` — Study:
- What a `pyarrow.dataset.Dataset` is, scanning columns, and why selecting columns matters.
- Conversion costs from Arrow to pandas, and how that interacts with multiprocessing.

## 9) Context managers and resource safety (to follow the control flow)

Because shared memory must be cleaned up correctly:

- `@contextmanager`, `__enter__` / `__exit__`
- Why `_SharedDataFrame` unlinks only in the parent.
- Why `_frozen_config` temporarily changes `__class__` (uncommon pattern) to prevent mutation during parallel execution.

-----

## Concrete “learning path” (condensed)

1. `ProcessPoolExecutor` + futures + pickling rules
1. `multiprocessing` start methods (`fork` vs `spawn`) + copy-on-write
1. `multiprocessing.shared_memory` (`close` vs `unlink`)
1. NumPy arrays backed by external buffers (`buffer=...`)
1. Pandas copy/view behavior when building Series/DataFrames
1. PyArrow dataset → pandas conversion and column projection