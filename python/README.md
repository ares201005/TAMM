# TAMM Python Wrapper (`pytamm`)

This folder contains a `pybind11` wrapper for core TAMM tensor workflows:

- create tiled index spaces
- create tensors
- allocate/deallocate tensors
- fill tensors
- perform tensor contractions
- move tensor data to/from NumPy

## Build

Configure TAMM with Python enabled:

```bash
cmake -S . -B build-macos -DTAMM_ENABLE_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python3)
cmake --build build-macos --target pytamm -j
```

The extension module will be generated in:

```text
build-macos/python/
```

## Usage

See `tensor_contraction.py` for a complete example.

Run the example directly (auto-selects `LOCAL` backend in single-process mode):

```bash
PYTHONPATH=build-macos/python python python/tensor_contraction.py
```

Run with MPI (auto-selects `GA` when world size >= 2):

```bash
PYTHONPATH=build-macos/python mpirun -np 2 python python/tensor_contraction.py
```

Force backend choice:

```bash
PYTHONPATH=build-macos/python python python/tensor_contraction.py --memory-manager local
PYTHONPATH=build-macos/python mpirun -np 2 python python/tensor_contraction.py --memory-manager ga
```

Typical flow:

1. Create a `Context`.
2. Define reusable tiled index spaces.
3. Build tensors from those spaces.
4. Allocate tensors.
5. Fill tensors or load from NumPy.
6. Contract tensors.
7. Read results with `to_numpy`.
8. Deallocate tensors and close context.

Convenience API (numpy-style):

- `pytamm.TammContext` is an alias of `Context` with defaults:
  - `distribution=DistributionKind.NW`
  - `memory_manager=MemoryManagerKind.GA`
- `ctx.einsum("ijk,km->ijm", a, b, out=None)` is available as an alias to
  `contract_einsum`, with automatic output tensor creation when `out` is omitted.
- `ctx.zeros(shape, tile=None)` and `ctx.ones(shape, tile=None)` create, allocate,
  and initialize tensors.
  - `shape` can be a tuple/list of dimensions (for example `(200, 100, 80)`), or
    a list of `TiledIndexSpace` objects.
  - If `tile` is omitted, tile sizes are auto-selected per dimension.
- Module-level helpers mirror numpy style:
  - `pytamm.zeros(ctx, shape, tile=None)`
  - `pytamm.ones(ctx, shape, tile=None)`

## Notes

- This first wrapper currently targets `double` tensors.
- `contract_einsum` supports binary expressions in the form `ab,bc->ac`.
- Keep one active `Context` at a time.
- Use the same Python executable for build and runtime to avoid ABI mismatch.
- In this TAMM build, GA requires at least 2 MPI ranks per node.
