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

## Notes

- This first wrapper currently targets `double` tensors.
- `contract_einsum` supports binary expressions in the form `ab,bc->ac`.
- Keep one active `Context` at a time.
- Use the same Python executable for build and runtime to avoid ABI mismatch.
- In this TAMM build, GA requires at least 2 MPI ranks per node.
