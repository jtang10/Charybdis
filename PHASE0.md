# Phase 0 — Build System & Bootstrapping

**Goal:** Establish that all tooling hangs together before writing any real compiler code. One dummy op (`kbd.identity`), full stack from TableGen through Python bindings, two tests that pass on macOS without a GPU.

**IR mnemonic:** `kbd` (e.g. `kbd.identity`)
**C++ namespace:** `mlir::charybdis`
**Python package:** `charybdis`

---

## Checklist

### Build system

- [x] `CMakeLists.txt` — top-level: `find_package(MLIR)`, subdirs, Python binding toggle

### Dialect definition (TableGen)

- [x] `include/charybdis/CMakeLists.txt` — `add_mlir_dialect(CharybdisOps charybdis)`
- [x] `include/charybdis/CharybdisDialect.td` — dialect def (`let name = "kbd"`) + `Charybdis_Op<>` base class
- [x] `include/charybdis/CharybdisOps.td` — `kbd.identity` op definition
- [x] `include/charybdis/CharybdisDialect.h` — C++ dialect class header
- [x] `include/charybdis/CharybdisOps.h` — C++ op class header

### C API (required by Python bindings layer)

- [x] `include/charybdis-c/Charybdis.h` — declares `mlirGetDialectHandle__kbd__()`
- [x] `lib/CAPI/CMakeLists.txt`
- [x] `lib/CAPI/Charybdis.cpp` — implements the C API handle via `MLIR_DEFINE_CAPI_DIALECT_REGISTRATION`

### Dialect C++ library

- [x] `lib/charybdis/CMakeLists.txt` — `add_mlir_dialect_library(MLIRCharybdis ...)`
- [x] `lib/charybdis/CharybdisDialect.cpp` — `initialize()` registers the identity op
- [x] `lib/charybdis/CharybdisOps.cpp` — empty placeholder for future op verifiers
- [x] `lib/CMakeLists.txt` — wires charybdis, CAPI, Transforms subdirs

### Lowering pass

- [x] `include/charybdis/Transforms/Passes.h` — declares `createConvertCharybdisToLLVMPass()` entry point
- [x] `lib/Transforms/CMakeLists.txt`
- [x] `lib/Transforms/CharybdisToLLVM.cpp` — one pattern: `kbd.identity` → operand passthrough

### Python bindings

- [x] `python/CMakeLists.txt` — `declare_mlir_python_sources`, `declare_mlir_python_extension`, `add_mlir_python_modules`
- [x] `python/CharybdisExtensionNanobind.cpp` — `NB_MODULE`: `register_dialects()` hook
- [x] `python/charybdis/__init__.py` — package root (empty in Phase 0)
- [x] `python/charybdis/dialects/charybdis.py` — thin wrapper; tblgen auto-generates `IdentityOp`
- [x] `python/charybdis/dialects/CharybdisBinding.td` — TableGen root for Python binding generation

### Standalone tool

- [x] `charybdis-opt/CMakeLists.txt`
- [x] `charybdis-opt/charybdis-opt.cpp` — `mlir-opt` clone linking the charybdis dialect

### Tests

- [x] `test/lit.cfg.py` — FileCheck test runner config
- [x] `test/lit.site.cfg.py.in` — CMake-generated site config for lit
- [x] `test/CMakeLists.txt` — lit configuration + `check-charybdis` target
- [x] `test/lit/identity.mlir` — FileCheck: round-trip `kbd.identity` + lowering check
- [x] `test/test_phase0.py` — Python: construct op → print MLIR → lower → print LLVM IR

---

## Build instructions

### Prerequisites (one-time)

```bash
# Install Python dependencies required by MLIR's CMake detection scripts.
# pybind11: required by MLIRDetectPythonEnv.cmake even though LLVM 20 uses nanobind at runtime.
# nanobind: used by the actual LLVM 20 Python bindings.
# numpy: required for the memref <-> numpy buffer protocol bridge.
pip3 install pybind11 nanobind numpy

# Clone LLVM 20 — use a shallow clone to avoid downloading the full 4-5 GB history.
# Run this from your home directory (or wherever you want llvm-project to live,
# outside the Charybdis repo).
cd ~
git clone --depth 1 --branch llvmorg-20.1.4 https://github.com/llvm/llvm-project.git
```

### Step 1 — Build and install LLVM 20 (one-time, ~60 min)

Run from the directory containing `llvm-project/` (e.g. `~`).

```bash
cd ~
cmake -S llvm-project/llvm -B llvm-build \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="NVPTX;AArch64" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-20-install
cmake --build llvm-build --target install -j$(sysctl -n hw.logicalcpu)
```

Note: use `$HOME` instead of `~` for the install prefix. CMake does not expand
`~` — passing `~/llvm-20-install` creates a literal directory named `~` inside
your current directory instead of installing to your home directory.

Verify the install succeeded before proceeding:
```bash
ls $HOME/llvm-20-install/lib/cmake/mlir/MLIRConfig.cmake
```

### Step 2 — Configure Charybdis

Run from the Charybdis repo root (`~/Charybdis`).

```bash
cd ~/Charybdis
cmake -B build \
  -DMLIR_DIR=$HOME/llvm-20-install/lib/cmake/mlir \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release
```

### Step 3 — Build Charybdis

```bash
cmake --build build -j$(sysctl -n hw.logicalcpu)
```

### Step 4 — Run FileCheck tests

```bash
cmake --build build --target check-charybdis
# Or directly:
llvm-lit build/test/
```

### Step 5 — Run Python end-to-end test

```bash
PYTHONPATH=build/python_packages python test/test_phase0.py
```

---

## Exit criterion

Both of the following pass on macOS without a GPU:

```bash
llvm-lit test/lit/
PYTHONPATH=build/python_packages python test/test_phase0.py
```

`test/test_phase0.py` must demonstrate:
1. `kbd.identity` appears in the printed MLIR module
2. `kbd.identity` is absent from the printed LLVM dialect IR (op was lowered away)

---

## Placeholders intentionally left for Phase 1+

| File | What's missing |
|---|---|
| `lib/charybdis/CharybdisOps.cpp` | Empty — future op verifiers go here |
| `lib/Transforms/CharybdisToLLVM.cpp` | Only identity lowering — `kbd.tile_elementwise` added in Phase 1 |
| `include/charybdis/Transforms/Passes.h` | One pass only — `CharybdisToNVGPU`, `CharybdisToLinalg` added later |
| `python/charybdis/dialects/charybdis.py` | Thin wrapper — `Tile` type and `@kernel` decorator added in Phase 1 |
| `test/test_phase0.py` PTX emission | Stubbed with comment — deferred to Phase 1 on Linux+GPU |
