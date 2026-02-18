//===- CharybdisExtensionNanobind.cpp - Python binding entry point --------===//
//
// This file is the nanobind C++ extension module that bridges the Charybdis
// C API to Python.
//
// How MLIR Python bindings work (the upstream approach):
//
//   The upstream MLIR Python package (mlir.ir, mlir.passmanager, etc.) is
//   built with nanobind and exposes MLIRContext, Module, Op, Type, etc.
//   Custom dialects plug into this infrastructure by:
//
//     1. Defining a C API (include/charybdis-c/Charybdis.h)
//     2. Writing a nanobind NB_MODULE that calls into that C API
//     3. Exposing a register_dialects() function that inserts the dialect
//        handle into an MlirDialectRegistry
//
//   When Python imports charybdis.dialects.charybdis, the binding
//   infrastructure calls register_dialects() to load the kbd dialect into
//   the current MLIRContext so that kbd.* ops can be constructed and parsed.
//
// Why nanobind instead of pybind11?
//   As of LLVM 20, the upstream MLIR Python bindings switched from pybind11
//   to nanobind (faster compile times, smaller binaries, cleaner ABI). Our
//   extension must use the same binding library as the upstream package to
//   avoid symbol conflicts when they share a Python process.
//
//===----------------------------------------------------------------------===//

#include "charybdis-c/Charybdis.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

// NB_MODULE declares the Python extension module. The name here must exactly
// match the MODULE_NAME argument passed to declare_mlir_python_extension()
// in python/CMakeLists.txt. Python's import system uses this name to dlopen
// the .so when `import charybdis._mlir_libs._charybdisDialectsNanobind`
// is executed (indirectly via the dialects/charybdis.py wrapper).
NB_MODULE(_charybdisDialectsNanobind, m) {
  // register_dialects is the hook called by MLIR's Python binding
  // infrastructure when the dialect module is imported. It inserts the
  // kbd dialect into the given DialectRegistry so that any MLIRContext
  // created from that registry can parse and construct kbd.* ops.
  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirDialectHandleInsertDialect(mlirGetDialectHandle__kbd__(), registry);
  });
}
