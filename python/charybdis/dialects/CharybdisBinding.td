//===- CharybdisBinding.td - TableGen root for Python bindings generation -===//
//
// This file is the TableGen entry point used by
// declare_mlir_dialect_python_bindings() in python/CMakeLists.txt.
//
// mlir-tblgen -gen-python-op-bindings reads this file to auto-generate
// Python wrapper classes for each op in the charybdis (kbd) dialect. The
// generated output is merged with our hand-written dialects/charybdis.py.
//
// A separate .td file is used here (rather than pointing at CharybdisOps.td
// directly) because the python binding generator requires includes relative
// to the TD file's location, and the python/ tree has a different base path
// than include/.
//
//===----------------------------------------------------------------------===//

#ifndef CHARYBDIS_BINDING
#define CHARYBDIS_BINDING

include "charybdis/CharybdisOps.td"

#endif // CHARYBDIS_BINDING
