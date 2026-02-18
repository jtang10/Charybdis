//===- Charybdis.cpp - C API implementation for the Charybdis dialect ----===//
//
// Implements the C API declared in include/charybdis-c/Charybdis.h.
//
// Why this layer exists:
//   Python's nanobind extension cannot safely link against C++ MLIR symbols
//   directly because C++ has no stable ABI. Instead, the extension calls
//   through this flat C API. MLIR's own Python bindings follow the same
//   pattern — every dialect exposes a C API that the nanobind extension calls.
//
//   The naming convention mlirGetDialectHandle__<mnemonic>__ is required by
//   the MLIR Python binding infrastructure. The mnemonic used here matches the
//   `let name = "kbd"` field in CharybdisDialect.td.
//
//===----------------------------------------------------------------------===//

#include "charybdis-c/Charybdis.h"
#include "charybdis/CharybdisDialect.h"

#include "mlir/CAPI/Registration.h"

// MLIR_DEFINE_CAPI_DIALECT_REGISTRATION is a macro that expands to the
// implementation of mlirGetDialectHandle__kbd__(). It uses the C++ dialect
// class (CharybdisDialect) to produce the handle that Python's
// mlirDialectHandleInsertDialect() consumes when registering the dialect
// into a context.
//
// The first argument is the C++ dialect class name (without the "Dialect"
// suffix in the macro — it appends "Dialect" internally).
// The second argument is the mnemonic string, which must match `let name`
// in the .td file.
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Charybdis, kbd,
                                       mlir::charybdis::CharybdisDialect)
