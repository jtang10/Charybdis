//===- Charybdis.h - C API for the Charybdis dialect ---------------------===//
//
// The C API is the ABI-stable boundary between the C++ dialect implementation
// and the Python bindings layer. Python never calls C++ directly — it calls
// through this header via the nanobind extension module.
//
// Why a C API instead of direct C++ exposure?
//   C++ has no stable ABI. Linking Python extensions directly against C++
//   MLIR symbols leads to ODR violations and subtle crashes when the Python
//   interpreter, the extension, and MLIR are built with different compiler
//   versions or flags. The C API sidesteps this by providing a flat,
//   ABI-stable surface that nanobind can call through safely.
//   This is the same pattern used by the upstream MLIR Python bindings.
//
// Phase 0: only one function — the dialect handle registration hook.
// Phase 1+ will add C API functions for Tile types, op builders, etc.
//
//===----------------------------------------------------------------------===//

#ifndef CHARYBDIS_C_CHARYBDIS_H
#define CHARYBDIS_C_CHARYBDIS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the dialect handle for the Charybdis (kbd) dialect.
//
// This function is called by the nanobind extension's register_dialects()
// hook to insert the dialect into an MlirDialectRegistry. Once registered,
// any MlirContext created from that registry will be able to parse and
// construct kbd.* ops.
//
// The naming convention mlirGetDialectHandle__<name>__ is required by the
// MLIR Python binding infrastructure — it is the hook that
// mlirDialectHandleInsertDialect() looks for by convention.
MLIR_CAPI_EXPORTED MlirDialectHandle mlirGetDialectHandle__kbd__(void);

#ifdef __cplusplus
}
#endif

#endif // CHARYBDIS_C_CHARYBDIS_H
