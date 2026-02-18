//===- CharybdisDialect.cpp - Charybdis dialect registration -------------===//
//
// Implements the CharybdisDialect class, specifically the initialize() method
// that registers all ops into the dialect at startup.
//
// MLIR's dialect system works as follows:
//   1. At program startup, each dialect is registered into a DialectRegistry.
//   2. When an MLIRContext is created, it loads dialects from the registry.
//   3. The dialect's initialize() method is called exactly once per context,
//      and it is responsible for calling addOperations<>() to register the
//      op classes that belong to this dialect.
//
// The generated .inc files (produced by mlir-tblgen from the .td files) are
// included here to pull in the C++ method bodies that TableGen emitted.
//
//===----------------------------------------------------------------------===//

#include "charybdis/CharybdisDialect.h"
#include "charybdis/CharybdisOps.h"

// Pull in the tblgen-generated dialect class definition.
// This file is written to the build tree by add_mlir_dialect().
#include "charybdis/CharybdisDialect.cpp.inc"

using namespace mlir;
using namespace mlir::charybdis;

void CharybdisDialect::initialize() {
  // Register all ops defined in CharybdisOps.td.
  // The variadic template pack is expanded by the generated .inc file —
  // addOperations<IdentityOp, TileOp, ...> registers each op class so that
  // the MLIR parser and verifier know about them.
  //
  // Phase 0: only IdentityOp. Future ops are added here as they are defined.
  addOperations<
#define GET_OP_LIST
#include "charybdis/CharybdisOps.cpp.inc"
  >();
}
