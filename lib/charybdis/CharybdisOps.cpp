//===- CharybdisOps.cpp - Charybdis op implementations -------------------===//
//
// Implements custom verifiers, builders, and folders for charybdis ops.
//
// Phase 0: empty. kbd.identity has no custom logic — its behavior is fully
// described by its TableGen traits (Pure, SameOperandsAndResultType) and
// assembly format. The generated .inc file provides all required methods.
//
// Phase 1+ will add implementations here as ops with non-trivial semantics
// are introduced. For example:
//   - kbd.tile_elementwise verifier: check that operand shapes are compatible
//   - kbd.warpgroup_mma verifier: enforce wgmma shape constraints (M=64/N=8/K
//     multiples, accumulator must be f32 when inputs are f16, etc.)
//
//===----------------------------------------------------------------------===//

#include "charybdis/CharybdisOps.h"

using namespace mlir;
using namespace mlir::charybdis;

// Pull in the tblgen-generated op method definitions (verify, build, print,
// parse). For Phase 0, this is all that's needed for kbd.identity.
#define GET_OP_CLASSES
#include "charybdis/CharybdisOps.cpp.inc"
