//===- Passes.h - Charybdis lowering pass declarations -------------------===//
//
// Declares the entry points for all lowering passes in the charybdis dialect.
//
// Pass architecture overview:
//
//   The charybdis dialect sits above two independent lowering targets:
//
//     GPU path:  charybdis → nvgpu + gpu + arith + memref → nvvm → PTX
//     CPU path:  charybdis → linalg → loops → LLVM IR (via MLIR JIT)
//
//   Phase 0 adds only convertCharybdisToLLVM, which handles the single
//   scaffolding op (kbd.identity) by erasing it and forwarding its operand.
//   This is enough to drive an end-to-end pipeline test on macOS without a
//   GPU.
//
//   Future passes added here:
//     Phase 1: convertCharybdisToGPU   — kbd.tile_elementwise → gpu + arith
//              convertCharybdisToLinalg — kbd.tile_elementwise → linalg
//     Phase 2: convertCharybdisToNVGPU — kbd.warpgroup_mma → nvgpu.warpgroup.mma
//                                         kbd.tma_load → nvgpu.tma.async.load
//
//===----------------------------------------------------------------------===//

#ifndef CHARYBDIS_TRANSFORMS_PASSES_H
#define CHARYBDIS_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir::charybdis {

// Creates a pass that lowers charybdis ops to LLVM dialect ops.
//
// Phase 0: the only lowering rule is kbd.identity → operand passthrough.
// The pass uses MLIR's dialect conversion framework (ConversionPattern +
// ConversionTarget) so that future rules can be added without restructuring
// the pass.
std::unique_ptr<mlir::Pass> createConvertCharybdisToLLVMPass();

} // namespace mlir::charybdis

#endif // CHARYBDIS_TRANSFORMS_PASSES_H
