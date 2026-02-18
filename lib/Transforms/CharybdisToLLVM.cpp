//===- CharybdisToLLVM.cpp - Lower charybdis ops to LLVM dialect ---------===//
//
// Implements the convertCharybdisToLLVM pass.
//
// MLIR's lowering model: progressive dialect conversion
//
//   MLIR does not lower dialects in one shot. Instead, a pipeline of passes
//   each converts a subset of ops from a higher-level dialect to a lower-level
//   one. The ConversionTarget + ConversionPattern mechanism enforces this:
//
//     ConversionTarget — declares which ops are "legal" (already lowered) and
//       which are "illegal" (must be converted). If any illegal op remains
//       after the pass, MLIR aborts with a diagnostic.
//
//     ConversionPattern — a rewrite pattern that converts one op from the
//       source dialect to one or more ops in the target dialect. Patterns are
//       collected into a RewritePatternSet and applied by the conversion driver.
//
// Phase 0 lowering:
//
//   kbd.identity is a pure passthrough. The correct lowering is to replace
//   the op's result with its operand. The ConversionPattern for IdentityOp
//   does exactly this using rewriter.replaceOp(op, op.getInput()).
//
//   This is the simplest possible lowering: no new ops are created, no memory
//   is touched, no types change. It is enough to drive the end-to-end test.
//
// Phase 1+ will add patterns for kbd.tile_elementwise (→ gpu + arith + memref
// for the GPU path, or → linalg for the CPU path). Those patterns are more
// complex and live in separate files (CharybdisToNVGPU.cpp, etc.).
//
//===----------------------------------------------------------------------===//

#include "charybdis/Transforms/Passes.h"
#include "charybdis/CharybdisOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::charybdis;

namespace {

// ---------------------------------------------------------------------------
// Conversion pattern: kbd.identity → operand passthrough
// ---------------------------------------------------------------------------
//
// ConversionPattern is preferred over RewritePattern here because it
// integrates with ConversionTarget's legality tracking. When the driver
// runs, it verifies that after applying all patterns no illegal ops remain.
//
// The pattern simply forwards the op's input as its result. MLIR's SSA
// form guarantees that all uses of the result are updated automatically.
struct IdentityOpLowering : public OpConversionPattern<IdentityOp> {
  using OpConversionPattern<IdentityOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IdentityOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace all uses of kbd.identity's result with its input operand,
    // then erase the op. No new IR is created — this is a pure erasure.
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

// ---------------------------------------------------------------------------
// Pass definition
// ---------------------------------------------------------------------------

struct ConvertCharybdisToLLVMPass
    : public PassWrapper<ConvertCharybdisToLLVMPass, OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCharybdisToLLVMPass)

  StringRef getArgument() const override {
    return "convert-charybdis-to-llvm";
  }

  StringRef getDescription() const override {
    return "Lower charybdis (kbd) dialect ops to LLVM dialect ops.";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Declare what is legal after this pass runs. We mark:
    //   - Everything in the LLVM dialect: legal (target dialect)
    //   - charybdis ops: illegal (must all be converted)
    //
    // If any kbd.* op survives the pass, the driver will emit an error
    // rather than silently producing incorrect IR.
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<CharybdisDialect>();

    // Collect all conversion patterns for this pass.
    // Phase 0: only the identity passthrough.
    RewritePatternSet patterns(ctx);
    patterns.add<IdentityOpLowering>(ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// Factory function called by charybdis-opt and the Python test to create the
// pass. Returning a unique_ptr<Pass> is the standard MLIR pass factory API.
std::unique_ptr<mlir::Pass>
mlir::charybdis::createConvertCharybdisToLLVMPass() {
  return std::make_unique<ConvertCharybdisToLLVMPass>();
}
