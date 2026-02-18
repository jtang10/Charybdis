//===- charybdis-opt.cpp - Charybdis dialect driver tool -----------------===//
//
// A minimal mlir-opt clone that registers the charybdis (kbd) dialect and
// its lowering pass. Used for FileCheck-based MLIR round-trip tests.
//
// Why a custom opt binary instead of upstream mlir-opt?
//   mlir-opt links every upstream dialect but knows nothing about charybdis.
//   We need an opt binary that can parse and print kbd.* ops. This is the
//   standard pattern for out-of-tree dialects — see mlir/examples/standalone.
//
// Usage (in a .mlir FileCheck test):
//   // RUN: charybdis-opt %s | FileCheck %s
//   // RUN: charybdis-opt %s --convert-charybdis-to-llvm | FileCheck %s
//
//===----------------------------------------------------------------------===//

#include "charybdis/CharybdisDialect.h"
#include "charybdis/CharybdisOps.h"
#include "charybdis/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register the charybdis (kbd) dialect so the parser accepts kbd.* ops.
  registry.insert<mlir::charybdis::CharybdisDialect>();

  // Register built-in dialects that appear in lowered output or test inputs.
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  // Register the charybdis lowering pass so it can be invoked via
  // --convert-charybdis-to-llvm on the command line.
  //
  // We use the factory function rather than PassRegistration<ConcretePass>
  // because the pass struct is defined in an anonymous namespace in the .cpp
  // and is not visible here. The factory function (declared in Passes.h) is
  // the public API surface.
  mlir::PassRegistration<mlir::Pass>(
      "convert-charybdis-to-llvm",
      "Lower charybdis (kbd) ops to LLVM dialect",
      []() { return mlir::charybdis::createConvertCharybdisToLLVMPass(); });

  // MlirOptMain handles arg parsing, IR parsing, pass pipeline execution,
  // and output printing. It is the standard entry point for opt tools.
  return mlir::MlirOptMain(argc, argv, "Charybdis optimizer driver\n",
                            registry)
             .succeeded()
         ? 0
         : 1;
}
