"""
test_phase0.py — Phase 0 end-to-end Python test.

Validates the full stack from Python down to LLVM dialect IR:
  1. The charybdis (kbd) dialect can be loaded into an MLIRContext.
  2. kbd.identity can be constructed in Python via the nanobind bindings.
  3. The MLIR module containing kbd.identity can be printed.
  4. The convertCharybdisToLLVM pass lowers kbd.identity away.
  5. The resulting LLVM dialect IR contains no charybdis ops.

PTX emission (step 6) is stubbed — it requires a Linux machine with an NVPTX
LLVM backend and is deferred to Phase 1 when GPU testing becomes available.

Run with:
  PYTHONPATH=build/python_packages python test/test_phase0.py
"""

import sys

# ---------------------------------------------------------------------------
# Imports
#
# The MLIR Python bindings are in the upstream mlir package, which is included
# in our built Python package via add_mlir_python_common_capi_library().
# Our charybdis.dialects.charybdis module wraps the tblgen-generated op
# classes and the nanobind register_dialects() hook.
# ---------------------------------------------------------------------------
from mlir import ir
from mlir import passmanager
from mlir.dialects import builtin
from mlir.dialects import func as func_dialect
import charybdis.dialects.charybdis as kbd


def build_test_module(ctx: ir.Context) -> ir.Module:
    """
    Construct a minimal MLIR module containing one kbd.identity op.

    The module looks like:
      func.func @test(%arg0: f32) -> f32 {
        %0 = kbd.identity %arg0 : f32
        return %0 : f32
      }

    This is the simplest possible kernel structure. Using func.func here is
    deliberate: Phase 1 will introduce gpu.func and the charybdis @kernel
    decorator, but for Phase 0 we want to exercise the dialect machinery
    without any GPU-specific scaffolding.
    """
    with ir.Location.unknown(ctx):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            f32 = ir.F32Type.get(ctx)
            func_type = ir.FunctionType.get([f32], [f32], ctx)
            func_op = func_dialect.FuncOp("test", func_type)
            func_op.add_entry_block()
            with ir.InsertionPoint(func_op.entry_block):
                arg = func_op.entry_block.arguments[0]
                # Construct kbd.identity. The IdentityOp class was generated
                # by mlir-tblgen from CharybdisOps.td.
                result = kbd.IdentityOp(arg).result
                func_dialect.ReturnOp([result])
    return module


def test_mlir_construction():
    """Step 1-3: construct the module and verify kbd.identity appears."""
    print("=== Step 1: Create MLIRContext and load charybdis dialect ===")
    with ir.Context() as ctx:
        # Loading the dialect registers it into the context so the parser
        # accepts kbd.* ops and the verifier knows their semantics.
        ctx.load_all_available_dialects()

        print("=== Step 2: Construct module with kbd.identity ===")
        module = build_test_module(ctx)

        print("=== Step 3: Print MLIR module ===")
        mlir_text = str(module)
        print(mlir_text)

        assert "kbd.identity" in mlir_text, (
            "FAIL: kbd.identity not found in printed module.\n"
            "This means either the dialect is not registered or the op "
            "construction failed silently."
        )
        print("PASS: kbd.identity is present in the module.\n")


def test_lowering():
    """Steps 4-5: run the lowering pass and verify kbd.identity is gone."""
    print("=== Step 4: Lower kbd.identity via convertCharybdisToLLVM pass ===")
    with ir.Context() as ctx:
        ctx.load_all_available_dialects()
        module = build_test_module(ctx)

        # Run the pass pipeline. The pass manager string syntax is the same
        # as mlir-opt's --pass-pipeline flag.
        #
        # convert-charybdis-to-llvm replaces kbd.identity with its operand.
        # reconcile-unrealized-casts cleans up any type cast scaffolding
        # introduced by the dialect conversion framework.
        pm = passmanager.PassManager.parse(
            "builtin.module(convert-charybdis-to-llvm,reconcile-unrealized-casts)",
            ctx,
        )
        pm.run(module.operation)

        print("=== Step 5: Print lowered LLVM dialect IR ===")
        llvm_text = str(module)
        print(llvm_text)

        assert "kbd.identity" not in llvm_text, (
            "FAIL: kbd.identity still present after lowering.\n"
            "The ConversionPattern in CharybdisToLLVM.cpp did not fire."
        )
        print("PASS: kbd.identity has been lowered away.\n")

    # Step 6 (stubbed): PTX emission.
    # On Linux with LLVM built with -DLLVM_TARGETS_TO_BUILD="NVPTX", the
    # following would emit PTX:
    #
    #   from mlir.execution_engine import ExecutionEngine
    #   # ... lower to LLVM IR, then:
    #   import subprocess
    #   ptx = subprocess.run(
    #       ["llc", "--march=nvptx64", "--mcpu=sm_90a", "-o", "-"],
    #       input=llvm_ir.encode(), capture_output=True
    #   ).stdout.decode()
    #
    # Deferred to Phase 1 when a Linux+GPU machine is available for testing.
    print("Step 6 (PTX emission): deferred to Phase 1 (requires Linux + NVPTX).")


if __name__ == "__main__":
    try:
        test_mlir_construction()
        test_lowering()
        print("=== All Phase 0 tests passed ===")
    except AssertionError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
