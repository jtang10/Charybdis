//===- CharybdisOps.td - Charybdis op definitions ----------*- tablegen -*-===//
//
// Defines all ops in the charybdis (kbd) dialect.
//
// Phase 0: one scaffolding op — kbd.identity — to exercise the full toolchain.
// Phase 1+ will add kbd.tile_elementwise, kbd.warpgroup_mma, etc.
//
//===----------------------------------------------------------------------===//

#ifndef CHARYBDIS_OPS
#define CHARYBDIS_OPS

include "charybdis/CharybdisDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// SameOperandsAndResultType is defined in InferTypeOpInterface.td — not in
// OpBase.td as one might expect. It is a NativeOpTrait that tells MLIR the
// result type can be inferred from the operand type, eliminating the need
// for a hand-written type verifier on this op.
include "mlir/Interfaces/InferTypeOpInterface.td"

// ---------------------------------------------------------------------------
// kbd.identity — Phase 0 scaffolding op
// ---------------------------------------------------------------------------
//
// This op exists only to validate the toolchain end-to-end. It has no GPU
// semantics; the lowering pass replaces it with its operand directly.
//
// Trait choices:
//
//   Pure — marks the op as having no memory side effects. This enables MLIR's
//   CSE and DCE passes to eliminate or deduplicate it without needing a custom
//   MemoryEffects interface implementation. Every op in the charybdis dialect
//   that is a pure computation (no shared memory writes, no barriers) should
//   carry this trait.
//
//   SameOperandsAndResultType — the result type is identical to the operand
//   type, so MLIR can infer the result type automatically. This eliminates the
//   need for a hand-written type verifier on this op.
//
// ---------------------------------------------------------------------------
def Charybdis_IdentityOp : Charybdis_Op<"identity",
    [Pure, SameOperandsAndResultType]> {
  let summary = "Returns its input unchanged (Phase 0 scaffolding op).";

  let description = [{
    kbd.identity is a no-op that passes its argument through unchanged.
    It is used in Phase 0 to verify that:
      - TableGen generates correct C++ headers
      - The dialect registers and loads correctly
      - The Python bindings can construct the op
      - The convertCharybdisToLLVM pass can lower it away

    Example:
    ```mlir
    %1 = kbd.identity %0 : f32
    ```
  }];

  let arguments = (ins AnyType:$input);
  let results  = (outs AnyType:$result);

  // Custom assembly format. "$input" prints the operand SSA value,
  // "attr-dict" prints any attributes (none here), "`:` type($input)"
  // prints the type. SameOperandsAndResultType lets the parser infer
  // the result type from the operand type without repeating it.
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // CHARYBDIS_OPS
