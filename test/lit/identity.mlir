// RUN: %charybdis-opt %s | FileCheck %s
// RUN: %charybdis-opt %s --convert-charybdis-to-llvm | FileCheck %s --check-prefix=LOWERED

// Test 1 (CHECK): Round-trip — charybdis-opt can parse and re-print kbd.identity.
// This validates that the dialect is registered, the op is recognized, and the
// assembly format printer/parser round-trips correctly.

// Test 2 (LOWERED): After --convert-charybdis-to-llvm, kbd.identity must be
// gone (lowered to a passthrough). The function should still exist and return
// its argument, just without any charybdis ops in the body.

// CHECK-LABEL: func.func @test_identity
// CHECK:         kbd.identity
// CHECK-NOT:     error

// LOWERED-LABEL: func.func @test_identity
// LOWERED-NOT:   kbd.identity

func.func @test_identity(%arg0: f32) -> f32 {
  // kbd.identity is the Phase 0 scaffolding op. It should round-trip
  // through charybdis-opt and be lowered away by --convert-charybdis-to-llvm.
  %0 = kbd.identity %arg0 : f32
  return %0 : f32
}
