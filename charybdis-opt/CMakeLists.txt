get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# charybdis-opt is a minimal clone of mlir-opt that links the charybdis
# dialect. It is used exclusively for FileCheck-based MLIR round-trip tests:
#
#   // RUN: charybdis-opt %s | FileCheck %s
#
# Having our own opt binary (rather than using upstream mlir-opt) ensures we
# control exactly which dialects are registered, making tests hermetic.
add_llvm_executable(charybdis-opt
  charybdis-opt.cpp
)

target_link_libraries(charybdis-opt PRIVATE
  # Core MLIR infrastructure needed to run any opt tool.
  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport

  # Built-in dialects that charybdis ops may reference (func, arith, etc.)
  MLIRFuncDialect
  MLIRArithDialect

  # Our dialect and its lowering pass.
  MLIRCharybdis
  MLIRCharybdisTransforms

  # Upstream dialect libs discovered at configure time.
  ${dialect_libs}
)

llvm_update_compile_flags(charybdis-opt)
mlir_check_all_link_libraries(charybdis-opt)
