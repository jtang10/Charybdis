# add_mlir_dialect(TD_STEM DIALECT_NAMESPACE) does two things:
#
# 1. Runs mlir-tblgen twice on CharybdisOps.td to generate:
#      CharybdisOps.h.inc      — C++ op class declarations
#      CharybdisOps.cpp.inc    — C++ op class definitions
#      CharybdisDialect.h.inc  — C++ dialect class declaration
#      CharybdisDialect.cpp.inc— C++ dialect class definition
#
# 2. Creates the CMake target MLIRCharybdisOpsIncGen, which lib/charybdis/
#    depends on so the .inc files exist before C++ compilation begins.
#
# The TD_STEM argument ("CharybdisOps") must match the filename of the root
# .td file that includes both the dialect and all ops. The namespace argument
# ("charybdis") matches the cppNamespace declared inside the .td file.
add_mlir_dialect(CharybdisOps charybdis)

# Also generate the Python op binding stubs from the same .td file.
# mlir-tblgen -gen-python-op-bindings reads the op defs and emits Python
# classes (e.g. IdentityOp) that wrap the C API. The output is placed in
# the build tree and picked up by the python/ CMakeLists.
add_mlir_doc(CharybdisOps CharybdisOps charybdis/ -gen-dialect-doc)
