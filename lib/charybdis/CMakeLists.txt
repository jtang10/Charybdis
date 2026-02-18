# add_mlir_dialect_library is a thin wrapper around add_mlir_library that
# additionally handles installing dialect headers and setting up the
# dependency on the TableGen-generated .inc targets.
#
# MLIRCharybdisOpsIncGen is the CMake target created by add_mlir_dialect()
# in include/charybdis/CMakeLists.txt. Listing it under DEPENDS ensures
# that the .inc files are generated before any C++ source in this library
# tries to #include them.
add_mlir_dialect_library(MLIRCharybdis
  CharybdisDialect.cpp
  CharybdisOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/charybdis

  DEPENDS
    MLIRCharybdisOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)
