add_mlir_library(MLIRCharybdisTransforms
  CharybdisToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/charybdis/Transforms

  DEPENDS
    MLIRCharybdisOpsIncGen

  LINK_LIBS PUBLIC
    MLIRCharybdis
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
)
