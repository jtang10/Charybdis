add_mlir_library(CharybdisCAP
  Charybdis.cpp

  LINK_LIBS PUBLIC
    # MLIRCAPIIR provides MlirDialectHandle and related C API types.
    MLIRCAPIIR
    # Our C++ dialect library — the C API delegates to it.
    MLIRCharybdis
)
