include(AddMLIRPython)

# ---------------------------------------------------------------------------
# Python source tree declaration
#
# declare_mlir_python_sources declares the root of our Python package tree.
# All subsequent declare_mlir_dialect_python_bindings and
# declare_mlir_python_extension calls add their outputs to this root.
# ---------------------------------------------------------------------------
declare_mlir_python_sources(CharybdisPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    charybdis/__init__.py
)

# ---------------------------------------------------------------------------
# Auto-generate Python op bindings from TableGen
#
# mlir-tblgen -gen-python-op-bindings reads CharybdisOps.td and emits Python
# classes (e.g. IdentityOp) that wrap the C API. The generated file is placed
# in the build tree and added to the Python package.
#
# The DIALECT_NAME argument must match the `let name` field in the .td file
# ("kbd"). The generated module registers the dialect when imported.
# ---------------------------------------------------------------------------
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CharybdisPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/charybdis"
  TD_FILE dialects/CharybdisBinding.td
  SOURCES
    dialects/charybdis.py
  DIALECT_NAME charybdis
  DIALECT_EXTENSION_NAME charybdis
)

# ---------------------------------------------------------------------------
# Nanobind C++ extension
#
# This builds the _charybdisDialectsNanobind.so that Python dlopen's when
# the charybdis.dialects.charybdis module is imported. It links CharybdisCAP
# (our C API library) and MLIRCAPIIR (the upstream MLIR C API).
# ---------------------------------------------------------------------------
declare_mlir_python_extension(CharybdisPythonSources.NanobindExtension
  MODULE_NAME _charybdisDialectsNanobind
  ADD_TO_PARENT CharybdisPythonSources
  SOURCES
    CharybdisExtensionNanobind.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    CharybdisCAP
)

# ---------------------------------------------------------------------------
# Bundle into a single shared CAPI library
#
# add_mlir_python_common_capi_library merges CharybdisCAP, MLIRCAPIIR, and
# the upstream mlir-core C API into one shared library to avoid duplicate
# symbol errors when multiple MLIR extensions are loaded in the same Python
# process.
# ---------------------------------------------------------------------------
add_mlir_python_common_capi_library(CharybdisPythonCAPI
  INSTALL_COMPONENT CharybdisPythonModules
  INSTALL_DESTINATION "python_packages/charybdis/_mlir_libs"
  OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python_packages/charybdis/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    CharybdisPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.func
)

# ---------------------------------------------------------------------------
# Assemble the built Python package in the build tree
#
# After this runs, build/python_packages/charybdis/ is a fully importable
# Python package. The test adds this path to PYTHONPATH.
# ---------------------------------------------------------------------------
add_mlir_python_modules(CharybdisPythonModules
  ROOT_PREFIX "${PROJECT_BINARY_DIR}/python_packages/charybdis"
  INSTALL_PREFIX "python_packages/charybdis"
  DECLARED_SOURCES
    CharybdisPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.func
  COMMON_CAPI_LINK_LIBS
    CharybdisPythonCAPI
)
