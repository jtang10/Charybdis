cmake_minimum_required(VERSION 3.20.0)
project(charybdis LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------------------
# Find MLIR (and transitively LLVM) from a pre-built install.
#
# Point -DMLIR_DIR at the cmake dir inside your install prefix, e.g.:
#   cmake -B build -DMLIR_DIR=~/llvm-20-install/lib/cmake/mlir
#
# We deliberately do NOT use LLVM as a git submodule. Building LLVM takes
# 30-90 min and ~50 GB of disk; doing it once and reusing the install is the
# standard approach used by Triton and Mosaic GPU.
# ---------------------------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Pull in the CMake helper modules that MLIR ships. These provide:
#   add_mlir_dialect()             — TableGen → .inc generation
#   add_mlir_dialect_library()     — link dialect C++ library
#   declare_mlir_python_sources()  — Python package tree
#   declare_mlir_python_extension()— nanobind .so
#   add_mlir_python_modules()      — assemble built package
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Output directories follow LLVM convention so that mlir-opt / lit can find
# dialect .so files next to the charybdis-opt binary.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

# Expose generated .inc files (e.g. CharybdisOps.h.inc) from the build tree.
# Both the source tree (hand-written headers) and the build tree
# (tblgen-generated .inc files) need to be on the include path.
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(charybdis-opt)

# ---------------------------------------------------------------------------
# Python bindings — only built when MLIR was itself built with Python support.
#
# The upstream MLIR Python bindings use nanobind (as of LLVM 20) and expose
# MLIRContext, Module, Op, Type, etc. Our extension registers the charybdis
# dialect into that infrastructure so Python can call:
#
#   from charybdis.dialects import charybdis as kbd
#   kbd.IdentityOp(...)
#
# We follow Mosaic GPU's approach: reuse the upstream MLIR Python binding
# infrastructure rather than writing our own pybind11/nanobind wrappers from
# scratch. This avoids reimplementing what MLIR already provides.
# ---------------------------------------------------------------------------
if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
  include(AddMLIRPython)
  add_subdirectory(python)
endif()
