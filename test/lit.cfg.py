# lit.cfg.py — LLVM Integrated Tester configuration for Charybdis FileCheck tests.
#
# lit (LLVM Integrated Tester) scans the test directory for files matching
# config.suffixes and runs the RUN: lines inside them. FileCheck is the
# tool that validates the output matches the CHECK: patterns.
#
# To run: llvm-lit test/lit/
# or:     cmake --build build --target check-charybdis  (once wired in CMake)

import lit.formats

config.name = "Charybdis"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# File extensions lit will scan for test files.
config.suffixes = [".mlir"]

# The test source directory (where .mlir files live).
config.test_source_root = os.path.dirname(__file__)

# The build directory where charybdis-opt was placed by CMake.
config.test_exec_root = os.path.join(config.charybdis_obj_root, "test")

# Make charybdis-opt and FileCheck available on PATH for RUN: lines.
config.substitutions.append(
    ("%charybdis-opt", os.path.join(config.charybdis_obj_root, "bin", "charybdis-opt"))
)

# llvm_config is provided by lit's LLVM integration and sets up FileCheck,
# not, count, and other standard LLVM test utilities.
llvm_config.use_default_substitutions()
llvm_config.add_tool_substitutions(["FileCheck", "count", "not"],
                                    config.llvm_tools_dir)
