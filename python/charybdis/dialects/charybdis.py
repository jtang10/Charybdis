# charybdis/dialects/charybdis.py
#
# Python module for the charybdis (kbd) dialect.
#
# How this file works:
#   MLIR's tblgen -gen-python-op-bindings generates Python op classes
#   (IdentityOp, etc.) from CharybdisOps.td and places them in an auto-
#   generated file in the build tree. This hand-written file acts as the
#   public module: it imports from the auto-generated binding and re-exports
#   everything, so users write:
#
#     from charybdis.dialects import charybdis as kbd
#     op = kbd.IdentityOp(value)
#
# The _Dialect class registration happens automatically when this module is
# imported, via the register_dialects() hook in the nanobind extension.
#
# Phase 0: re-exports IdentityOp from the auto-generated binding.
# Phase 1+: will add Tile type helpers, op builder wrappers, etc.

from ._charybdis_ops_ext import *  # auto-generated op classes from tblgen
