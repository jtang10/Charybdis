# Charybdis Python package root.
#
# Phase 0: empty. The only entry point is the dialect module at
# charybdis.dialects.charybdis, which exposes IdentityOp.
#
# Phase 1 will add:
#   from charybdis.kernel import kernel   # @kernel decorator
#   from charybdis.types import Tile      # Tile[dtype, shape] type
#   from charybdis.ops import reduce      # DSL ops
