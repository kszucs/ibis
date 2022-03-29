from __future__ import annotations

from public import public

from .generic import Expr, HasName


@public
class SortExpr(Expr, HasName):
    def get_name(self) -> str | None:
        return self.op().resolve_name()
