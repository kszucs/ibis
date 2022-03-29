from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt

from .generic import Expr, HasName


@public
class SortExpr(Expr, HasName):
    def get_name(self) -> str | None:
        return self.op().resolve_name()

    def type(self) -> dt.DataType:
        return self.op().expr.type()
