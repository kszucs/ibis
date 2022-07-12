from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ibis.expr.types as ir

from public import public

import ibis.expr.operations as ops
from ibis.expr.types.core import _binop
from ibis.expr.types.numeric import NumericColumn, NumericScalar, NumericValue


@public
class BooleanValue(NumericValue):
    def ifelse(
        self,
        true_expr: ir.Value,
        false_expr: ir.Value,
    ) -> ir.Value:
        """Construct a ternary conditional expression.

        Parameters
        ----------
        true_expr
            Expression to return if `self` evaluates to `True`
        false_expr
            Expression to return if `self` evaluates to `False`

        Returns
        -------
        Value
            The value of `true_expr` if `arg` is `True` else `false_expr`

        Examples
        --------
        >>> import ibis
        >>> t = ibis.table([("is_person", "boolean")], name="t")
        >>> expr = t.is_person.ifelse("yes", "no")
        >>> print(ibis.impala.compile(expr))
        SELECT CASE WHEN `is_person` THEN 'yes' ELSE 'no' END AS `tmp`
        FROM t
        """
        import ibis

        # Result will be the result of promotion of true/false exprs. These
        # might be conflicting types; same type resolution as case expressions
        # must be used.
        return ibis.case().when(self, true_expr).else_(false_expr).end()

    def __and__(self, other: BooleanValue) -> BooleanValue:
        return _binop(ops.And, self, other)

    __rand__ = __and__

    def __or__(self, other: BooleanValue) -> BooleanValue:
        return _binop(ops.Or, self, other)

    __ror__ = __or__

    def __xor__(self, other: BooleanValue) -> BooleanValue:
        return _binop(ops.Xor, self, other)

    __rxor__ = __xor__

    def __invert__(self) -> BooleanValue:
        return self.negate()

    @staticmethod
    def __negate_op__():
        return ops.Not


@public
class BooleanScalar(NumericScalar, BooleanValue):
    pass  # noqa: E701,E302


@public
class BooleanColumn(NumericColumn, BooleanValue):
    def any(self) -> BooleanValue:
        import ibis.expr.analysis as L

        return L._make_any(self, ops.Any)

    def notany(self) -> BooleanValue:
        import ibis.expr.analysis as L

        return L._make_any(self, ops.NotAny)

    def all(self) -> BooleanScalar:
        return ops.All(self).to_expr()

    def notall(self) -> BooleanScalar:
        return ops.NotAll(self).to_expr()

    def cumany(self) -> BooleanColumn:
        return ops.CumulativeAny(self).to_expr()

    def cumall(self) -> BooleanColumn:
        return ops.CumulativeAll(self).to_expr()
