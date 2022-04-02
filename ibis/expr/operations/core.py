from __future__ import annotations

from abc import abstractmethod

import toolz
from public import public

from ...common.exceptions import ExpressionError
from ...common.grounds import Comparable
from ...common.validators import immutable_property
from ...util import UnnamedMarker
from .. import rules as rlz
from .. import types as ir
from ..schema import Schema
from ..signature import Annotable


@public
def distinct_roots(*expressions):
    # TODO: move to analysis
    roots = toolz.concat(expr.op().root_tables() for expr in expressions)
    return list(toolz.unique(roots))


def _compare_items(a, b):
    try:
        return a.equals(b)
    except AttributeError:
        if isinstance(a, tuple):
            return _compare_tuples(a, b)
        else:
            return a == b


def _compare_tuples(a, b):
    if len(a) != len(b):
        return False
    return all(map(_compare_items, a, b))


@public
class Node(Annotable, Comparable):
    @immutable_property
    def _flat_ops(self):
        import ibis.expr.types as ir

        return tuple(
            arg.op() for arg in self.flat_args() if isinstance(arg, ir.Expr)
        )

    def __equals__(self, other):
        return self._hash == other._hash and _compare_tuples(
            self.args, other.args
        )

    def equals(self, other):
        if not isinstance(other, Node):
            raise TypeError(
                "invalid equality comparison between Node and "
                f"{type(other)}"
            )
        return self.__cached_equals__(other)

    @property
    def inputs(self):
        return self.args

    @property
    def exprs(self):
        return [arg for arg in self.args if isinstance(arg, ir.Expr)]

    def blocks(self):
        # The contents of this node at referentially distinct and may not be
        # analyzed deeper
        return False

    def compatible_with(self, other):
        return self.equals(other)

    def is_ancestor(self, other):
        try:
            other = other.op()
        except AttributeError:
            pass

        return self.equals(other)

    def to_expr(self):
        return self.output_type(self)

    def resolve_name(self):
        raise ExpressionError(f'Expression is not named: {type(self)}')

    def has_resolved_name(self):
        return False

    def output_type(self):
        """Resolve the output type of the expression."""
        raise NotImplementedError(
            f"output_type not implemented for {type(self)}"
        )

    def flat_args(self):
        for arg in self.args:
            # print(type(arg))
            if isinstance(arg, tuple):
                yield from arg
            else:
                yield arg
            # if not isinstance(arg, Schema) and is_iterable(arg):
            #     print("flattened")
            #     yield from arg
            # else:
            #     print("not flattened")
            #     yield arg


@public
class ValueOp(Node):
    # columnar if any of the arguments is column-like
    output_shape = rlz.shape_like("args")

    # TODO(kszucs): we may want to enable this by default instead of
    # requiring to implement output_dtype abstract property
    # data type which all of the arguments are implicitly castable to
    # output_dtype = rlz.dtype_like("args")

    def root_tables(self):
        return distinct_roots(*self.exprs)

    @property
    @abstractmethod
    def output_dtype(self):
        ...

    # TODO(kszucs): use immutable_property instead
    @property
    def output_type(self):
        if self.output_shape is rlz.Shape.COLUMNAR:
            return self.output_dtype.column
        else:
            return self.output_dtype.scalar


@public
class Alias(ValueOp):
    arg = rlz.any
    name = rlz.instance_of((str, UnnamedMarker))

    output_shape = rlz.shape_like('arg')
    output_dtype = rlz.dtype_like('arg')

    def has_resolved_name(self):
        return True

    def resolve_name(self):
        return self.name


@public
class UnaryOp(ValueOp):
    """A unary operation."""

    arg = rlz.any

    # specialization of the parent class' rule to only check a single field
    output_shape = rlz.shape_like("arg")


@public
class BinaryOp(ValueOp):
    """A binary operation."""

    left = rlz.any
    right = rlz.any
