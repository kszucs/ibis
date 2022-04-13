from __future__ import annotations

from abc import abstractmethod

import toolz
from public import public

from ibis import util

from ...common.exceptions import ExpressionError
from ...common.grounds import Comparable
from ...common.validators import immutable_property
from ...util import UnnamedMarker, is_iterable
from .. import rules as rlz
from .. import types as ir
from ..rules import Shape
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


def _erase_exprs(arg):
    """
    Remove intermediate expressions.
    """
    if isinstance(arg, ir.Expr):
        return arg.op()
    elif isinstance(arg, tuple):
        return tuple(map(_erase_exprs, arg))
    else:
        return arg


def _create_exprs(arg):
    if isinstance(arg, Node):
        return arg.to_expr()
    elif isinstance(arg, tuple):
        return tuple(map(_create_exprs, arg))
    else:
        return arg


# TODO(kszucs): should rename to Operator
@public
class Node(Annotable, Comparable):

    __slots__ = ('_cached_expr', 'args')

    def __init__(self, **kwargs):
        object.__setattr__(self, 'args', tuple(kwargs.values()))
        kwargs = {k: _erase_exprs(v) for k, v in kwargs.items()}
        super().__init__(**kwargs)

    def __post_init__(self):
        for arg in self.__args__:
            assert not isinstance(arg, ir.Expr)
        for arg in self.args:
            assert not isinstance(arg, Node)
        expr = self.output_type(self)
        object.__setattr__(self, '_cached_expr', expr)

    @property
    def argnames(self):
        return self.__argnames__

    def __getattribute__(self, name):
        arg = super().__getattribute__(name)
        if name in type(self).__argnames__:
            arg = _create_exprs(arg)
        return arg

    def __reduce__(self):
        kwargs = dict(zip(self.argnames, self.args))
        return (self._reconstruct, (kwargs,))

    @immutable_property
    def _flat_ops(self):
        import ibis.expr.types as ir

        return tuple(
            arg.op() for arg in self.flat_args() if isinstance(arg, ir.Expr)
        )

    def __equals__(self, other):
        return self._hash == other._hash and self.__args__ == other.__args__

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
        return self._cached_expr

    def resolve_name(self):
        raise ExpressionError(f'Expression is not named: {type(self)}')

    def has_resolved_name(self):
        return False

    @property
    def output_type(self):
        """Resolve the output type of the expression."""
        raise NotImplementedError(
            f"output_type not implemented for {type(self)}"
        )

    def flat_args(self):
        for arg in self.args:
            if not isinstance(arg, Schema) and is_iterable(arg):
                yield from arg
            else:
                yield arg


@public
class ValueOp(Node):
    def root_tables(self):
        return distinct_roots(*self.exprs)

    @property
    @abstractmethod
    def output_dtype(self):
        """
        Ibis datatype of the produced value expression.

        Returns
        -------
        dt.DataType
        """

    @property
    @abstractmethod
    def output_shape(self):
        """
        Shape of the produced value expression.

        Possible values are: "scalar" and "columnar"

        Returns
        -------
        rlz.Shape
        """

    @property
    def output_type(self):
        if self.output_shape is Shape.COLUMNAR:
            return self.output_dtype.column
        else:
            return self.output_dtype.scalar


@public
class Alias(ValueOp):
    arg = rlz.any
    name = rlz.instance_of((str, UnnamedMarker))

    output_shape = rlz.shape_like("arg")
    output_dtype = rlz.dtype_like("arg")

    def has_resolved_name(self):
        return True

    def resolve_name(self):
        return self.name


@public
class UnaryOp(ValueOp):
    """A unary operation."""

    arg = rlz.any

    @property
    def output_shape(self):
        return self.arg.op().output_shape


@public
class BinaryOp(ValueOp):
    """A binary operation."""

    left = rlz.any
    right = rlz.any

    @property
    def output_shape(self):
        return max(self.left.op().output_shape, self.right.op().output_shape)
