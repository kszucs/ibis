from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis import util
from ibis.common.graph import Node as Traversable
from ibis.common.grounds import Concrete
from ibis.common.patterns import coerce
from ibis.common.typing import Coercible, CoercionError

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt


@public
class Node(Concrete, Traversable):
    def equals(self, other):
        if not isinstance(other, Node):
            raise TypeError(
                f"invalid equality comparison between Node and {type(other)}"
            )
        return self.__cached_equals__(other)

    @util.deprecated(as_of='4.0', instead='remove intermediate .op() calls')
    def op(self):
        """Make `Node` backwards compatible with code that uses `Expr.op()`."""
        return self

    @abstractmethod
    def to_expr(self):
        ...

    # Avoid custom repr for performance reasons
    __repr__ = object.__repr__

    def __rich_repr__(self):
        return zip(self.__argnames__, self.__args__)


@public
class Named(ABC):
    __slots__ = tuple()

    @property
    @abstractmethod
    def name(self):
        """Name of the operation.

        Returns
        -------
        str
        """


class DataShape:
    pass


class Scalar(DataShape):
    pass


class Columnar(DataShape):
    pass


T = TypeVar("T", bound=dt.DataType)
S = TypeVar("S", bound=DataShape)  # or rather as constraints


@public
class Value(Node, Named, Coercible, Generic[T, S]):
    @classmethod
    def __coerce__(cls, value, T=..., S=...):
        # should move the Literal.__coerce__ implementation here
        from ibis.expr.operations import Literal
        from ibis.expr.types import Expr

        if isinstance(value, Expr):
            value = value.op()

        if not isinstance(value, Value):
            value = coerce(value, Literal[T])

        if T is not Ellipsis:
            dtype = dt.dtype(T)
            # retrieve literal values for implicit cast check
            pyval = getattr(value, 'value', None)
            if not dt.castable(value.output_dtype, dtype, value=pyval):
                raise CoercionError(
                    f'Given argument with datatype {value.output_dtype} is not '
                    f'implicitly castable to {dtype}'
                )

        if S is not Ellipsis:
            if not issubclass(S, DataShape):
                raise TypeError(
                    f"Shape specification {S} is not a subclass of DataShape"
                )
            if value.output_shape is not S:
                raise CoercionError(
                    f"Given argument with shape {value.output_shape} is not {S}"
                )

        return value

    # TODO(kszucs): cover it with tests
    # TODO(kszucs): figure out how to represent not named arguments
    @property
    def name(self):
        args = ", ".join(arg.name for arg in self.__args__ if isinstance(arg, Named))
        return f"{self.__class__.__name__}({args})"

    @property
    @abstractmethod
    def output_dtype(self) -> dt.DataType:
        """Ibis datatype of the produced value expression.

        Returns
        -------
        dt.DataType
        """

    @property
    @abstractmethod
    def output_shape(self) -> rlz.Shape:
        """Shape of the produced value expression.

        Possible values are: "scalar" and "columnar"

        Returns
        -------
        rlz.Shape
        """

    def to_expr(self):
        import ibis.expr.types as ir

        if self.output_shape.is_columnar():
            typename = self.output_dtype.column
        else:
            typename = self.output_dtype.scalar

        return getattr(ir, typename)(self)


@public
class Alias(Value):
    arg: Value
    name: str

    output_shape = rlz.shape_like("arg")
    output_dtype = rlz.dtype_like("arg")


@public
class Unary(Value):
    """A unary operation."""

    arg: Value

    @property
    def output_shape(self):
        return self.arg.output_shape


@public
class Binary(Value):
    """A binary operation."""

    left: Value
    right: Value

    @property
    def output_shape(self):
        return max(self.left.output_shape, self.right.output_shape)


@public
class Argument(Value):
    name: str
    shape = rlz.instance_of(rlz.Shape)
    dtype: dt.DataType

    @property
    def output_dtype(self) -> dt.DataType:
        return self.dtype

    @property
    def output_shape(self) -> rlz.Shape:
        return self.shape


public(ValueOp=Value, UnaryOp=Unary, BinaryOp=Binary)
