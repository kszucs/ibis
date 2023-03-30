from __future__ import annotations

import abc
import datetime
import decimal
import enum
import ipaddress
import itertools
import uuid
from typing import Literal as In
from typing import Optional, TypeVar, Any

import numpy as np
from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common import exceptions as com
from ibis.common.annotations import attribute
from ibis.common.collections import frozendict
from ibis.common.exceptions import IbisInputError, IbisTypeError
from ibis.common.grounds import Singleton
from ibis.common.patterns import coerce
from ibis.common.typing import Coercible, CoercionError
from ibis.expr.operations.core import Columnar, DataShape, Named, Scalar, Unary, Value


@public
class TableColumn(Value, Named):
    """Selects a column from a `Table`."""

    table = rlz.table
    name: str | int

    output_shape = rlz.Shape.COLUMNAR

    def __init__(self, table, name):
        if isinstance(name, int):
            name = table.schema.name_at_position(name)

        if name not in table.schema:
            columns_formatted = ', '.join(map(repr, table.schema.names))
            raise com.IbisTypeError(
                f"Column {name!r} is not found in table. "
                f"Existing columns: {columns_formatted}."
            )

        super().__init__(table=table, name=name)

    @property
    def output_dtype(self):
        return self.table.schema[self.name]


@public
class RowID(Value, Named):
    """The row number (an autonumeric) of the returned result."""

    name = "rowid"
    table = rlz.table
    output_shape = rlz.Shape.COLUMNAR
    output_dtype = dt.int64


@public
class TableArrayView(Value, Named):
    """Helper operation class for creating scalar subqueries."""

    table = rlz.table

    output_shape = rlz.Shape.COLUMNAR

    @property
    def output_dtype(self):
        return self.table.schema[self.name]

    @property
    def name(self):
        return self.table.schema.names[0]


@public
class Cast(Value):
    """Explicitly cast value to a specific data type."""

    arg: Value
    to: dt.DataType

    output_shape = rlz.shape_like("arg")

    @property
    def name(self):
        return f"{self.__class__.__name__}({self.arg.name}, {self.to})"

    @property
    def output_dtype(self):
        return self.to


@public
class TypeOf(Unary):
    output_dtype = dt.string


@public
class IsNull(Unary):
    """Return true if values are null.

    Returns
    -------
    ir.BooleanValue
        Value expression indicating whether values are null
    """

    output_dtype = dt.boolean


@public
class NotNull(Unary):
    """Returns true if values are not null.

    Returns
    -------
    ir.BooleanValue
        Value expression indicating whether values are not null
    """

    output_dtype = dt.boolean


@public
class ZeroIfNull(Unary):
    output_dtype = rlz.dtype_like("arg")


@public
class IfNull(Value):
    """Set values to ifnull_expr if they are equal to NULL."""

    arg: Value
    ifnull_expr: Value
    output_dtype = rlz.dtype_like("args")
    output_shape = rlz.shape_like("args")


@public
class NullIf(Value):
    """Set values to NULL if they equal the null_if_expr."""

    arg: Value
    null_if_expr: Value
    output_dtype = rlz.dtype_like("args")
    output_shape = rlz.shape_like("args")


@public
class Coalesce(Value):
    arg: tuple[Value, ...]
    output_shape = rlz.shape_like('arg')
    output_dtype = rlz.dtype_like('arg')


@public
class Greatest(Value):
    arg: tuple[Value, ...]
    output_shape = rlz.shape_like('arg')
    output_dtype = rlz.dtype_like('arg')


@public
class Least(Value):
    arg: tuple[Value, ...]
    output_shape = rlz.shape_like('arg')
    output_dtype = rlz.dtype_like('arg')


T = TypeVar("T", bound=dt.DataType)
S = TypeVar("S", bound=DataShape)


@public
class Literal(Value[T, Scalar], Coercible):
    __valid_input_types__ = (
        bytes,
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
        enum.Enum,
        float,
        frozenset,
        int,
        ipaddress.IPv4Address,
        ipaddress.IPv6Address,
        frozendict,
        np.generic,
        np.ndarray,
        str,
        tuple,
        type(None),
        uuid.UUID,
        decimal.Decimal,
    )
    value = rlz.one_of(
        (
            rlz.instance_of(__valid_input_types__),
            rlz.lazy_instance_of("shapely.geometry.BaseGeometry"),
        )
    )
    dtype: dt.DataType

    # TODO(kszucs): it should be named actually

    output_shape = rlz.Shape.SCALAR

    # TODO(kszucs): support ... as a placeholder for None
    # call dtype => T to indicate typevar bound
    # should support only dtype classes not instances
    @classmethod
    def __coerce__(cls, value, T=...):
        if isinstance(value, cls):
            return coerce(value, Value[T, ...])

        try:
            inferred_dtype = dt.infer(value)
        except (IbisInputError, IbisTypeError):
            # TODO(kszucs): this should be InferenceError
            has_inferred = False
        else:
            has_inferred = True

        if T is Ellipsis:  # or any
            has_explicit = False
        else:
            has_explicit = True
            explicit_dtype = dt.dtype(T)

        if has_explicit and has_inferred:
            # ensure type correctness: check that the inferred dtype is
            # implicitly castable to the explicitly given dtype and value
            # TODO(kszucs): this could be added to Dtype.__coerce__ then
            # T could be a coercer/matcher instead of a type
            if not dt.castable(inferred_dtype, explicit_dtype, value=value):
                raise CoercionError(
                    f"Value {value!r} cannot be safely coerced to `{explicit_dtype}`"
                )
            dtype = explicit_dtype
        elif has_explicit:
            dtype = explicit_dtype
        elif has_inferred:
            dtype = inferred_dtype
        else:
            raise CoercionError(
                f"The datatype of value {value!r} cannot be inferred, try "
                "passing it explicitly with the `type` keyword."
            )

        if dtype.is_null():
            return NullLiteral()

        value = dt.normalize(dtype, value)
        return Literal(value, dtype=dtype)

    @property
    def name(self):
        return repr(self.value)

    @property
    def output_dtype(self):
        return self.dtype


@public
class NullLiteral(Literal[dt.Null], Singleton):
    """Typeless NULL literal."""

    value: None = None
    dtype: Optional[dt.Null] = dt.null


@public
class ScalarParameter(Value, Named):
    _counter = itertools.count()

    dtype: dt.DataType
    counter = rlz.optional(
        rlz.instance_of(int), default=lambda: next(ScalarParameter._counter)
    )

    output_shape = rlz.Shape.SCALAR

    @property
    def name(self):
        return f'param_{self.counter:d}'

    @property
    def output_dtype(self):
        return self.dtype


@public
class Constant(Value, Singleton):
    output_shape = rlz.Shape.SCALAR


@public
class TimestampNow(Constant):
    output_dtype = dt.timestamp


@public
class RandomScalar(Constant):
    output_dtype = dt.float64


@public
class E(Constant):
    output_dtype = dt.float64


@public
class Pi(Constant):
    output_dtype = dt.float64


@public
class Hash(Value):
    arg: Value
    how: In['fnv', 'farm_fingerprint']

    output_dtype = dt.int64
    output_shape = rlz.shape_like("arg")


@public
class HashBytes(Value):
    arg = rlz.one_of({rlz.value(dt.string), rlz.value(dt.binary)})
    # arg: Value[dt.String, ...] | Value[dt.Binary, ...]
    # arg: Value[dt.String | dt.Binary, ...]
    how: In['md5', 'sha1', 'sha256', 'sha512']

    output_dtype = dt.binary
    output_shape = rlz.shape_like("arg")


# TODO(kszucs): we should merge the case operations by making the
# cases, results and default optional arguments like they are in
# api.py
@public
class SimpleCase(Value):
    base: Value
    cases: tuple[Value, ...]
    results: tuple[Value, ...]
    default: Value

    output_shape = rlz.shape_like("base")

    def __init__(self, cases, results, **kwargs):
        assert len(cases) == len(results)
        super().__init__(cases=cases, results=results, **kwargs)

    @attribute.default
    def output_dtype(self):
        values = [*self.results, self.default]
        return rlz.highest_precedence_dtype(values)


@public
class SearchedCase(Value):
    cases: tuple[Value[dt.Boolean, Any]]
    results: tuple[Value]
    default: Value

    def __init__(self, cases, results, default):
        assert len(cases) == len(results)
        super().__init__(cases=cases, results=results, default=default)

    @attribute.default
    def output_shape(self):
        # TODO(kszucs): can be removed after making Sequence iterable
        return rlz.highest_precedence_shape(self.cases)

    @attribute.default
    def output_dtype(self):
        exprs = [*self.results, self.default]
        return rlz.highest_precedence_dtype(exprs)


class _Negatable(abc.ABC):
    @abc.abstractmethod
    def negate(self):  # pragma: no cover
        ...
