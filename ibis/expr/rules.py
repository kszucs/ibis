import enum
from itertools import product, starmap
from typing import Any

import toolz

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util as util
from ibis.common.validators import (  # noqa: F401
    immutable_property,
    instance_of,
    isin,
    list_of,
    map_to,
    one_of,
    optional,
    tuple_of,
    validator,
)


class Shape(enum.IntEnum):
    SCALAR = 0
    COLUMNAR = 1
    # TABULAR = 2


def highest_precedence_dtype(args):
    """Return the highest precedence type from the passed expressions

    Also verifies that there are valid implicit casts between any of the types
    and the selected highest precedence type.
    This is a thin wrapper around datatypes highest precedence check.

    Parameters
    ----------
    exprs : Iterable[ir.Value]
      A sequence of Expressions

    Returns
    -------
    dtype: DataType
      The highest precedence datatype
    """
    return dt.highest_precedence(arg.output_dtype for arg in args)


def castable(source, target):
    """Return whether source ir type is implicitly castable to target

    Based on the underlying datatypes and the value in case of Literals
    """
    value = getattr(source, 'value', None)
    return dt.castable(source.output_dtype, target.output_dtype, value=value)


def comparable(left, right):
    return castable(left, right) or castable(right, left)


class rule(validator):
    def _erase_expr(self, value):
        if isinstance(value, ir.Expr):
            return value.op()
        else:
            return value

    def __call__(self, *args, **kwargs):
        args = map(self._erase_expr, args)
        kwargs = toolz.valmap(self._erase_expr, kwargs)
        return super().__call__(*args, **kwargs)


# ---------------------------------------------------------------------
# Input type validators / coercion functions


# TODO(kszucs): deprecate then remove
@validator
def member_of(obj, arg, **kwargs):
    if isinstance(arg, ir.EnumValue):
        arg = arg.op().value
    if isinstance(arg, enum.Enum):
        enum.unique(obj)  # check that enum has unique values
        arg = arg.name

    if not hasattr(obj, arg):
        raise com.IbisTypeError(
            f'Value with type {type(arg)} is not a member of {obj}'
        )
    return getattr(obj, arg)


@validator
def value_list_of(inner, arg, **kwargs):
    # TODO(kszucs): would be nice to remove ops.ValueList
    # the main blocker is that some of the backends execution
    # model depends on the wrapper operation, for example
    # the dispatcher in pandas requires operation objects
    import ibis.expr.operations as ops

    values = tuple_of(inner, arg, **kwargs)
    return ops.ValueList(values).to_expr()


@validator
def sort_key(key, *, from_=None, this):
    import ibis.expr.operations as ops

    table = this[from_] if from_ is not None else None
    return ops.sortkeys._to_sort_key(key, table=table)


@rule
def datatype(arg, **kwargs):
    return dt.dtype(arg)


# TODO(kszucs): make type argument the first and mandatory, similarly to the
# value rule, move out the type inference to `ir.literal()` method
@rule
def literal(dtype, value, **kwargs):
    import ibis.expr.operations as ops

    if isinstance(value, ops.Literal):
        return value

    try:
        inferred_dtype = dt.infer(value)
    except com.InputTypeError:
        has_inferred = False
    else:
        has_inferred = True

    if dtype is None:
        has_explicit = False
    else:
        has_explicit = True
        explicit_dtype = dt.dtype(dtype)

    if has_explicit and has_inferred:
        try:
            # ensure type correctness: check that the inferred dtype is
            # implicitly castable to the explicitly given dtype and value
            dtype = inferred_dtype.cast(explicit_dtype, value=value)
        except com.IbisTypeError:
            raise TypeError(
                f'Value {value!r} cannot be safely coerced to {type}'
            )
    elif has_explicit:
        dtype = explicit_dtype
    elif has_inferred:
        dtype = inferred_dtype
    else:
        raise TypeError(
            'The datatype of value {!r} cannot be inferred, try '
            'passing it explicitly with the `type` keyword.'.format(value)
        )

    # if dtype is dt.null:
    #     return null().cast(dtype)
    # else:

    return ops.Literal(value, dtype=dtype)


@rule
def value(dtype, arg, **kwargs):
    """Validates that the given argument is a Value with a particular datatype

    Parameters
    ----------
    dtype : DataType subclass or DataType instance
    arg : python literal or an ibis expression
      If a python literal is given the validator tries to coerce it to an ibis
      literal.

    Returns
    -------
    arg : Value
      An ibis value expression with the specified datatype
    """
    import ibis.expr.operations as ops

    if not isinstance(arg, ops.Value):
        # coerce python literal to ibis literal
        arg = literal(dtype, arg)

    if dtype is None:
        # no datatype restriction
        return arg
    elif isinstance(dtype, type):
        # dtype class has been specified like dt.Interval or dt.Decimal
        if not issubclass(dtype, dt.DataType):
            raise com.IbisTypeError(
                f"Datatype specification {dtype} is not a subclass dt.DataType"
            )
        elif isinstance(arg.output_dtype, dtype):
            return arg
        else:
            raise com.IbisTypeError(
                f'Given argument with datatype {arg.output_dtype} is not '
                f'subtype of {dtype}'
            )
    elif isinstance(dtype, (dt.DataType, str)):
        # dtype instance or string has been specified and arg's dtype is
        # implicitly castable to it, like dt.int8 is castable to dt.int64
        dtype = dt.dtype(dtype)
        # retrieve literal values for implicit cast check
        value = getattr(arg, 'value', None)
        if dt.castable(arg.output_dtype, dtype, value=value):
            return arg
        else:
            raise com.IbisTypeError(
                f'Given argument with datatype {arg.output_dtype} is not '
                f'implicitly castable to {dtype}'
            )
    else:
        raise com.IbisTypeError(f'Invalid datatype specification {dtype}')


@validator
def scalar(inner, arg, **kwargs):
    return instance_of(ir.Scalar, inner(arg, **kwargs))


@validator
def column(inner, arg, **kwargs):
    return instance_of(ir.Column, inner(arg, **kwargs))


any = value(None)
# TODO(kszucs): or it should rather be
# any = value(dt.DataType)
double = value(dt.double)
string = value(dt.string)
boolean = value(dt.boolean)
integer = value(dt.int64)
decimal = value(dt.Decimal)
floating = value(dt.float64)
date = value(dt.date)
time = value(dt.time)
timestamp = value(dt.Timestamp)
category = value(dt.category)
temporal = one_of([timestamp, date, time])

strict_numeric = one_of([integer, floating, decimal])
soft_numeric = one_of([integer, floating, decimal, boolean])
numeric = soft_numeric

set_ = value(dt.Set)
array = value(dt.Array)
struct = value(dt.Struct)
mapping = value(dt.Map)

geospatial = value(dt.GeoSpatial)
point = value(dt.Point)
linestring = value(dt.LineString)
polygon = value(dt.Polygon)
multilinestring = value(dt.MultiLineString)
multipoint = value(dt.MultiPoint)
multipolygon = value(dt.MultiPolygon)


@validator
def interval(arg, units=None, **kwargs):
    arg = value(dt.Interval, arg)
    unit = arg.output_dtype.unit
    if units is not None and unit not in units:
        msg = 'Interval unit `{}` is not among the allowed ones {}'
        raise com.IbisTypeError(msg.format(unit, units))
    return arg


@validator
def client(arg, **kwargs):
    from ibis.backends.base import BaseBackend

    return instance_of(BaseBackend, arg)


# ---------------------------------------------------------------------
# Ouput type functions


def dtype_like(name):
    @immutable_property
    def output_dtype(self):
        arg = getattr(self, name)
        if util.is_iterable(arg):
            return highest_precedence_dtype(arg)
        else:
            return arg.output_dtype

    return output_dtype


def shape_like(name):
    @immutable_property
    def output_shape(self):
        arg = getattr(self, name)
        if util.is_iterable(arg):
            for op in arg:
                try:
                    if op.output_shape is Shape.COLUMNAR:
                        return Shape.COLUMNAR
                except AttributeError:
                    continue
            return Shape.SCALAR
        else:
            return arg.output_shape

    return output_shape


# TODO(kszucs): might just use bounds instead of actual literal values
# that could simplify interval binop output_type methods
# TODO(kszucs): pre-generate mapping?
def _promote_numeric_binop(exprs, op):
    bounds, dtypes = [], []
    for arg in exprs:
        dtypes.append(arg.type())
        if hasattr(arg.op(), 'value'):
            # arg.op() is a literal
            bounds.append([arg.op().value])
        else:
            bounds.append(arg.type().bounds)

    # In some cases, the bounding type might be int8, even though neither
    # of the types are that small. We want to ensure the containing type is
    # _at least_ as large as the smallest type in the expression.
    values = starmap(op, product(*bounds))
    dtypes += [dt.infer(value) for value in values]

    return dt.highest_precedence(dtypes)


def numeric_like(name, op):
    @immutable_property
    def output_dtype(self):
        args = getattr(self, name)
        if util.all_of(args, ir.IntegerValue):
            result = _promote_numeric_binop(args, op)
        else:
            result = highest_precedence_dtype(args)

        return result

    return output_dtype


@validator
def table(arg, *, schema=None, **kwargs):
    """A table argument.

    Parameters
    ----------
    schema : Union[sch.Schema, List[Tuple[str, dt.DataType], None]
        A validator for the table's columns. Only column subset validators are
        currently supported. Accepts any arguments that `sch.schema` accepts.
        See the example for usage.
    arg : The validatable argument.

    Examples
    --------
    The following op will accept an argument named ``'table'``. Note that the
    ``schema`` argument specifies rules for columns that are required to be in
    the table: ``time``, ``group`` and ``value1``. These must match the types
    specified in the column rules. Column ``value2`` is optional, but if
    present it must be of the specified type. The table may have extra columns
    not specified in the schema.
    """
    if not isinstance(arg, ir.Table):
        raise com.IbisTypeError(
            f'Argument is not a table; got type {type(arg).__name__}'
        )

    if schema is not None:
        if arg.schema() >= sch.schema(schema):
            return arg

        raise com.IbisTypeError(
            f'Argument is not a table with column subset of {schema}'
        )
    return arg


@validator
def column_from(name, column, *, this):
    """A column from a named table.

    This validator accepts columns passed as string, integer, or column
    expression. In the case of a column expression, this validator
    checks if the column in the table is equal to the column being
    passed.
    """
    if name not in this:
        raise com.IbisTypeError(f"Could not get table {name} from {this}")
    table = this[name]

    if isinstance(column, (str, int)):
        return table[column]
    elif isinstance(column, ir.Column):
        if not column.has_name():
            raise com.IbisTypeError(f"Passed column {column} has no name")

        maybe_column = column.get_name()
        try:
            if column.equals(table[maybe_column]):
                return column
            else:
                raise com.IbisTypeError(
                    f"Passed column is not a column in {type(table)}"
                )
        except com.IbisError:
            raise com.IbisTypeError(
                f"Cannot get column {maybe_column} from {type(table)}"
            )

    raise com.IbisTypeError(
        "value must be an int or str or Column, got "
        f"{type(column).__name__}"
    )


@validator
def base_table_of(name, *, this):
    from ibis.expr.analysis import find_first_base_table

    arg = this[name]
    base = find_first_base_table(arg)
    if base is None:
        raise com.IbisTypeError(f"`{arg}` doesn't have a base table")
    else:
        return base.to_expr()


@validator
def function_of(
    arg,
    fn,
    *,
    output_rule=any,
    this,
):
    if not util.is_function(fn):
        raise com.IbisTypeError(
            'argument `fn` must be a function, lambda or deferred operation'
        )

    if isinstance(arg, str):
        arg = this[arg]
    elif callable(arg):
        arg = arg(this=this)

    return output_rule(fn(arg), this=this)


@validator
def reduction(argument, **kwargs):
    from ibis.expr.analysis import is_reduction

    if not is_reduction(argument):
        raise com.IbisTypeError("`argument` must be a reduction")

    return argument


@validator
def non_negative_integer(arg, **kwargs):
    if not isinstance(arg, int):
        raise com.IbisTypeError(
            f"positive integer must be int type, got {type(arg).__name__}"
        )
    if arg < 0:
        raise ValueError("got negative value for non-negative integer rule")
    return arg


@validator
def python_literal(value, arg, **kwargs):
    if (
        not isinstance(arg, type(value))
        or not isinstance(value, type(arg))
        or arg != value
    ):
        raise ValueError(
            "arg must be a literal exactly equal in type and value to value "
            f"{value} with type {type(value)}, got `arg` with type {type(arg)}"
        )
    return arg


@validator
def pair(inner_left, inner_right, a, b, **kwargs):
    return inner_left(a, **kwargs), inner_right(b, **kwargs)


@validator
def analytic(arg, **kwargs):
    from ibis.expr.analysis import is_analytic

    if not is_analytic(arg):
        raise com.IbisInputError(
            'Expression does not contain a valid window operation'
        )
    return arg


@validator
def window(win, *, from_base_table_of, this):
    from ibis.expr.analysis import find_first_base_table
    from ibis.expr.window import Window

    if not isinstance(win, Window):
        raise com.IbisTypeError(
            "`win` argument should be of type `ibis.expr.window.Window`; "
            f"got type {type(win).__name__}"
        )

    table = find_first_base_table(this[from_base_table_of])
    if table is not None:
        win = win.bind(table.to_expr())

    if win.max_lookback is not None:
        error_msg = (
            "'max lookback' windows must be ordered " "by a timestamp column"
        )
        if len(win._order_by) != 1:
            raise com.IbisInputError(error_msg)
        order_var = win._order_by[0].op().args[0]
        if not isinstance(order_var.type(), dt.Timestamp):
            raise com.IbisInputError(error_msg)
    return win
