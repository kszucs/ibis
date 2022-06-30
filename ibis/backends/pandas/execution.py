import functools
import math
import operator
from typing import Mapping

import numpy as np
import pandas as pd
from multipledispatch import Dispatcher

import ibis.expr.analysis as an
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.pandas.aggcontext import Summarize
from ibis.backends.pandas.client import PandasTable
from ibis.backends.pandas.operations import (
    PandasConcatenation,
    PandasFilter,
    PandasGroupby,
    PandasJoin,
    PandasProjection,
    PandasSelection,
    PandasSort,
    simplify,
)
from ibis.common.exceptions import IbisTypeError
from ibis.common.graph import Graph
from ibis.expr.rules import Shape

en = Dispatcher("execute_node")


@en.register(ops.Node)
def execute_node(node, **kwargs):
    raise IbisTypeError(node)


@en.register(ops.Alias)
def execute_alias(node, arg, name):
    if isinstance(arg, pd.Series):
        return arg.rename(name)
    else:
        return arg


@en.register(ops.DatabaseTable)
def execute_database_table(node, name, schema, source):
    # TODO(kszucs): port the timecontext handling from the original
    # implementation
    return source.dictionary[name]


@en.register(ops.TableColumn)
def execute_database_table(node, table, name):
    return table[name]


@en.register(PandasSelection)
def execute_pandas_projection(node, table, columns):
    return table[list(columns)]


@en.register(PandasProjection)
def execute_pandas_projection(node, values, reset_index=False):
    columns = []
    for value, result in zip(node.values, values):
        if not isinstance(result, pd.Series):
            result = pd.Series([result], name=value.resolve_name())
        columns.append(result)

    df = pd.concat(columns, axis=1)
    if reset_index:
        df = df.reset_index()

    return df


@en.register(PandasConcatenation)
def execute_pandas_projection(node, tables):
    return pd.concat(tables, axis=1).reset_index(drop=True)


@en.register(PandasFilter)
def execute_pandas_filter(node, table, predicate):
    return table.loc[predicate]


@en.register(PandasSort)
def execute_pandas_sort(node, table, fields, ascendings):
    return table.sort_values(list(fields), ascending=list(ascendings))


@en.register(PandasJoin)
def execute_pandas_join(node, left, right, how, left_on, right_on):
    return pd.merge(left, right, how=how, left_on=left_on, right_on=right_on)


@en.register(PandasGroupby)
def execute_pandas_groupby(node, table, by):
    return table.groupby(list(by))


REDUCTION_OPERATIONS = {
    ops.BitAnd: np.bitwise_and.reduce,
    ops.BitOr: np.bitwise_or.reduce,
    ops.BitXor: np.bitwise_xor.reduce,
    ops.CountDistinct: "nunique",
    ops.ApproxCountDistinct: "nunique",
    ops.Mean: "mean",
    ops.Max: "max",
    ops.Min: "min",
    ops.Sum: "sum",
    ops.Count: "count",
    ops.Variance: "var",
    ops.StandardDev: "std",
    ops.Any: "any",
    ops.All: "all",
}


@en.register(ops.Reduction)
def execute_reduction(node, arg, where=None):
    func = REDUCTION_OPERATIONS[type(node)]
    data = arg[where] if where is not None else arg

    aggctx = Summarize()
    result = aggctx.agg(data, func)

    return result


@en.register((ops.Variance, ops.StandardDev))
def execute_variance(node, arg, how, where=None):
    func = REDUCTION_OPERATIONS[type(node)]
    data = arg[where] if where is not None else arg
    ddof = {'pop': 0, 'sample': 1}

    aggctx = Summarize()
    return aggctx.agg(data, func, ddof=ddof[how])


@en.register(ops.Arbitrary)
def execute_arbitrary(node, arg, how, where):
    if how is None:
        how = 'first'

    if how not in {'first', 'last'}:
        raise com.OperationNotDefinedError(
            f"Arbitrary {how!r} is not supported"
        )

    aggctx = Summarize()
    return aggctx.agg(arg, how)


# @en.register((ops.Any, ops.All))
# def execute_any_all(node, arg):
#     func = REDUCTION_OPERATIONS[type(node)]

#     aggctx = Summarize()
#     return aggctx.agg(arg, func)
# if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
#     result = aggcontext.agg(data, type(op).__name__.lower())
# else:
#     result = aggcontext.agg(
#         data, lambda data: getattr(data, type(op).__name__.lower())()
#     )
# try:
#     return result.astype(bool)
# except TypeError:
#     return result


@en.register(ops.Distinct)
def execute_distinct(node, table):
    return table.drop_duplicates()


@en.register(ops.Literal)
def execute_literal(node, value, dtype):
    if isinstance(dtype, dt.Boolean):
        return bool(value)
    elif isinstance(dtype, dt.Integer):
        return int(value)
    else:
        return value


@en.register(ops.ValueList)
def execute_literal(node, values):
    return values


@en.register(ops.Cast)
def execute_cast(node, arg, to):
    if isinstance(arg, pd.Series):
        return arg.astype(to.to_pandas())
    else:
        return arg


@en.register(ops.Not)
def execute_not(node, arg):
    # TODO(kszucs): handle scalars
    if isinstance(arg, (bool, np.bool_)):
        return not arg
    elif isinstance(arg, pd.Series):
        return ~arg
    else:
        raise NotImplementedError(arg)


@en.register(ops.IsNan)
def execute_isnan(node, arg):
    return np.isnan(arg)


@en.register(ops.FillNa)
def execute_fillna(node, table, replacements):
    # TODO(kszucs): handle multiple replacement types
    if isinstance(node.replacements, Mapping):
        return table.fillna(dict(replacements))
    else:
        return table.fillna(replacements)


@en.register(ops.NullIf)
def execute_nullif(node, arg, null_if_expr):
    if isinstance(arg, pd.Series):
        return arg.where(arg != null_if_expr)
    elif isinstance(null_if_expr, pd.Series):
        raise NotImplementedError()
    else:
        return np.nan if arg == null_if_expr else arg


@en.register(ops.NullIfZero)
def execute_null_if_zero(op, arg):
    return arg.where(arg != 0, np.nan)


@en.register(ops.IfNull)
def execute_ifnull(node, arg, ifnull_expr):
    if isinstance(arg, pd.Series):
        return arg.fillna(ifnull_expr)
    elif isinstance(ifnull_expr, pd.Series):
        return (
            ifnull_expr
            if pd.isnull(arg)
            else pd.Series(arg, index=ifnull_expr.index)
        )
    else:
        return ifnull_expr if pd.isnull(arg) else arg


@en.register(ops.ZeroIfNull)
def execute_zero_if_null(node, arg):
    pandas_dtype = node.arg.output_dtype.to_pandas()
    pandas_zero = pandas_dtype.type(0)
    if isinstance(arg, pd.Series):
        return arg.replace(
            {np.nan: pandas_zero, None: pandas_zero, pd.NA: pandas_zero}
        )
    else:
        if arg is None or pd.isna(arg) or math.isnan(arg) or np.isnan(arg):
            return pandas_zero
        else:
            return arg


@en.register(ops.DropNa)
def execute_dropna(node, table, how, subset):
    subset = [col.name for col in subset] if subset else None
    return table.dropna(how=how, subset=subset)


@en.register(ops.Contains)
def execute_contains(node, value, options):
    return value.isin(options)


@en.register(ops.NotContains)
def execute_not_contains(node, value, options):
    return ~(value.isin(options))


@en.register(ops.Where)
def execute_where(node, bool_expr, true_expr, false_null_expr):
    # TODO(kszucs): handle the other cases too
    if isinstance(true_expr, pd.Series):
        return true_expr.where(bool_expr, other=false_null_expr)
    elif isinstance(bool_expr, pd.Series):
        return pd.Series(np.repeat(true_expr, len(bool_expr))).where(
            bool_expr, other=false_null_expr
        )
    else:
        return true_expr if bool_expr else false_null_expr


BINARY_OPERATIONS = {
    ops.Greater: operator.gt,
    ops.Less: operator.lt,
    ops.LessEqual: operator.le,
    ops.GreaterEqual: operator.ge,
    ops.Equals: operator.eq,
    ops.NotEquals: operator.ne,
    ops.And: operator.and_,
    ops.Or: operator.or_,
    ops.Xor: operator.xor,
    ops.Add: operator.add,
    ops.Subtract: operator.sub,
    ops.Multiply: operator.mul,
    ops.Divide: operator.truediv,
    ops.FloorDivide: operator.floordiv,
    ops.Modulus: operator.mod,
    ops.Power: operator.pow,
    ops.IdenticalTo: lambda x, y: (x == y) | (pd.isnull(x) & pd.isnull(y)),
}


@en.register(ops.Binary)
def execute_binary_op(node, left, right, **kwargs):
    try:
        op = BINARY_OPERATIONS[type(node)]
    except KeyError:
        raise NotImplementedError(
            f'Binary operation {node.__class__.__name__} not implemented'
        )
    else:
        return op(left, right)


STRING_OPERATIONS = {
    ops.Lowercase: "lower",
    ops.Uppercase: "upper",
    ops.Capitalize: "capitalize",
    ops.Strip: "strip",
    ops.LStrip: "lstrip",
    ops.RStrip: "rstrip",
}


@en.register(ops.StringUnary)
def execute_string_unary(node, arg):
    try:
        op = STRING_OPERATIONS[type(node)]
    except KeyError:
        raise NotImplementedError(
            f'Unary string operation {node.__class__.__name__} not implemented'
        )
    else:
        func = getattr(arg.str, op)
        return func()


@en.register(ops.StringLength)
def execute_string_length(node, arg):
    return arg.str.len().astype('int32')


@en.register(ops.LPad)
def execute_lpad(node, arg, length, pad):
    return arg.str.pad(length, side='left', fillchar=pad)


@en.register(ops.RPad)
def execute_rpad(node, arg, length, pad):
    return arg.str.pad(length, side='right', fillchar=pad)


@en.register(ops.Reverse)
def execute_string_reverse(node, arg):
    return arg.str[::-1]


@en.register(ops.StringAscii)
def execute_ascii(node, arg):
    return arg.map(ord).astype('int32')


@en.register(ops.StartsWith)
def execute_startswith(node, arg, start):
    return arg.str.startswith(start)


@en.register(ops.EndsWith)
def execute_endswith(node, arg, end):
    return arg.str.endswith(end)


@en.register(ops.StrRight)
def execute_str_right(node, arg, nchars):
    return arg.str[-nchars:]


@en.register(ops.Repeat)
def execute_string_repeat(node, arg, times):
    return arg.str.repeat(times)


@en.register(ops.StringJoin)
def execute_string_join(node, sep, arg):
    return functools.reduce(lambda x, y: x + sep + y, arg)


@en.register(ops.StringContains)
def execute_string_contains(node, haystack, needle):
    return haystack.str.contains(needle)


@en.register(ops.StringFind)
def execute_string_find(op, arg, substr, start, end):
    return arg.str.find(substr, start, end)


@en.register(ops.Substring)
def execute_substring(node, arg, start, length):
    return arg.str[start : start + length]


# @en.register(ops.StringSplit)
# def execute_split(node, arg, delimiter):
#     pass

NUMERIC_FUNCTIONS = {
    ops.Floor: math.floor,
    ops.Ln: math.log,
    ops.Log2: lambda x: math.log(x, 2),
    ops.Log10: math.log10,
    ops.Exp: math.exp,
    ops.Sqrt: math.sqrt,
    ops.Abs: abs,
    ops.Ceil: math.ceil,
    ops.Sign: lambda x: 0 if not x else -1 if x < 0 else 1,
    ops.Acos: np.arccos,
    ops.Asin: np.arcsin,
    ops.Atan: np.arctan,
    ops.Cos: np.cos,
    ops.Sin: np.sin,
    ops.Tan: np.tan,
    ops.Cot: lambda x: np.cos(x) / np.sin(x),
    ops.Radians: np.radians,
    ops.Degrees: np.degrees,
    ops.Ceil: np.ceil,
    ops.Floor: np.floor,
    ops.IsInf: np.isinf,
    ops.Greatest: np.maximum.reduce,
    ops.Least: np.minimum.reduce,
}

# def call_numpy_ufunc(func, op, data, **kwargs):
#     if getattr(data, "dtype", None) == np.dtype(np.object_):
#         return data.apply(functools.partial(execute_node, op, **kwargs))
#     return func(data)


@en.register(tuple(NUMERIC_FUNCTIONS.keys()))
def execute_unary_numeric(op, arg):
    return NUMERIC_FUNCTIONS[type(op)](arg)


@en.register(ops.Negate)
def execute_negate(op, arg):
    return -arg
    # if isinstance(arg, pd.Series):
    #     return call_numpy_ufunc(np.negative, op, arg)
    # else:
    #     return -data


@en.register(ops.Atan2)
def execute_atan2(node, left, right):
    return np.arctan2(left, right)


@en.register(ops.Clip)
def execute_clip(op, arg, lower, upper):
    return arg.clip(lower=lower, upper=upper)


# TEMPORAL


@en.register(ops.ExtractTemporalField)
def execute_extract_temporal_field(node, arg):
    field_name = type(node).__name__.lower().replace('extract', '')
    if isinstance(arg, pd.Series):
        if field_name == 'weekofyear':
            return arg.dt.isocalendar().week.astype(np.int32)
        return getattr(arg.dt, field_name).astype(np.int32)
    else:
        return getattr(arg, field_name)


def day_name(obj):
    """Backwards compatible name of day getting function.

    Parameters
    ----------
    obj : Union[Series, pd.Timestamp]

    Returns
    -------
    str
        The name of the day corresponding to `obj`
    """
    try:
        return obj.day_name()
    except AttributeError:
        return obj.weekday_name


@en.register(ops.DayOfWeekIndex)
def execute_day_of_week_index(op, arg):
    if isinstance(arg, pd.Series):
        return arg.dt.dayofweek.astype(np.int16)
    else:
        # TODO(kszucs) may not need pd.Timestamp here
        return pd.Timestamp(arg).dayofweek


@en.register(ops.DayOfWeekName)
def execute_day_of_week_name(op, arg):
    if isinstance(arg, pd.Series):
        return day_name(arg.dt)
    else:
        return day_name(pd.Timestamp(arg))


@en.register(ops.StructField)
def execute_struct_field(node, arg, field):
    if isinstance(arg, Mapping):
        return arg[field]
    else:
        return pd.NA


# @en.register(ops.Coalesce)
# def execute_string_length(node, arg):
#     return arg.str.len().astype('int32')


def execute(node, params, **kwargs):

    node = an.rewrite(simplify, node)

    g = Graph(node)

    # print()
    # print(f"============= PLAN for {node.__class__.__name__} ===============")
    # for e in g.toposort():
    #     print(type(e))

    results = g.map(en)

    return results[node]
