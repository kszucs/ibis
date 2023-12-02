from __future__ import annotations

import operator
from collections.abc import Sized
from functools import reduce, singledispatch

import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis import util
from ibis.backends.pandas.newutils import asframe, asseries, columnwise
from ibis.backends.pandas.rewrites import (
    ColumnRef,
    PandasAggregate,
    PandasJoin,
    PandasReduce,
    PandasRename,
)
from ibis.common.exceptions import OperationNotDefinedError
from ibis.formats.pandas import PandasData, PandasSchema, PandasType

################## PANDAS SPECIFIC NODES ######################

# class PandasRelation(ops.Relation):
#     pass


# class SelectColumns(PandasRelation):
#     fields


@singledispatch
def execute(node, **kwargs):
    raise OperationNotDefinedError(f"no rule for {type(node)}")


@execute.register(ops.Literal)
def execute_literal(op, value, dtype):
    if dtype.is_interval():
        value = pd.Timedelta(value, dtype.unit.short)
    elif dtype.is_array():
        value = np.array(value)
    elif dtype.is_date():
        value = pd.Timestamp(value, tz="UTC").tz_localize(None)

    return value


@execute.register(ops.DatabaseTable)
def execute_database_table(op, name, schema, source, namespace):
    return source.dictionary[name]
    # if timecontext:
    #     begin, end = timecontext
    #     time_col = get_time_col()
    #     if time_col not in df:
    #         raise com.IbisError(
    #             f"Table {op.name} must have a time column named {time_col}"
    #             " to execute with time context."
    #         )
    #     # filter with time context
    #     mask = df[time_col].between(begin, end)
    #     return df.loc[mask].reset_index(drop=True)


@execute.register(ops.InMemoryTable)
def execute_in_memory_table(op, name, schema, data):
    return data.to_frame()


@execute.register(ops.DummyTable)
def execute_dummy_table(op, values):
    df, _ = asframe(values)
    return df


@execute.register(ops.Limit)
def execute_limit(op, parent, n, offset):
    if n is None:
        return parent.iloc[offset:]
    else:
        return parent.iloc[offset : offset + n]


@execute.register(ops.Sample)
def execute_sample(op, parent, fraction, method, seed):
    return parent.sample(frac=fraction, random_state=seed)


@execute.register(ops.Filter)
def execute_filter(op, parent, predicates):
    if predicates:
        pred = reduce(operator.and_, predicates)
        if len(pred) != len(parent):
            raise RuntimeError(
                "Selection predicate length does not match underlying table"
            )
        parent = parent.loc[pred].reset_index(drop=True)
    return parent


@execute.register(PandasRename)
def execute_rename(op, parent, mapping):
    return parent.rename(columns=mapping)


@execute.register(PandasJoin)
def execute_join(op, left, right, left_on, right_on, how):
    # broadcase predicates if they are scalar values
    left_size = len(left)
    left_on = [asseries(v, left_size) for v in left_on]
    right_size = len(right)
    right_on = [asseries(v, right_size) for v in right_on]

    if how == "cross":
        assert not left_on and not right_on
        return pd.merge(left, right, how="cross")
    else:
        df = left.merge(right, how=how, left_on=left_on, right_on=right_on)
        # drop the join key columns
        return df.drop(columns=["key_0"])


@execute.register(ops.Union)
def execute_union(op, left, right, distinct):
    result = pd.concat([left, right], axis=0)
    return result.drop_duplicates() if distinct else result


@execute.register(ops.Intersection)
def execute_intersection_dataframe_dataframe(op, left, right, distinct):
    if not distinct:
        raise NotImplementedError(
            "`distinct=False` is not supported by the pandas backend"
        )
    return left.merge(right, on=list(left.columns), how="inner")


@execute.register(ops.Distinct)
def execute_distinct(op, parent):
    return parent.drop_duplicates()


@execute.register(ops.Project)
def execute_project(op, parent, values):
    df, _ = asframe(values)
    return df


@execute.register(ops.Sort)
def execute_sort(op, parent, keys):
    # 1. add sort key columns to the dataframe if they are not already present
    # 2. sort the dataframe using those columns
    # 3. drop the sort key columns

    names = []
    newcols = {}
    for key, keycol in zip(op.keys, keys):
        if not isinstance(key, ops.Field):
            name = util.gen_name("sort_key")
            newcols[name] = keycol
        names.append(name)

    result = parent.assign(**newcols)
    ascending = [key.ascending for key in op.keys]
    result = result.sort_values(by=names, ascending=ascending, ignore_index=True)

    return result.drop(newcols.keys(), axis=1)


@execute.register(ColumnRef)
def execute_column_ref(op, name, dtype):
    return name


@execute.register(ops.Field)
def execute_field(op, rel, name):
    return rel[name]


@execute.register(ops.Alias)
def execute_alias(op, arg, name):
    try:
        return arg.rename(name)
    except AttributeError:
        return arg


@execute.register(ops.SortKey)
def execute_sort_key(op, expr, ascending):
    return expr


@execute.register(ops.Not)
def execute_not(op, arg):
    if isinstance(arg, (bool, np.bool_)):
        return not arg
    else:
        return ~arg


@execute.register(ops.Negate)
def execute_negate(op, arg):
    if isinstance(arg, (bool, np.bool_)):
        return not arg
    else:
        return -arg


@execute.register(ops.Cast)
def execute_cast(op, arg, to):
    if isinstance(arg, pd.Series):
        return PandasData.convert_column(arg, to)
    else:
        return PandasData.convert_scalar(arg, to)


_unary_operations = {
    ops.Abs: abs,
    ops.Ceil: lambda x: np.ceil(x).astype("int64"),
    ops.Floor: lambda x: np.floor(x).astype("int64"),
    ops.Sqrt: np.sqrt,
    ops.Sign: np.sign,
    ops.Log2: np.log2,
    ops.Log10: np.log10,
    ops.Ln: np.log,
    ops.Exp: np.exp,
    ops.Tan: np.tan,
    ops.Cos: np.cos,
    ops.Cot: lambda x: 1 / np.tan(x),
    ops.Sin: np.sin,
    ops.Atan: np.arctan,
    ops.Acos: np.arccos,
    ops.Asin: np.arcsin,
    ops.BitwiseNot: np.invert,
    ops.Radians: np.radians,
    ops.Degrees: np.degrees,
}

_binary_operations = {
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
    ops.BitwiseXor: lambda x, y: np.bitwise_xor(x, y),
    ops.BitwiseOr: lambda x, y: np.bitwise_or(x, y),
    ops.BitwiseAnd: lambda x, y: np.bitwise_and(x, y),
    ops.BitwiseLeftShift: lambda x, y: np.left_shift(x, y).astype("int64"),
    ops.BitwiseRightShift: lambda x, y: np.right_shift(x, y).astype("int64"),
    ops.Atan2: np.arctan2,
}


@execute.register(ops.Unary)
def execute_unary(op, arg):
    return _unary_operations[type(op)](arg)


@execute.register(ops.Binary)
def execute_equals(op, left, right):
    return _binary_operations[type(op)](left, right)


@execute.register(ops.Log)
def execute_log(op, arg, base):
    if base is None:
        return np.log(arg)
    else:
        return np.log(arg) / np.log(base)


@execute.register(ops.Round)
def execute_round(op, arg, digits):
    if digits is None:
        return np.round(arg).astype("int64")
    else:
        return np.round(arg, digits).astype("float64")


@execute.register(ops.Clip)
def execute_clip(op, **kwargs):
    return columnwise(
        lambda df: df["arg"].clip(lower=df["lower"], upper=df["upper"]), kwargs
    )


@execute.register(ops.IfElse)
def execute_if_else(op, **kwargs):
    return columnwise(
        lambda df: df["true_expr"].where(df["bool_expr"], other=df["false_null_expr"]),
        kwargs,
    )


@execute.register(ops.TypeOf)
def execute_typeof(op, arg):
    raise OperationNotDefinedError("TypeOf is not implemented")


@execute.register(ops.NullIf)
def execute_null_if(op, **kwargs):
    return columnwise(
        lambda df: df["arg"].where(df["arg"] != df["null_if_expr"]), kwargs
    )


@execute.register(ops.IsNull)
def execute_series_isnull(op, arg):
    return arg.isnull()


@execute.register(ops.NotNull)
def execute_series_notnnull(op, arg):
    return arg.notnull()


@execute.register(ops.FillNa)
def execute_fillna(op, parent, replacements):
    return parent.fillna(replacements)


@execute.register(ops.IsNan)
def execute_isnan(op, arg):
    try:
        return np.isnan(arg)
    except (TypeError, ValueError):
        # if `arg` contains `None` np.isnan will complain
        # so we take advantage of NaN not equaling itself
        # to do the correct thing
        return arg != arg


@execute.register(ops.IsInf)
def execute_isinf(op, arg):
    return np.isinf(arg)


@execute.register(ops.DropNa)
def execute_dropna(op, parent, how, subset):
    if op.subset is not None:
        subset = [col.name for col in op.subset]
    else:
        subset = None
    return parent.dropna(how=how, subset=subset)


# these are serieswise functions
_reduction_functions = {
    ops.Min: lambda x: x.min(),
    ops.Max: lambda x: x.max(),
    ops.Sum: lambda x: x.sum(),
    ops.Mean: lambda x: x.mean(),
    ops.Count: lambda x: x.count(),
    ops.Mode: lambda x: x.mode().iloc[0],
    ops.Any: lambda x: x.any(),
    ops.All: lambda x: x.all(),
    ops.Median: lambda x: x.median(),
    ops.ApproxMedian: lambda x: x.median(),
    ops.BitAnd: lambda x: np.bitwise_and.reduce(x.values),
    ops.BitOr: lambda x: np.bitwise_or.reduce(x.values),
    ops.BitXor: lambda x: np.bitwise_xor.reduce(x.values),
    ops.Last: lambda x: x.iloc[-1],
    ops.First: lambda x: x.iloc[0],
    ops.CountDistinct: lambda x: x.nunique(),
    ops.ApproxCountDistinct: lambda x: x.nunique(),
}


# could columnwise be used here?
def agg(func, arg_column, where_column):
    if where_column is None:

        def applier(df):
            return func(df[arg_column])
    else:

        def applier(df):
            mask = df[where_column]
            col = df[arg_column][mask]
            return func(col)

    return applier


@execute.register(ops.Reduction)
def execute_reduction(op, arg, where):
    func = _reduction_functions[type(op)]
    return agg(func, arg, where)


@execute.register(PandasReduce)
def execute_pandas_reduce(op, parent, metrics):
    results = {k: v(parent) for k, v in metrics.items()}
    combined, _ = asframe(results)
    return combined


@execute.register(PandasAggregate)
def execute_pandas_aggregate(op, parent, groups, metrics):
    groupby = parent.groupby(list(groups))
    metrics = {k: groupby.apply(v) for k, v in metrics.items()}

    result = pd.concat(metrics, axis=1).reset_index()
    return result


variance_ddof = {"pop": 0, "sample": 1}


@execute.register(ops.Variance)
def execute_variance(op, arg, where, how):
    ddof = variance_ddof[how]
    return agg(lambda x: x.var(ddof=ddof), arg, where)


@execute.register(ops.StandardDev)
def execute_standard_dev(op, arg, where, how):
    ddof = variance_ddof[how]
    return agg(lambda x: x.std(ddof=ddof), arg, where)


@execute.register(ops.Correlation)
def execute_correlation(op, left, right, where, how):
    if where is None:

        def agg(df):
            return df[left].corr(df[right])
    else:

        def agg(df):
            mask = df[where]
            lhs = df[left][mask]
            rhs = df[right][mask]
            return lhs.corr(rhs)

    return agg


@execute.register(ops.Covariance)
def execute_covariance(op, left, right, where, how):
    ddof = variance_ddof[how]
    if where is None:

        def agg(df):
            return df[left].cov(df[right], ddof=ddof)
    else:

        def agg(df):
            mask = df[where]
            lhs = df[left][mask]
            rhs = df[right][mask]
            return lhs.cov(rhs, ddof=ddof)

    return agg


@execute.register(ops.ArgMin)
@execute.register(ops.ArgMax)
def execute_argminmax(op, arg, key, where):
    func = operator.methodcaller(op.__class__.__name__.lower())

    if where is None:

        def agg(df):
            indices = func(df[key])
            return df[arg].iloc[indices]
    else:

        def agg(df):
            mask = df[where]
            filtered = df[mask]
            indices = func(filtered[key])
            return filtered[arg].iloc[indices]

    return agg


@execute.register(ops.ArrayCollect)
def execute_array_collect(op, arg, where):
    if where is None:

        def agg(df):
            return df[arg].tolist()
    else:

        def agg(df):
            mask = df[where]
            return df[arg][mask].tolist()

    return agg


@execute.register(ops.GroupConcat)
def execute_group_concat(op, arg, sep, where):
    if where is None:

        def agg(df):
            return sep.join(df[arg].astype(str))
    else:

        def agg(df):
            mask = df[where]
            group = df[arg][mask]
            if group.empty:
                return pd.NA
            return sep.join(group)

    return agg


@execute.register(ops.Quantile)
@execute.register(ops.MultiQuantile)
def execute_quantile(op, arg, quantile, where):
    return agg(lambda x: x.quantile(quantile), arg, where)


@execute.register(ops.Arbitrary)
def execute_arbitrary(op, arg, where, how):
    # TODO(kszucs): could be rewritten to ops.Last and ops.First prior to execution
    if how == "first":
        return agg(lambda x: x.iloc[0], arg, where)
    elif how == "last":
        return agg(lambda x: x.iloc[-1], arg, where)
    else:
        raise OperationNotDefinedError(f"Arbitrary {how!r} is not supported")


@execute.register(ops.CountStar)
def execute_count_star(op, arg, where):
    # TODO(kszucs): revisit the arg handling here
    def agg(df):
        if where is None:
            return len(df)
        else:
            return df[where].sum()

    return agg


@execute.register(ops.CountDistinctStar)
def execute_count_distinct_star(op, arg, where):
    def agg(df):
        if where is None:
            return df.nunique()
        else:
            return df[where].nunique()

    return agg


@execute.register(ops.InValues)
def execute_in_values(op, value, options):
    if isinstance(value, pd.Series):
        return value.isin(options)
    # elif isinstance(value, SeriesGroupBy):
    #     return data.obj.isin(elements).groupby(
    #         get_grouping(data.grouper.groupings), group_keys=False
    #     )
    else:
        return value in options


@execute.register(ops.InSubquery)
def execute_in_subquery(op, rel, needle):
    first_column = rel.iloc[:, 0]
    if isinstance(needle, pd.Series):
        return needle.isin(first_column)
    # elif isinstance(needle, SeriesGroupBy):
    #     return data.obj.isin(elements).groupby(
    #         get_grouping(data.grouper.groupings), group_keys=False
    #     )
    else:
        return needle in first_column


@execute.register(ops.Date)
def execute_date(op, arg):
    return arg.dt.floor("d")


@execute.register(ops.TimestampNow)
def execute_timestamp_now(op, *args, **kwargs):
    # timecontext = kwargs.get("timecontext", None)
    return pd.Timestamp("now", tz="UTC").tz_localize(None)


@execute.register(ops.TimestampDiff)
def execute_timestamp_diff(op, left, right):
    return left - right


@execute.register(ops.Greatest)
def execute_greatest(op, arg):
    return columnwise(lambda df: df.max(axis=1), arg)


@execute.register(ops.Least)
def execute_least(op, arg):
    return columnwise(lambda df: df.min(axis=1), arg)


@execute.register(ops.Coalesce)
def execute_coalesce(op, arg):
    return columnwise(lambda df: df.bfill(axis=1).iloc[:, 0], arg)


@execute.register(ops.Between)
def execute_between(op, arg, lower_bound, upper_bound):
    return arg.between(lower_bound, upper_bound)


def zuper(node, params):
    from ibis.backends.pandas.rewrites import (
        aggregate_to_groupby,
        join_chain_to_nested_joins,
    )
    from ibis.expr.rewrites import _, p

    replace_literals = p.ScalarParameter >> (
        lambda _: ops.Literal(value=params[_], dtype=_.dtype)
    )

    def fn(node, _, **kwargs):
        result = execute(node, **kwargs)
        return result

    original = node

    node = node.to_expr().as_table().op()
    node = node.replace(
        aggregate_to_groupby | join_chain_to_nested_joins | replace_literals
    )
    # print(node.to_expr())
    result = node.map(fn)[node]

    if isinstance(original, ops.Value):
        if original.shape.is_scalar():
            return result.iloc[0, 0]
        elif original.shape.is_columnar():
            return result.iloc[:, 0]
        else:
            raise TypeError(f"Unexpected shape: {original.shape}")

    return result
