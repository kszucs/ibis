"""Execution rules for generic ibis operations."""

import collections
import datetime
import decimal
import functools
import math
import numbers
import operator
from collections.abc import Sized
from typing import Dict, Optional

import numpy as np
import pandas as pd
import toolz
from pandas.api.types import DatetimeTZDtype
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.expr.schema import Schema
from ibis.expr.scope import Scope
from ibis.expr.timecontext import get_time_col
from ibis.expr.typing import TimeContext

from .. import Backend as PandasBackend
from .. import aggcontext as agg_ctx
from ..client import PandasTable
from ..core import (
    boolean_types,
    execute,
    fixed_width_types,
    floating_types,
    integer_types,
    numeric_types,
    scalar_types,
    simple_types,
    timedelta_types,
    timestamp_types,
)
from ..dispatch import execute_literal, execute_node
from ..execution import constants
from ..execution.util import coerce_to_output

# @execute_node.register(ops.Reduction, SeriesGroupBy, type(None))
# def execute_reduction_series_groupby(op, data, mask, **kwargs):
#     def metric(aggcontext, data=data):
#         return aggcontext.agg(data, type(op).__name__.lower())

#     return metric


variance_ddof = {'pop': 0, 'sample': 1}


# @execute_node.register(ops.Variance, SeriesGroupBy, type(None))
# def execute_reduction_series_groupby_var(op, data, _, **kwargs):
#     def metric(aggcontext, data=data):
#         return aggcontext.agg(data, 'var', ddof=variance_ddof[op.how])

#     return metric


# @execute_node.register(ops.StandardDev, SeriesGroupBy, type(None))
# def execute_reduction_series_groupby_std(op, data, _, **kwargs):
#     def metric(aggcontext, data=data):
#         return aggcontext.agg(data, 'std', ddof=variance_ddof[op.how])

#     return metric


# @execute_node.register(
#     (ops.CountDistinct, ops.HLLCardinality), SeriesGroupBy, type(None)
# )
# def execute_count_distinct_series_groupby(op, data, _, **kwargs):
#     def metric(aggcontext, data=data):
#         return aggcontext.agg(data, 'nunique')

#     return metric


# @execute_node.register(ops.Arbitrary, SeriesGroupBy, type(None))
# def execute_arbitrary_series_groupby(op, data, _, **kwargs):
#     how = op.how
#     if how is None:
#         how = 'first'

#     if how not in {'first', 'last'}:
#         raise com.OperationNotDefinedError(
#             f'Arbitrary {how!r} is not supported'
#         )

#     def metric(aggcontext, data=data):
#         return aggcontext.agg(data, how)

#     return metric


def _filtered_reduction(mask, method, data):
    return method(data[mask[data.index]])


# @execute_node.register(ops.Reduction, SeriesGroupBy, SeriesGroupBy)
# def execute_reduction_series_gb_mask(op, data, mask, **kwargs):
#     method = operator.methodcaller(type(op).__name__.lower())
#     function = functools.partial(_filtered_reduction, mask.obj, method)

#     def metric(aggcontext, data=data):
#         return aggcontext.agg(data, function)


# @execute_node.register(
#     (ops.CountDistinct, ops.HLLCardinality), SeriesGroupBy, SeriesGroupBy
# )
# def execute_count_distinct_series_groupby_mask(
#     op, data, mask, aggcontext=None, **kwargs
# ):
#     function = (
#         functools.partial(_filtered_reduction, mask.obj, pd.Series.nunique),
#     )

#     def metric(aggcontext, data=data):
#         return aggcontext.agg(data, function)

#     return metric


# @execute_node.register(ops.Variance, SeriesGroupBy, SeriesGroupBy)
# def execute_var_series_groupby_mask(op, data, mask, **kwargs):
#     def function(x, mask=mask.obj, ddof=variance_ddof[op.how]):
#         return x[mask[x.index]].var(ddof=ddof)

#     def metric(aggcontext, data=data):
#         return aggcontext.agg(data, function)

#     return metric


# @execute_node.register(ops.StandardDev, SeriesGroupBy, SeriesGroupBy)
# def execute_std_series_groupby_mask(op, data, mask, **kwargs):
#     def function(x, mask=mask.obj, ddof=variance_ddof[op.how]):
#         return x[mask[x.index]].std(ddof=ddof)

#     def metric(aggcontext, data=data):
#         return aggcontext.agg(data, function)

#     return metric


# @execute_node.register(ops.Count, DataFrameGroupBy, type(None))
# def execute_count_frame_groupby(op, data, _, **kwargs):
#     def metric(aggcontext, data=data):
#         result = data.size()
#         # FIXME(phillipc): We should not hard code this column name
#         result.name = 'count'
#         return result

#     return metric


@execute_node.register(ops.Reduction, pd.Series, (pd.Series, type(None)))
def execute_reduction_series_mask(op, data_, mask, **kwargs):
    def metric(aggcontext, data=None):

        if data is None:
            data = data_
            data = data[mask] if mask is not None else data
        else:
            data = data[data_.name]
            data = data[mask] if mask is not None else data

        return aggcontext.agg(data, type(op).__name__.lower())

    return metric


@execute_node.register(
    (ops.CountDistinct, ops.HLLCardinality), pd.Series, (pd.Series, type(None))
)
def execute_count_distinct_series_mask(op, data_, mask, **kwargs):
    def metric(aggcontext, data=None):
        if data is None:
            data = data_
        else:
            data = data[data_.name]
        return aggcontext.agg(
            data[mask] if mask is not None else data, 'nunique'
        )

    return metric


@execute_node.register(ops.Arbitrary, pd.Series, (pd.Series, type(None)))
def execute_arbitrary_series_mask(op, data_, mask, **kwargs):
    if op.how == 'first':
        index = 0
    elif op.how == 'last':
        index = -1
    else:
        raise com.OperationNotDefinedError(
            f'Arbitrary {op.how!r} is not supported'
        )

    def metric(aggcontext, data=None):
        if data is None:
            data = data_
        else:
            data = data[data_.name]
        data = data[mask] if mask is not None else data
        return data.iloc[index]

    return metric


@execute_node.register(
    ops.StandardDev, pd.Series, str, (pd.Series, type(None))
)
def execute_standard_dev_series(op, data_, how, mask, **kwargs):
    def metric(aggcontext, data=None):
        if data is None:
            data = data_
        else:
            data = data[data_.name]
        return aggcontext.agg(
            data[mask] if mask is not None else data,
            'std',
            ddof=variance_ddof[how],
        )

    return metric


@execute_node.register(ops.Variance, pd.Series, str, (pd.Series, type(None)))
def execute_variance_series(op, data_, how, mask, **kwargs):
    def metric(aggcontext, data=None):
        if data is None:
            data = data_
        else:
            data = data[data_.name]
        return aggcontext.agg(
            data[mask] if mask is not None else data,
            'var',
            ddof=variance_ddof[how],
        )

    return metric


@execute_node.register((ops.Any, ops.All), (pd.Series, SeriesGroupBy))
def execute_any_all_series(op, data_, **kwargs):
    def metric(aggcontext, data=None):
        if data is None:
            data = data_
        else:
            data = data[data_.name]
        if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
            result = aggcontext.agg(data, type(op).__name__.lower())
        else:
            result = aggcontext.agg(
                data, lambda data: getattr(data, type(op).__name__.lower())()
            )
        try:
            return result.astype(bool)
        except TypeError:
            return result

    return metric


@execute_node.register(ops.NotAny, (pd.Series, SeriesGroupBy))
def execute_notany_series(op, data_, **kwargs):
    def metric(aggcontext, data=None):
        if data is None:
            data = data_
        else:
            data = data[data_.name]
        if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
            result = ~(aggcontext.agg(data, 'any'))
        else:
            result = aggcontext.agg(data, lambda data: ~(data.any()))
        try:
            return result.astype(bool)
        except TypeError:
            return result

    return metric


@execute_node.register(ops.NotAll, (pd.Series, SeriesGroupBy))
def execute_notall_series(op, data_, **kwargs):
    def metric(aggcontext, data=None):
        if data is None:
            data = data_
        else:
            data = data[data_.name]
        if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
            result = ~(aggcontext.agg(data, 'all'))
        else:
            result = aggcontext.agg(data, lambda data: ~(data.all()))
        try:
            return result.astype(bool)
        except TypeError:
            return result

    return metric


@execute_node.register(ops.Count, pd.DataFrame, type(None))
def execute_count_frame(op, data_, _, **kwargs):
    def metric(aggcontext, data):
        if data is None:
            data = data_
        return data.count()  # len(data)

    return metric
