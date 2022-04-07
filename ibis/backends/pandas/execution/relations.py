"""Execution rules for generic ibis operations."""

import datetime
import numbers

import numpy as np
import pandas as pd

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis import util
from ibis.common.exceptions import UnboundExpressionError
from ibis.expr.operations.generic import TableColumn

from .. import Backend as PandasBackend
from ..core import (
    PandasFilter,
    PandasJoin,
    PandasProjection,
    PandasTable,
    execute_node,
)

register = execute_node.register


@register(ops.UnboundTable, sch.Schema, str)
def execute_unbound_table(op, schema, name, **kwargs):
    raise UnboundExpressionError(op)


@register(PandasTable, str, sch.Schema, PandasBackend)
def execute_database_table_client(op, name, schema, client, **kwargs):
    df = client.dictionary[op.name]
    # if timecontext:
    #     begin, end = timecontext
    #     time_col = get_time_col()
    #     if time_col not in df:
    #         raise com.IbisError(
    #             f'Table {op.name} must have a time column named {time_col}'
    #             ' to execute with time context.'
    #         )
    #     # filter with time context
    #     mask = df[time_col].between(begin, end)
    #     return df.loc[mask].reset_index(drop=True)
    return df


@register(ops.TableColumn, pd.DataFrame, str)
def execute_table_column(op, df, name, **kwargs):
    # perhaps just return with the name so Selection and Aggregation
    # can handle the execution one level upper
    return df[name]


@register(ops.Selection, pd.DataFrame, tuple, tuple, tuple)
def execute_selection(op, df, selections, predicates, sort_keys, **kwargs):
    result = df

    # Build up the individual pandas structures from column expressions
    if selections:
        result = pd.concat(selections, axis=1)
        # if all(isinstance(s.op(), ops.TableColumn) for s in selections):
        #     result = build_df_from_selection(selections, data, op.table.op())
        # else:
        #     result = build_df_from_projection(
        #         selections,
        #         op,
        #         data,
        #         scope=scope,
        #         timecontext=timecontext,
        #         **kwargs,
        #     )

    return result
    # if predicates:
    #     predicates = _compute_predicates(
    #         op.table.op(), predicates, data, scope, timecontext, **kwargs
    #     )
    #     predicate = functools.reduce(operator.and_, predicates)
    #     assert len(predicate) == len(
    #         result
    #     ), 'Selection predicate length does not match underlying table'
    #     result = result.loc[predicate]

    # if sort_keys:
    #     result, grouping_keys, ordering_keys = util.compute_sorted_frame(
    #         result,
    #         order_by=sort_keys,
    #         scope=scope,
    #         timecontext=timecontext,
    #         **kwargs,
    #     )
    # else:
    #     grouping_keys = ordering_keys = ()

    # # return early if we do not have any temporary grouping or ordering columns
    # assert not grouping_keys, 'group by should never show up in Selection'
    # if not ordering_keys:
    #     return result

    # # create a sequence of columns that we need to drop
    # temporary_columns = pd.Index(
    #     concatv(grouping_keys, ordering_keys)
    # ).difference(data.columns)

    # # no reason to call drop if we don't need to
    # if temporary_columns.empty:
    #     return result

    # # drop every temporary column we created for ordering or grouping
    # return result.drop(temporary_columns, axis=1)


_join_suffixes = (f'_ibis_left_{util.guid()}', f'_ibis_right_{util.guid()}')


@register(PandasJoin, pd.DataFrame, pd.DataFrame, str, tuple, tuple)
def execute_pandas_join(op, left, right, how, left_on, right_on, **kwargs):
    return pd.merge(
        left,
        right,
        how=how,
        left_on=left_on,
        right_on=right_on,
        suffixes=_join_suffixes,
    )
    # remap column names or create an Op for remapping column names


@register(PandasProjection, pd.DataFrame, tuple)
def execute_pandas_projection(op, table, columns, **kwargs):
    print(table)
    return table[list(columns)]


# @register(PandasFilter,)
