from multipledispatch import Dispatcher

import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.pandas.client import PandasTable
from ibis.backends.pandas.operations import (
    PandasFilter,
    PandasProjection,
    PandasSort,
    simplify,
)
from ibis.common.exceptions import IbisTypeError
from ibis.common.graph import Graph

en = Dispatcher("execute_node")


@en.register(ops.Node)
def execute_node(node, **kwargs):
    raise IbisTypeError(node)


@en.register(ops.DatabaseTable)
def execute_database_table(node, name, schema, source):
    # TODO(kszucs): port the timecontext handling from the original
    # implementation
    return source.dictionary[name]


@en.register(ops.TableColumn)
def execute_database_table(node, table, name):
    return table[name]


@en.register(PandasFilter)
def execute_pandas_filter(node, table, predicate):
    return table.loc[predicate]


@en.register(PandasSort)
def execute_pandas_sort(node, table, fields, ascendings):
    return table.sort_values(list(fields), ascending=list(ascendings))


def execute(node, params, **kwargs):
    node = an.rewrite(simplify, node)

    g = Graph(node)

    # print()
    # print(f"============= PLAN for {node.__class__.__name__} ===============")
    # for e in g.toposort():
    #     print(type(e))

    results = g.map(en)

    return results[node]
