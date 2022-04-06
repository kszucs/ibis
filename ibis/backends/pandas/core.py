import pandas as pd
from multipledispatch import Dispatcher

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import Database

from .dispatcher import TwoLevelDispatcher


class Result:
    __slots__ = ("_value",)

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class ResultStore:

    __slots__ = ("_results", "_arguments")

    def __init__(self, dag):
        self._results = {op: Result() for op in dag.keys()}
        self._arguments = {}

        for op in dag.keys():
            if isinstance(op, ops.Literal):
                self._arguments[op] = op.args
            else:
                self._arguments[op] = tuple(map(self._construct_args, op.args))

        # convert to weak dict to clean up memory as soon as results not needed
        # results = weakref.WeakValueDictionary(results)

    def _construct_args(self, expr):
        if isinstance(expr, tuple):
            return tuple(self._results[e.op()] for e in expr)
        elif isinstance(expr, ir.Expr):
            return self._results[expr.op()]
        else:
            return expr

    def _retrieve_args(self, result):
        if isinstance(result, tuple):
            return tuple(map(self._retrieve_args, result))
        elif isinstance(result, Result):
            return result.get()
        else:
            return result

    def arguments_for(self, op):
        args = self._arguments[op]
        args = tuple(map(self._retrieve_args, args))
        return args

    def set(self, op, value):
        self._results[op].set(value)

    def get(self, op):
        self._results[op].get()


def execute(expr, clients=None, **kwargs):
    dag = util.to_op_dag(expr)

    store = ResultStore(dag)

    for op in util.toposort(dag):
        args = store.arguments_for(op)
        result = execute_node(op, *args, timecontext=None, aggcontext=None)
        store.set(op, result)

    # FIXME(kszucs): hack, should wrap the incoming operation to another
    # used for single field reductions: t.int64.sum()
    # if callable(result):
    #     aggcontext = agg_ctx.Summarize()
    #     value = value(aggcontext)

    # reset index of output dataframes
    if isinstance(result, pd.DataFrame):
        schema = expr.schema()
        df = result.reset_index()
        return df.loc[:, schema.names]
    elif isinstance(result, pd.Series):
        return result.reset_index(drop=True)
    else:
        return result


# Individual operation execution
execute_node = TwoLevelDispatcher(
    'execute_node',
    doc=(
        'Execute an individual operation given the operation and its computed '
        'arguments'
    ),
)


class PandasTable(ops.DatabaseTable):
    pass


class PandasDatabase(Database):
    pass
