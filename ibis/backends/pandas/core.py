from functools import singledispatch

import pandas as pd
import toolz
from multipledispatch import Dispatcher

import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
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


# class Result:
#     __slots__ = ("_value", "_callback")

#     def get(self):
#         if self._callback:
#             self._value = self._callback()
#             self._callback = None
#         return self._value

#     def set(self, value):
#         if callable(value):
#             self._callback = value
#         else:
#             self._value = value


class ResultStore:

    __slots__ = ("_results", "_arguments")

    def __init__(self, dag):
        self._results = {op: Result() for op in dag.keys()}
        self._arguments = {}

        for op in dag.keys():
            assert not isinstance(op, ir.Expr)
            if isinstance(op, ops.Literal):
                self._arguments[op] = op.args
            else:
                self._arguments[op] = self._construct_results(op.args)

        # convert to weak dict to clean up memory as soon as results not needed
        # results = weakref.WeakValueDictionary(results)

    def _construct_results(self, arg):
        if isinstance(arg, tuple):
            return tuple(map(self._construct_results, arg))
        elif isinstance(arg, ir.Expr):
            return self._results[arg.op()]
        else:
            return arg

    def _retrieve_results(self, result):
        if isinstance(result, tuple):
            return tuple(map(self._retrieve_results, result))
        elif isinstance(result, Result):
            return result.get()
        else:
            return result

    def arguments_for(self, op):
        args = self._arguments[op]
        args = self._retrieve_results(args)
        return args

    def set(self, op, value):
        assert not isinstance(value, ir.Expr)
        self._results[op].set(value)

    def get(self, op):
        value = self._results[op].get()
        assert not isinstance(value, ir.Expr)
        return value


def execute(expr, clients=None, params=None, **kwargs):
    from . import execution

    params = {expr.op(): value for expr, value in (params or {}).items()}

    expr = rewrite(expr)

    dag = util.to_op_dag(expr)
    store = ResultStore(dag)

    for op in util.toposort(dag):
        args = store.arguments_for(op)
        result = execute_node(op, *args, params=params, **kwargs)
        store.set(op, result)

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


class PhysicalOp:
    # could/should validate that none of the args is an Expr
    pass


class PandasOp:
    pass


class PandasTable(ops.DatabaseTable, PandasOp):
    pass


class PandasDatabase(Database, PandasOp):
    pass


# we can directly call pd.merge on these arguments
class PandasJoin(ops.TableNode, PandasOp):
    # add the contents of ibis.backends.pandas_old.execution.join.execute_join here
    left = rlz.table()
    right = rlz.table()
    how = rlz.instance_of(str)
    left_on = rlz.tuple_of(
        rlz.one_of([rlz.instance_of(str), rlz.column(rlz.any)])
    )
    right_on = rlz.tuple_of(
        rlz.one_of([rlz.instance_of(str), rlz.column(rlz.any)])
    )

    @property
    def schema(self):
        # For joins retaining both table schemas, merge them together here
        left = dict(self.left.schema().items())
        right = dict(self.right.schema().items())
        merged = {**left, **right}
        return sch.schema(merged)

    def root_tables(self):
        return [self.left.op(), self.right.op()]


class PandasProjection(ops.TableNode, PandasOp):
    table = rlz.table()
    columns = rlz.tuple_of(rlz.instance_of(str))

    @property
    def schema(self):
        # For joins retaining both table schemas, merge them together here
        schema = self.table.schema()
        return sch.schema({name: schema[name] for name in self.columns})


# class PandasProjection


@singledispatch
def rewrite(op):
    return op


@rewrite.register(tuple)
def rewrite_tuple(args):
    return tuple(map(rewrite, args))


@rewrite.register(ir.Expr)
def rewrite_expr(expr):
    old_op = expr.op()
    new_op = rewrite(old_op)
    if new_op != old_op:
        expr = new_op.to_expr()
    return expr


@rewrite.register(ops.Node)
def rewrite_node(op):
    old_args = op.args
    new_args = rewrite(old_args)
    if new_args != old_args:
        op = type(op)(*new_args)
    return op


_join_types = {
    ops.LeftJoin: 'left',
    ops.RightJoin: 'right',
    ops.InnerJoin: 'inner',
    ops.OuterJoin: 'outer',
}


@rewrite.register(ops.Join)
def rewrite_join(op):
    on = {op.left.op(): [], op.right.op(): []}

    for predicate in op.predicates:
        pred = predicate.op()
        if not isinstance(pred, ops.Equals):
            raise TypeError(
                'Only equality join predicates supported with pandas'
            )

        left = pred.left.op()
        (table,) = left.root_tables()
        if isinstance(left, ops.TableColumn):
            on[table].append(left.name)
        else:
            on[table].append(rewrite(pred.left))

        right = pred.right.op()
        (table,) = right.root_tables()
        if isinstance(right, ops.TableColumn):
            on[table].append(right.name)
        else:
            on[table].append(rewrite(pred.right))

    return PandasJoin(
        left=op.left,
        right=op.right,
        how=_join_types[type(op)],
        left_on=on[op.left.op()],
        right_on=on[op.right.op()],
    )
    # return PandasProjection(join, ...)
    # return PandasRemapNames(
    #     table=PandasJoin(
    #         left=op.left,
    #         right=op.right,
    #         how=_join_types[type(op)],
    #         left_on=on[op.left.op()],
    #         right_on=on[op.right.op()],
    #     )
    # )


@rewrite.register(ops.Selection)
def rewrite_selection(op):
    table = rewrite(op.table)

    if op.selections:
        columns = []
        for field in op.selections:
            node = field.op()
            if isinstance(node, sch.HasSchema):
                columns.extend(node.schema.names)
            else:
                columns.append(node.name)

        table = PandasProjection(
            table=table, columns=tuple(toolz.unique(columns))
        )

    return table
