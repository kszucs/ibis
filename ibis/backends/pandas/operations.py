import functools
from tkinter.tix import Select

import pandas as pd
import toolz
from multipledispatch import Dispatcher

import ibis.expr.analysis as an
import ibis.expr.lineage as lin
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import Database

# class PandasOp:
#     pass


class PandasTable(ops.DatabaseTable):
    pass


class PandasDatabase(Database):
    pass


# we can directly call pd.merge on these arguments
# class PandasJoin(ops.TableNode, PandasOp):
#     # add the contents of ibis.backends.pandas_old.execution.join.execute_join here
#     left = rlz.table()
#     right = rlz.table()
#     how = rlz.instance_of(str)
#     left_on = rlz.tuple_of(
#         rlz.one_of([rlz.instance_of(str), rlz.column(rlz.any)])
#     )
#     right_on = rlz.tuple_of(
#         rlz.one_of([rlz.instance_of(str), rlz.column(rlz.any)])
#     )

#     @property
#     def schema(self):
#         # For joins retaining both table schemas, merge them together here
#         left = dict(self.left.schema.items())
#         right = dict(self.right.schema.items())
#         merged = {**left, **right}
#         return sch.schema(merged)

#     def blocks(self):
#         return False


class PandasProjection(ops.TableNode):
    table = rlz.table()
    columns = rlz.tuple_of(rlz.instance_of(str))

    @property
    def schema(self):
        # For joins retaining both table schemas, merge them together here
        fields = {name: self.table.schema[name] for name in self.columns}
        return sch.schema(fields)


class PandasFilter(ops.TableNode):
    table = rlz.table()
    predicate = rlz.boolean

    # TODO(kszucs): make implementing schema property mandatory by adding
    # an abstract property to TableNode
    @property
    def schema(self):
        return self.table.schema


class PandasSort(ops.TableNode):
    table = rlz.table()
    fields = rlz.tuple_of(rlz.instance_of(str))
    ascendings = rlz.tuple_of(rlz.instance_of(bool))

    @property
    def schema(self):
        return self.table.schema


_join_types = {
    ops.LeftJoin: 'left',
    ops.RightJoin: 'right',
    ops.InnerJoin: 'inner',
    ops.OuterJoin: 'outer',
}


@functools.singledispatch
def simplify(op, **kwargs):
    return op.__class__(**kwargs)


# @simplify.register(ops.Join)
# def simplify_join(op):
#     on = {op.left: [], op.right: []}

#     for pred in op.predicates:
#         if not isinstance(pred, ops.Equals):
#             raise TypeError(
#                 'Only equality join predicates supported with pandas'
#             )

#         if isinstance(pred.left, ops.TableColumn):
#             on[op.left].append(pred.left.name)
#         else:
#             on[op.left].append(pred.left)

#         if isinstance(pred.right, ops.TableColumn):
#             on[op.right].append(pred.right.name)
#         else:
#             on[op.right].append(pred.right)

#     return PandasJoin(
#         left=op.left,
#         right=op.right,
#         how=_join_types[type(op)],
#         left_on=on[op.left],
#         right_on=on[op.right],
#     )

#     # return PandasProjection(join, ...)
#     # return PandasRemapNames(
#     #     table=PandasJoin(
#     #         left=op.left,
#     #         right=op.right,
#     #         how=_join_types[type(op)],
#     #         left_on=on[op.left.op()],
#     #         right_on=on[op.right.op()],
#     #     )
#     # )


@simplify.register(ops.Selection)
def simplify_selection(op, table, selections, predicates, sort_keys):
    assert not isinstance(table, ops.Selection)

    if selections:
        columns = []
        for field in selections:
            if isinstance(field, ops.TableNode):
                columns.extend(field.schema.names)
            else:
                columns.append(field.name)

        table = PandasProjection(
            table=table, columns=tuple(toolz.unique(columns))
        )

    if predicates:
        predicate = functools.reduce(ops.And, predicates)
        table = PandasFilter(table, predicate=predicate)

    if sort_keys:
        pairs = ((key.expr.name, key.ascending) for key in sort_keys)
        fields, ascendings = zip(*pairs)

        table = PandasSort(table, fields=fields, ascendings=ascendings)

    return table
