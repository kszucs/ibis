import functools

import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
from ibis.backends.base import Database
from ibis.util import frozendict


class PandasTable(ops.DatabaseTable):
    pass


class PandasDatabase(Database):
    pass


# we can directly call pd.merge on these arguments
class PandasJoin(ops.TableNode):
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
        return self.left.schema.merge(self.right.schema)


class PandasRename(ops.TableNode):
    table = rlz.table
    mapping = rlz.instance_of(frozendict)

    def __init__(self, table, mapping):
        assert all(key in table.schema for key in mapping.keys()), table.schema

    @property
    def schema(self):
        schema = self.table.schema
        names = [self.mapping.get(n, n) for n in schema.names]
        return sch.schema(names, schema.types)


class PandasSelection(ops.TableNode):
    table = rlz.table
    columns = rlz.tuple_of(rlz.instance_of(str))

    @property
    def schema(self):
        return sch.schema(
            {
                name: dtype
                for name, dtype in self.table.schema.items()
                if name in self.columns
            }
        )


class PandasProjection(ops.TableNode):
    values = rlz.tuple_of(rlz.any)
    reset_index = rlz.optional(rlz.instance_of(bool), default=False)

    @property
    def schema(self):
        return sch.schema(
            {value.resolve_name(): value.output_dtype for value in self.values}
        )


class PandasConcatenation(ops.TableNode):
    tables = rlz.tuple_of(rlz.table)

    @property
    def schema(self):
        return functools.reduce(
            lambda a, b: a.merge(b),
            (t.schema for t in self.tables),
            sch.schema({}),
        )


class PandasFilter(ops.TableNode):
    table = rlz.table
    predicate = rlz.boolean

    # TODO(kszucs): make implementing schema property mandatory by adding
    # an abstract property to TableNode
    @property
    def schema(self):
        return self.table.schema


class PandasSort(ops.TableNode):
    table = rlz.table
    fields = rlz.tuple_of(rlz.instance_of(str))
    ascendings = rlz.tuple_of(rlz.instance_of(bool))

    @property
    def schema(self):
        return self.table.schema


class PandasGroupby(ops.TableNode):
    table = rlz.table
    by = rlz.tuple_of(rlz.any)

    @property
    def schema(self):
        return self.table.schema


class PandasAggregate(ops.TableNode):
    table = rlz.table  # perhaps no need for the table at all
    metrics = rlz.tuple_of(rlz.any)


# TODO(kszucs): rewrite aggregation so that the groupby gets executed first
# and the metrc functions can be called on the grouped data afterwards


_join_types = {
    ops.LeftJoin: 'left',
    ops.RightJoin: 'right',
    ops.InnerJoin: 'inner',
    ops.OuterJoin: 'outer',
}


@functools.singledispatch
def simplify(op, **kwargs):
    return op.__class__(**kwargs)


@simplify.register(ops.NotAny)
def simplify_not_any(op, arg):
    return ops.Not(ops.Any(arg))


@simplify.register(ops.NotAll)
def simplify_not_all(op, arg):
    return ops.Not(ops.All(arg))


@simplify.register(ops.Join)
def simplify_join(op, left, right, predicates):
    on = {left: [], right: []}

    # remap overlapping column names, perhaps create a pandasrename object for
    # both sides

    for pred in predicates:
        if not isinstance(pred, ops.Equals):
            raise TypeError(
                'Only equality join predicates supported with pandas'
            )

        on[left].append(pred.left)
        on[right].append(pred.right)

    return PandasJoin(
        left=left,
        right=right,
        how=_join_types[type(op)],
        left_on=on[left],
        right_on=on[right],
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


@simplify.register(ops.Selection)
def simplify_selection(op, table, selections, predicates, sort_keys):
    assert not isinstance(table, ops.Selection)

    if selections:
        tables = []
        values = []
        columns = []
        renames = {}
        for value in selections:
            # need to hack with overlapping column names
            if isinstance(value, ops.Alias) and value.arg.has_resolved_name():
                renames[value.arg.resolve_name()] = value.name
                value = value.arg

            if isinstance(value, ops.TableColumn):
                # TODO(kszucs): check whether the column exists in table?
                columns.append(value.name)
            elif isinstance(value, ops.TableNode):
                tables.append(table)
            else:
                values.append(value)

        if columns:
            tables.append(PandasSelection(table, columns))
        if values:
            tables.append(PandasProjection(values))

        table = PandasConcatenation(tables)
        # if len(tables) == 1:
        #     table = tables[0]
        # elif len(tables) > 1:
        #     table = PandasConcatenation(tables)
        # else:
        #     raise ValueError("EEEEE")

        if renames:
            table = PandasRename(table, mapping=frozendict(renames))

    if predicates:
        predicate = functools.reduce(ops.And, predicates)
        table = PandasFilter(table, predicate=predicate)

    if sort_keys:
        pairs = ((key.expr.name, key.ascending) for key in sort_keys)
        fields, ascendings = zip(*pairs)
        table = PandasSort(table, fields=fields, ascendings=ascendings)

    return table


@simplify.register(ops.Aggregation)
def simplify_aggregation(
    op, table, metrics, by, having, predicates, sort_keys
):
    original_table = table

    if sort_keys:
        raise NotImplementedError(
            'sorting on aggregations not yet implemented'
        )

    if predicates:
        predicate = functools.reduce(ops.And, predicates)
        table = PandasFilter(table, predicate=predicate)

    if by:
        table = PandasGroupby(table, by)

    if metrics:
        # TODO(kszucs): need to rewrite metrics to use the previously created
        # table as base `table`` instead the original `table`, can use
        # an.sub_for for this exact purpose
        new_metrics = [an.replace({original_table: table}, m) for m in metrics]
        table = PandasProjection(new_metrics, reset_index=bool(by))

    return table
