from ibis.expr.types.relations import (
    bind,
    dereference_values,
    unwrap_aliases,
    dereference_mapping,
)
from public import public
import ibis.expr.operations as ops
from ibis.expr.types import TableExpr, ValueExpr
from typing import Any
from collections.abc import Iterator, Mapping
from ibis.common.deferred import Deferred
from ibis import util
import ibis


def prepare_predicates(chain, right, predicates, dereference_tables):
    """Bind predicates to the left and right tables."""
    # we need to replace fields referencing the current state of the join
    # chain with a field referencing one of the relations in the join chain
    chain_fields = {ops.Field(chain, k): v for k, v in chain.fields.items()}
    # dereference the values in the predicates to one of the join tables but
    # only if the origin of the input is unclear to minimize the change of
    # dereferencing a value to the wrong table in case of self joins
    deref_mapping = dereference_mapping(dereference_tables, extra=chain_fields)

    chain_expr = chain.to_expr()
    right_expr = right.to_expr()
    for pred in util.promote_list(predicates):
        if pred is True or pred is False:
            yield ops.Literal(pred, dtype="bool")
        elif isinstance(pred, ValueExpr):
            node = pred.op()
            yield node.replace(deref_mapping, filter=ops.Value)
        elif isinstance(pred, Deferred):
            # resolve deferred expressions on the left table
            node = pred.resolve(chain_expr).op()
            yield node.replace(deref_mapping, filter=ops.Value)
        else:
            if isinstance(pred, tuple):
                if len(pred) != 2:
                    raise com.ExpressionError("Join key tuple must be length 2")
                lk, rk = pred
            else:
                lk = rk = pred

            # bind the predicates to the join chain
            (left_value,) = bind(chain_expr, lk)
            (right_value,) = bind(right_expr, rk)

            # dereference the left value to one of the relations in the join chain
            left_value = left_value.op().replace(chain_fields, filter=ops.Value)

            yield ops.Equals(left_value, right_value).to_expr()


def disambiguate_fields(how, left_fields, right_fields, lname, rname):
    if how in ("semi", "anti"):
        # discard the right fields
        return left_fields

    lname = lname or "{name}"
    rname = rname or "{name}"
    overlap = left_fields.keys() & right_fields.keys()

    fields = {}
    for name, field in left_fields.items():
        if name in overlap:
            name = lname.format(name=name)
        fields[name] = field
    for name, field in right_fields.items():
        if name in overlap:
            name = rname.format(name=name)
        # only add if there is no collision
        if name not in fields:
            fields[name] = field

    return fields


@public
class JoinExpr(TableExpr):
    def join(
        self,
        right,
        predicates: Any,
        # TODO(kszucs): add typehint about the possible join kinds
        how: str = "inner",
        *,
        lname: str = "",
        rname: str = "{name}_right",
    ):
        """Join with another table."""
        from ibis.expr.analysis import flatten_predicates
        import pyarrow as pa
        import pandas as pd

        if isinstance(right, (pd.DataFrame, pa.Table)):
            right = ibis.memtable(right)
        elif not isinstance(right, TableExpr):
            raise TypeError(
                f"right operand must be a TableExpr, got {type(right).__name__}"
            )

        chain = self.op()
        right = right.op()
        tables = list(self._relations())
        if not isinstance(right, ops.SelfReference):
            # dereference the values in the predicates, if the right table is
            # already a self reference, we don't try to dereference fields to
            # the right side because we assume that the underlying table is
            # already contained by the join chain
            right = ops.SelfReference(right)
            tables.append(right)

        # bind the predicates to the left and right tables
        preds = prepare_predicates(chain, right, predicates, tables)
        preds = flatten_predicates(list(preds))

        # calculate the fields based in lname and rname, this should be a best
        # effort guess but shouldn't raise on conflicts, if there are conflicts
        # they will be raised later on when the join is finished
        left_fields = chain.values
        right_fields = {k: ops.Field(right, k) for k in right.schema}
        values = disambiguate_fields(how, left_fields, right_fields, lname, rname)

        # construct a new join link and add it to the join chain
        link = ops.JoinLink(how, table=right, predicates=preds)
        chain = chain.copy(rest=chain.rest + (link,), values=values)

        # return with a new JoinExpr wrapping the new join chain
        return self.__class__(chain)

    def select(self, *args, **kwargs):
        """Select expressions."""
        values = bind(self, (args, kwargs))
        values = unwrap_aliases(values)

        # if there are values referencing fields from the join chain constructed
        # so far, we need to replace them the fields from one of the join links
        tables = list(self._relations())
        extra = {ops.Field(self, k): v for k, v in self.op().fields.items()}
        values = dereference_values(tables, values, extra=extra)

        return self.finish(values)

    # TODO(kszucs): figure out a solution to automatically wrap all the
    # TableExpr methods including the docstrings and the signature
    def filter(self, *predicates):
        """Filter with `predicates`."""
        return self.finish().filter(*predicates)

    def order_by(self, *keys):
        """Order the join by the given keys."""
        return self.finish().order_by(*keys)

    def _relations(self) -> Iterator[ops.Relation]:
        """Yield all tables in this join expression."""
        node = self.op()
        yield node.first
        for join in node.rest:
            if join.how not in ("semi", "anti"):
                yield join.table

    def finish(self, values: Mapping[str, ops.Field] | None = None) -> TableExpr:
        """Construct a valid table expression from this join expression."""
        node = self.op()
        if values is None:
            # TODO(kszucs): clean this up with a nicer error message
            # raise if there are missing fields from either of the tables
            # raise on collisions
            collisions = []
            values = frozenset(self.op().values.values())
            for rel in self._relations():
                for k in rel.schema:
                    f = ops.Field(rel, k)
                    if f not in values:
                        collisions.append(f)
            if collisions:
                raise com.IntegrityError(f"Name collisions: {collisions}")
        else:
            node = node.copy(values=values)
        # important to explicitly create a TableExpr because .to_expr() would construct
        # a JoinExpr which should be finished again causing an infinite recursion
        return TableExpr(node)
