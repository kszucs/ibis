from __future__ import annotations

from abc import abstractmethod
from itertools import zip_longest
from typing import Annotated, Any

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.common.collections import FrozenDict  # noqa: TCH001
from ibis.common.deferred import Deferred, Item, deferred, var
from ibis.common.exceptions import IntegrityError
from ibis.common.graph import Traversable
from ibis.common.grounds import Concrete
from ibis.common.patterns import Check, In, InstanceOf, _, pattern, replace
from ibis.common.typing import Coercible, VarTuple
from ibis.expr.operations import Alias, Node, SortKey, Value
from ibis.expr.schema import Schema
from ibis.expr.types import Expr, literal
from ibis.expr.types import Value as ValueExpr
from ibis.selectors import Selector
from ibis.util import Namespace

# need a parallel Expression and Operation class hierarchy to decompose ops.Selection
# into proper relational algebra operations


################################ OPERATIONS ################################


class Relation(Node, Coercible):
    @classmethod
    def __coerce__(cls, value):
        if isinstance(value, Relation):
            return value
        elif isinstance(value, TableExpr):
            return value.op()
        else:
            raise TypeError(f"Cannot coerce {value!r} to a Relation")

    @property
    @abstractmethod
    def fields(self):
        ...

    @property
    @abstractmethod
    def schema(self):
        ...

    def to_expr(self):
        return TableExpr(self)


class Field(Value):
    rel: Relation
    name: str

    shape = ds.columnar

    @property
    def dtype(self):
        return self.rel.schema[self.name]


def _check_integrity(values, allowed_fields):
    allowed_fields = set(allowed_fields)
    for value in values:
        for field in value.find(Field, filter=Value):
            if field not in allowed_fields:
                raise IntegrityError(
                    f"Cannot add {field!r} to projection, "
                    f"it belongs to {field.rel!r}"
                )


class Project(Relation):
    parent: Relation
    values: FrozenDict[str, Annotated[Value, ~InstanceOf(Alias)]]

    def __init__(self, parent, values):
        _check_integrity(values.values(), allowed_fields=parent.fields.values())
        super().__init__(parent=parent, values=values)

    # reprojection  cuold be just carried over from the parent?
    @property
    def fields(self):
        return {k: Field(self, k) for k in self.values}

    @property
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


# TODO(kszucs): consider to have a specialization projecting only fields not
# generic value expressions
# class ProjectFields(Relation):
#     parent: Relation
#     fields: FrozenDict[str, Field]

#     @attribute
#     def schema(self):
#         return Schema({f.name: f.dtype for f in self.fields})

# TODO(kszucs): Subquery(value, outer_relation)

class Join(Relation):
    left: Relation
    right: Relation
    fields: FrozenDict[str, Annotated[Field, ~InstanceOf(Alias)]]
    predicates: VarTuple[Value[dt.Boolean]]
    how: str = "inner"

    def __init__(self, left, right, fields, predicates, how):
        allowed_fields = {*left.fields.values(), *right.fields.values()}
        _check_integrity(fields.values(), allowed_fields)
        _check_integrity(predicates, allowed_fields)
        super().__init__(
            left=left, right=right, fields=fields, predicates=predicates, how=how
        )

    @property
    def schema(self):
        return Schema({k: v.dtype for k, v in self.fields.items()})


# class JoinProject(Relation):
#     parent: Join
#     fields: FrozenDict[str, Field]

#     @property
#     def schema(self):
#         return Schema({f.name: f.dtype for f in self.fields})


class Sort(Relation):
    parent: Relation
    keys: VarTuple[SortKey]

    def __init__(self, parent, keys):
        _check_integrity(keys, allowed_fields=parent.fields.values())
        super().__init__(parent=parent, keys=keys)

    @property
    def fields(self):
        return self.parent.fields

    @property
    def schema(self):
        return self.parent.schema


class Filter(Relation):
    parent: Relation
    predicates: VarTuple[Value[dt.Boolean]]

    def __init__(self, parent, predicates):
        # TODO(kszucs): use toolz.unique(predicates) to remove duplicates
        _check_integrity(predicates, allowed_fields=parent.fields.values())
        super().__init__(parent=parent, predicates=predicates)

    @property
    def fields(self):
        return self.parent.fields

    @property
    def schema(self):
        return self.parent.schema


# class Aggregate(Relation):
#     parent: Relation
#     groups: FrozenDict[str, Annotated[Column, ~InstanceOf(Alias)]]
#     metrics: FrozenDict[str, Annotated[Scalar, ~InstanceOf(Alias)]]

#     @attribute
#     def schema(self):
#         # schema is consisting both by and metrics, use .from_tuples() to disallow
#         # duplicate names in the schema
#         return Schema.from_tuples([*self.groups.items(), *self.metrics.items()])


class UnboundTable(Relation):
    name: str
    schema: Schema

    @property
    def fields(self):
        return {k: Field(self, k) for k in self.schema}


################################ TYPES ################################


class TableExpr(Expr):
    def schema(self):
        return self.op().schema

    def __getattr__(self, key):
        return next(bind(self, key))

    def select(self, *args, **kwargs):
        values = bind(self, (args, kwargs))
        values = unwrap_aliases(values)
        # TODO(kszucs): windowization of reductions should happen here
        node = Project(self, values)
        node = node.replace(complete_reprojection | subsequent_projects)
        return node.to_expr()

    def filter(self, *predicates):
        preds = bind(self, predicates)
        preds = unwrap_aliases(predicates).values()
        # TODO(kszucs): add predicate flattening
        node = Filter(self, preds)
        node = node.replace(subsequent_filters)
        return node.to_expr()

    def order_by(self, *keys):
        keys = bind(self, keys)
        keys = unwrap_aliases(keys).values()
        node = Sort(self, keys)
        node = node.replace(subsequent_sorts)
        return node.to_expr()

    def join(self, right, predicates, how="inner"):
        return JoinChain(self).join(right, predicates, how)

    def aggregate(self, groups, metrics):
        groups = bind(self, groups)
        metrics = bind(self, metrics)
        groups = unwrap_aliases(groups)
        metrics = unwrap_aliases(metrics)
        node = Aggregate(self, groups, metrics)
        return node.to_expr()


class JoinLink(Concrete, Traversable):
    how: str = "inner"
    right: Relation
    predicates: VarTuple[Value[dt.Boolean]]


class JoinChain(Concrete):
    start: Relation
    links: VarTuple[JoinLink] = ()

    def join(self, right, predicates, how="inner"):
        # do the predicate coercion and binding here
        link = JoinLink(how=how, right=right, predicates=predicates)
        return self.copy(links=self.links + (link,))

    def select(self, *args, **kwargs):
        # do the fields projection here
        # TODO(kszucs): need to do smarter binding here since references may
        # point to any of the relations in the join chain
        values = bind(self.start.to_expr(), (args, kwargs))
        values = unwrap_aliases(values)

        # TODO(kszucs): go over the values and pull out the fields only, until
        # that just raise if the value is a computed expression
        for value in values.values():
            if not isinstance(value, Field):
                raise TypeError("Only fields can be selected in a join")

        return self.finish(values)

    # TODO(kszucs): figure out a solution to automatically wrap all the
    # TableExpr methods including the docstrings and the signature
    def where(self, *predicates):
        return self.finish().where(*predicates)

    def order_by(self, *keys):
        return self.finish().order_by(*keys)

    def guess(self):
        # TODO(kszucs): more advanced heuristics can be applied here
        # trying to figure out the join intent here
        fields = self.start.fields
        for link in self.links:
            if fields.keys() & link.right.fields.keys():
                # overlapping fields
                raise IntegrityError("Overlapping fields in join")
            fields.update(link.right.fields)
        return fields

    def finish(self, fields=None):
        if fields is None:
            fields = self.guess()

        # build the possibly nested join operation from the join chain
        left = self.start
        for link, next_link in zip_longest(self.links, self.links[1:]):
            # pick the fields which are referenced by the final projection
            possible_fields = {*left.fields.values(), *link.right.fields.values()}
            selected_fields = {k: v for k, v in fields.items() if v in possible_fields}

            # pick fields from either the left or the right side of the join which
            # are referenced by the predicates of the upcoming join
            if next_link is not None:
                rule = p.Field(In({left, link.right}))
                for field in next_link.match(rule, filter=~p.Relation):
                    selected_fields[field.name] = field

            left = Join(
                how=link.how,
                left=left,
                right=link.right,
                predicates=link.predicates,
                fields=selected_fields,
            )
        return left.to_expr()

    def __getattr__(self, name):
        return next(bind(self.finish(), name))


def bind(table: TableExpr, value: Any) -> ir.Value:
    node = table.op()
    if isinstance(value, ValueExpr):
        yield value
    elif isinstance(value, TableExpr):
        for name in value.schema().keys():
            yield Field(value, name).to_expr()
    elif isinstance(value, str):
        # column peeling / dereferencing
        yield node.fields[value].to_expr().name(value)
    elif isinstance(value, Deferred):
        yield value.resolve(table)
    elif isinstance(value, Selector):
        yield from value.expand(table)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from bind(table, v)
    elif isinstance(value, dict):
        for k, v in value.items():
            for val in bind(table, v):
                yield val.name(k)
    elif callable(value):
        yield value(table)
    else:
        yield literal(value)


def unwrap_aliases(values):
    result = {}
    for value in values:
        node = value.op()
        if isinstance(node, Alias):
            result[node.name] = node.arg
        else:
            result[node.name] = node
    return result


################################ REWRITES ################################


p = Namespace(pattern, module=__name__)
d = Namespace(deferred, module=__name__)

name = var("name")

x = var("x")
y = var("y")
values = var("values")


@replace(p.Project(y @ p.Relation) & Check(_.schema == y.schema))
def complete_reprojection(_, y):
    # TODO(kszucs): this could be moved to the pattern itself but not sure how
    # to express it, especially in a shorter way then the following check
    for name in _.schema:
        if _.values[name] != Field(y, name):
            return x
    return y


@replace(p.Project(y @ p.Project))
def subsequent_projects(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    values = {k: v.replace(rule) for k, v in _.values.items()}
    return Project(y.parent, values)


@replace(p.Filter(y @ p.Filter))
def subsequent_filters(_, y):
    return Filter(y.parent, y.predicates + _.predicates)


# TODO(kszucs): this may work if the sort keys are not overlapping, need to revisit
@replace(p.Sort(y @ p.Sort))
def subsequent_sorts(_, y):
    return Sort(y.parent, y.keys + _.keys)


# TODO(kszucs): support t.select(*t) syntax by implementing TableExpr.__iter__()
