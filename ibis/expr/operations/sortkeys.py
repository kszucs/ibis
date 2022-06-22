from public import public

import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis import util
from ibis.common import exceptions as com
from ibis.expr.operations.core import Node


# TODO(kszucs): rewrite to both receive operations and return with operations
def _to_sort_key(key, *, table=None):
    if isinstance(key, DeferredSortKey):
        if table is None:
            raise com.IbisTypeError(
                "cannot resolve DeferredSortKey with table=None"
            )
        key = key.resolve(table)

    # TODO(kszucs): refactor to only work with operation classes
    if isinstance(key, ir.SortExpr):
        return key

    if isinstance(key, ops.SortKey):
        return key.to_expr()

    if isinstance(key, (tuple, list)):
        key, sort_order = key
    else:
        sort_order = True

    if not isinstance(key, ir.Expr):
        if table is None:
            raise com.IbisTypeError("cannot resolve key with table=None")
        key = table._ensure_expr(key)
        if isinstance(key, (ir.SortExpr, DeferredSortKey)):
            return _to_sort_key(key, table=table)

    if isinstance(sort_order, str):
        if sort_order.lower() in ('desc', 'descending'):
            sort_order = False
        elif not isinstance(sort_order, bool):
            sort_order = bool(sort_order)

    return SortKey(key, ascending=sort_order).to_expr()


# TODO(kszucs): rewrite to both receive operations and return with operations
def _maybe_convert_sort_keys(tables, exprs):
    exprs = util.promote_list(exprs)
    keys = exprs[:]
    for i, key in enumerate(exprs):
        step = -1 if isinstance(key, (str, DeferredSortKey)) else 1
        for table in tables[::step]:
            try:
                sort_key = _to_sort_key(key, table=table)
            except Exception:
                continue
            else:
                keys[i] = sort_key
                break

    return [k.op() for k in keys]


@public
class SortKey(Node):
    expr = rlz.column(rlz.any)
    ascending = rlz.optional(
        rlz.map_to(
            {
                True: True,
                False: False,
                1: True,
                0: False,
            },
        ),
        default=True,
    )

    output_type = ir.SortExpr

    def resolve_name(self):
        return self.expr.get_name()


@public
class DeferredSortKey:
    def __init__(self, what, ascending=True):
        self.what = what
        self.ascending = ascending

    def resolve(self, parent):
        what = parent._ensure_expr(self.what)
        return SortKey(what, ascending=self.ascending).to_expr()
