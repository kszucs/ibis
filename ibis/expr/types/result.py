from ibis.expr.types import Expr


class Result(Expr):
    pass


class ScalarResult(Result):
    pass


class ColumnResult(Result):
    pass


class TableResult(Result):
    pass
