from ibis.expr.types import Expr


class Result(Expr):
    pass


class ScalarResult(Result):
    pass


class ColumnResult(Result):
    pass


class TableResult(Result):
    def to_csv(self, path):
        table = self.op().table.to_expr()
        return table.to_csv(path)

    def to_parquet(self, path):
        table = self.op().table.to_expr()
        return table.to_parquet(path)

    def to_pylist(self):
        table = self.op().table.to_expr()
        return table.to_pylist()

    def to_pandas(self):
        table = self.op().table.to_expr()
        return table.to_pandas()

    def to_pyarrow(self):
        table = self.op().table.to_expr()
        return table.to_pyarrow()
