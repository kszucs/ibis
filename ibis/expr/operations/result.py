import pathlib
from typing import Mapping

import ibis.expr.rules as rlz
from ibis.common.grounds import Annotable
from ibis.expr.operations.core import Node
from ibis.util import frozendict


class FileFormat(Annotable):
    # each file format should explicitly define the possible options
    options = rlz.optional(rlz.instance_of(Mapping), default=frozendict())


class CsvFormat(FileFormat):
    pass


class ParquetFormat(FileFormat):
    pass


class Result(Node):
    pass


class ScalarResult(Result):
    scalar = rlz.scalar(rlz.any)

    def to_expr(self):
        import ibis.expr.types as ir

        return ir.ScalarResult(self)


class ColumnResult(Result):
    column = rlz.column(rlz.any)

    def to_expr(self):
        import ibis.expr.types as ir

        return ir.ColumnResult(self)


class TableResult(Result):
    table = rlz.table

    def to_expr(self):
        import ibis.expr.types as ir

        return ir.TableResult(self)


class FileTableResult(TableResult):
    path = rlz.instance_of(pathlib.Path)
    format = rlz.instance_of(FileFormat)


class PandasColumnResult(ColumnResult):
    pass


class PandasTableResult(TableResult):
    pass


class PyArrowColumnResult(ColumnResult):
    pass


class PyArrowTableResult(TableResult):
    pass
