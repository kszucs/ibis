from functools import singledispatch

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq

import ibis.expr.operations as ops
from ibis.backends.datafusion.compiler import translate


@singledispatch
def execute(op, **kwargs):
    raise NotImplementedError(f'No execution rule for {type(op)}')


@execute.register
def to_pylist(op: ops.TableResult, **kwargs):
    raise NotImplementedError(f'No execution rule for {type(op)}')


@execute.register
def to_pandas_dataframe(op: ops.PandasTableResult, **kwargs):
    result = ops.PyArrowTableResult(op.table)
    table = execute(result, **kwargs)
    return table.to_pandas()


@execute.register
def to_pyarrow_table(op: ops.PyArrowTableResult, **kwargs):
    frame = translate(op.table, **kwargs)
    batches = frame.collect()
    return pa.Table.from_batches(batches)


@execute.register
def to_file(op: ops.FileTableResult, **kwargs):
    result = ops.PyArrowTableResult(op.table)
    table = execute(result, **kwargs)

    if isinstance(op.format, ops.CsvFormat):
        return csv.write_csv(table, op.path, **op.format.options)
    elif isinstance(op.format, ops.ParquetFormat):
        return pq.write_table(table, op.path, **op.format.options)
    else:
        raise NotImplementedError(f'No execution rule for {type(op.format)}')
