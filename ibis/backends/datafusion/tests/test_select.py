import pickle

import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import pytest

from ibis.backends.datafusion.tests.conftest import BackendTest

pytest.importorskip("datafusion")


def test_where_multiple_conditions(alltypes, alltypes_df):
    expr = alltypes.filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    result = expr.execute()

    expected = alltypes_df[
        (alltypes_df['float_col'] > 0)
        & (alltypes_df['smallint_col'] == 9)
        & (alltypes_df['int_col'] < alltypes_df['float_col'] * 2)
    ]

    BackendTest.assert_frame_equal(result, expected)


def test_different_result_types(alltypes, tmp_path):
    expr = alltypes.filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )

    assert isinstance(expr.to_pyarrow().execute(), pa.Table)
    assert isinstance(expr.to_pandas().execute(), pd.DataFrame)

    # TODO(kszucs): it would be nice to late bind parts of the paths using
    # something like ibis.param()
    csv_result = expr.to_csv(tmp_path / "test.csv")
    parquet_result = expr.to_parquet(tmp_path / "test.parquet")

    csv_result.execute()
    table = csv.read_csv(tmp_path / "test.csv")
    assert isinstance(table, pa.Table)

    parquet_result.execute()
    table = pq.read_table(tmp_path / "test.parquet")
    assert isinstance(table, pa.Table)


def test_pyarrow_result_serialization(client, alltypes):
    expr = alltypes.filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    expr = expr.to_pyarrow()

    restored = pickle.loads(pickle.dumps(expr))
    table = client.execute(restored)
    assert isinstance(table, pa.Table)
