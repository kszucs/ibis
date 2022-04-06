from typing import Any

import pandas as pd
import pandas.testing as tm
import pytest
from multipledispatch.conflict import ambiguities

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.pandas import Backend
from ibis.backends.pandas.core import execute, execute_node


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            'plain_int64': list(range(1, 4)),
            'plain_strings': list('abc'),
            'dup_strings': list('dad'),
        }
    )


@pytest.fixture
def core_client(dataframe):
    return Backend().connect({'df': dataframe})


@pytest.fixture
def ibis_table(core_client):
    return core_client.table('df')


@pytest.mark.parametrize('func', [execute_node])
def test_no_execute_ambiguities(func):
    assert not ambiguities(func.funcs)


def test_from_dataframe(dataframe, ibis_table, core_client):
    t = Backend().from_dataframe(dataframe)
    result = t.execute()
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)

    t = Backend().from_dataframe(dataframe, name='foo')
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)

    client = core_client
    t = Backend().from_dataframe(dataframe, name='foo', client=client)
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)


def test_missing_data_sources():
    t = ibis.table([('a', 'string')])
    expr = t.a.length()
    with pytest.raises(com.UnboundExpressionError):
        execute(expr)


def test_missing_data_on_custom_client():
    class MyBackend(Backend):
        def table(self, name):
            return ops.DatabaseTable(
                name, ibis.schema([('a', 'int64')]), self
            ).to_expr()

    con = MyBackend()
    t = con.table('t')
    with pytest.raises(
        NotImplementedError,
        match=(
            'Could not find signature for execute_node: '
            '<DatabaseTable, str, Schema, MyBackend>'
        ),
    ):
        con.execute(t)
