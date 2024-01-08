from __future__ import annotations

import contextlib
import uuid

import pytest
import sqlalchemy.exc
from packaging.version import parse as vparse

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt

RAW_TEST_UUID = "08f48812-7948-4718-96c7-27fa6a398db6"
TEST_UUID = uuid.UUID(RAW_TEST_UUID)

SQLALCHEMY2 = vparse(sqlalchemy.__version__) >= vparse("2")

UUID_BACKEND_TYPE = {
    "bigquery": "STRING",
    "duckdb": "UUID",
    "exasol": "UUID",
    "flink": "CHAR(36) NOT NULL",
    "sqlite": "text",
    "snowflake": "VARCHAR",
    "trino": "varchar(32)" if SQLALCHEMY2 else "uuid",
    "postgres": "uuid",
    "clickhouse": "Nullable(UUID)",
}

pytestmark = pytest.mark.notimpl(
    ["druid"],
    raises=sqlalchemy.exc.CompileError,
    reason=(
        "No literal value renderer is available for literal value "
        "\"UUID('08f48812-7948-4718-96c7-27fa6a398db6')\" with datatype NULL"
    ),
)


@pytest.mark.notimpl(["impala", "datafusion", "polars"], raises=NotImplementedError)
@pytest.mark.notimpl(["datafusion"], raises=Exception)
def test_uuid_literal(con, backend):
    backend_name = backend.name()

    expr = ibis.literal(RAW_TEST_UUID, type=dt.uuid)
    result = con.execute(expr)

    assert result == TEST_UUID

    with contextlib.suppress(com.OperationNotDefinedError):
        assert con.execute(expr.typeof()) == UUID_BACKEND_TYPE[backend_name]
