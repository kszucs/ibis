import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.pandas import Backend
from ibis.backends.pandas.core import execute, execute_node


def test_execute_parameter_only():
    param = ibis.param('int64')
    result = execute(param, params={param: 42})
    assert result == 42
