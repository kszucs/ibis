"""Execution rules for generic ibis operations."""

import datetime
import numbers

import numpy as np
import pandas as pd

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis import util
from ibis.common.exceptions import UnboundExpressionError
from ibis.expr.operations.generic import TableColumn

from .. import Backend as PandasBackend
from ..core import execute_node

register = execute_node.register


@execute_node.register(ops.StringLength, pd.Series)
def execute_string_length_series(op, data, **kwargs):
    return data.str.len().astype('int32')
