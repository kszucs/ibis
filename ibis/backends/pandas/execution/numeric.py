"""Execution rules for generic ibis operations."""

import datetime
import numbers
from multiprocessing.sharedctypes import Value

import numpy as np
import pandas as pd

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

from ..core import execute_node

register = execute_node.register


@register(ops.Multiply, pd.Series, int)
def execute_pandas_join(op, left, right, **kwargs):
    return left * right
