"""Execution rules for generic ibis operations."""

import datetime
import numbers

import numpy as np
import pandas as pd

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

from ..core import execute_node

register = execute_node.register


integer_types = np.integer, int
floating_types = (numbers.Real,)
numeric_types = integer_types + floating_types
boolean_types = bool, np.bool_
fixed_width_types = numeric_types + boolean_types
date_types = (datetime.date,)
time_types = (datetime.time,)
timestamp_types = pd.Timestamp, datetime.datetime, np.datetime64
timedelta_types = pd.Timedelta, datetime.timedelta, np.timedelta64
temporal_types = date_types + time_types + timestamp_types + timedelta_types
scalar_types = fixed_width_types + temporal_types
simple_types = scalar_types + (str, type(None))


# By default return the literal value
@register(ops.Literal, object, dt.DataType)
def execute_literal_value_datatype(op, value, datatype, **kwargs):
    return value


# Because True and 1 hash to the same value, if we have True or False in scope
# keys while executing anything that should evaluate to 1 or 0 evaluates to
# True or False respectively. This is a hack to work around that by casting the
# bool to an integer.
@register(ops.Literal, object, dt.Integer)
def execute_literal_any_integer_datatype(op, value, datatype, **kwargs):
    return int(value)


@register(ops.Literal, object, dt.Boolean)
def execute_literal_any_boolean_datatype(op, value, datatype, **kwargs):
    return bool(value)


@register(ops.Literal, object, dt.Floating)
def execute_literal_any_floating_datatype(op, value, datatype, **kwargs):
    return float(value)


@register(ops.Literal, object, dt.Array)
def execute_literal_any_array_datatype(op, value, datatype, **kwargs):
    return np.array(value)


@register(ops.Literal, dt.DataType)
def execute_literal_datatype(op, datatype, **kwargs):
    return op.value


@register(ops.Literal, timedelta_types + (str,) + integer_types, dt.Interval)
def execute_interval_literal(op, value, dtype, **kwargs):
    return pd.Timedelta(value, dtype.unit)


@register(ops.Alias, pd.Series, str)
def execute_alias(op, obj, name, **kwargs):
    return obj.rename(name)
