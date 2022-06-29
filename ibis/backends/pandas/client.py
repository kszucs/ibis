"""The pandas client implementation."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import toolz
from pandas.api.types import CategoricalDtype, DatetimeTZDtype

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base import Database

infer_pandas_dtype = pd.api.types.infer_dtype


class PandasTable(ops.DatabaseTable):
    pass


class PandasDatabase(Database):
    pass
