from __future__ import annotations

import numbers
from typing import Literal

from public import public

import ibis.expr.datatypes as dt
from ibis.common.annotations import attribute
from ibis.expr import rules as rlz
from ibis.expr.operations.core import Value


@public
class Bucket(Value):
    arg = rlz.column(rlz.numeric)
    buckets = rlz.tuple_of(rlz.instance_of(numbers.Real))
    closed: Literal['left', 'right'] = 'left'
    close_extreme: bool = True
    include_under: bool = False
    include_over: bool = False
    output_shape = rlz.Shape.COLUMNAR

    @attribute.default
    def output_dtype(self):
        return dt.infer(self.nbuckets)

    def __init__(self, buckets, include_under, include_over, **kwargs):
        if not buckets:
            raise ValueError('Must be at least one bucket edge')
        elif len(buckets) == 1:
            if not include_under or not include_over:
                raise ValueError(
                    'If one bucket edge provided, must have '
                    'include_under=True and include_over=True'
                )
        super().__init__(
            buckets=buckets,
            include_under=include_under,
            include_over=include_over,
            **kwargs,
        )

    @property
    def nbuckets(self):
        return len(self.buckets) - 1 + self.include_over + self.include_under
