from typing import Optional

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.exceptions import IbisTypeError, InputTypeError
from ibis.common.grounds import Concrete
from ibis.common.patterns import CoercionError, Matcher, MatchError, NoMatch, Pattern

# USE THESE FROM the __coerce__ methods
# class DataType(Pattern):
#     def match(self, value, context):
#         if not isinstance(value, dt.DataType):
#             return NoMatch
#         return value

# extend concrete in the future
# class Value(Matcher):
#     __slots__ = ("dtype", "shape")

#     def __init__(self, dtype=None, shape=None):
#         self.dtype = dt.dtype(dtype) if dtype is not None else None
#         self.shape = shape

#     def match(self, value, context):


# class Literal(Concrete, Pattern):
#     dtype: Optional[dt.DataType] = None

#     def match(self, value, context):
#         has_explicit = self.dtype is not None

#         try:
#             inferred_dtype = dt.infer(value)
#         except InputTypeError:
#             has_inferred = False
#         else:
#             has_inferred = True

#         if has_explicit and has_inferred:
#             # ensure type correctness: check that the inferred dtype is
#             # implicitly castable to the explicitly given dtype and value
#             if not dt.castable(inferred_dtype, self.dtype, value=value):
#                 return NoMatch
#             dtype = self.dtype
#         elif has_explicit:
#             dtype = self.dtype
#         elif has_inferred:
#             dtype = inferred_dtype
#         else:
#             raise MatchError(
#                 f"The datatype of value {value!r} cannot be inferred, try "
#                 "passing it explicitly with the `type` keyword."
#             )

#         if dtype.is_null():
#             return ops.NullLiteral()

#         value = dt.normalize(dtype, value)
#         return ops.Literal(value, dtype=dtype)


# class Value(Concrete, Pattern):
#     dtype: Optional[dt.DataType] = None
#     shape: Optional[dt.Shape] = None

#     def match(self, value, context):
#         # if isinstance(arg, Deferred):
#         #     raise com.IbisTypeError(
#         #         "Deferred input is not allowed, try passing a lambda function instead. "
#         #         "For example, instead of writing `f(_.a)` write `lambda t: f(t.a)`"
#         #     )

#         if not isinstance(arg, ops.Value):
#             # coerce python literal to ibis literal
#             arg = literal(None, arg)

#         if dtype is None:
#             # no datatype restriction
#             return arg
#         elif isinstance(dtype, type):
#             # dtype class has been specified like dt.Interval or dt.Decimal
#             if not issubclass(dtype, dt.DataType):
#                 raise com.IbisTypeError(
#                     f"Datatype specification {dtype} is not a subclass dt.DataType"
#                 )
#             elif isinstance(arg.output_dtype, dtype):
#                 return arg
#             else:
#                 raise com.IbisTypeError(
#                     f'Given argument with datatype {arg.output_dtype} is not '
#                     f'subtype of {dtype}'
#                 )
#         elif isinstance(dtype, (dt.DataType, str)):
#             # dtype instance or string has been specified and arg's dtype is
#             # implicitly castable to it, like dt.int8 is castable to dt.int64
#             dtype = dt.dtype(dtype)
#             # retrieve literal values for implicit cast check
#             value = getattr(arg, 'value', None)
#             if dt.castable(arg.output_dtype, dtype, value=value):
#                 return arg
#             else:
#                 raise com.IbisTypeError(
#                     f'Given argument with datatype {arg.output_dtype} is not '
#                     f'implicitly castable to {dtype}'
#                 )
#         else:
#             raise com.IbisTypeError(f'Invalid datatype specification {dtype}')

#         # if self.dtype is None:
#         #     return value
#         # else:
#         #     return Literal(self.dtype).match(value, context
