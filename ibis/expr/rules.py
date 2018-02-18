# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import six
import sys
import operator

from collections import Counter

from ibis.common import IbisTypeError
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.common as com
import ibis.util as util


class BinaryPromoter(object):
    # placeholder for type promotions for basic binary arithmetic

    def __init__(self, left, right, op):
        self.args = [left, right]
        self.left = left
        self.right = right
        self.op = op

        self._check_compatibility()

    def get_result(self):
        promoted_type = self._get_type()
        return shape_like_args(self.args, promoted_type)

    def _get_type(self):
        if util.any_of(self.args, ir.DecimalValue):
            return _decimal_promoted_type(self.args)
        elif util.any_of(self.args, ir.FloatingValue):
            if util.any_of(self.args, ir.DoubleValue):
                return 'double'
            else:
                return 'float'
        elif util.all_of(self.args, ir.IntegerValue):
            return self._get_int_type()
        elif self.left.type().equals(self.right.type()):
            return self.left.type()
        else:
            raise NotImplementedError(
                'Operands {}, {} not supported for binary operation {}'.format(
                    type(self.left).__name__, type(self.right).__name__,
                    self.op.__name__
                )
            )

    def _get_int_type(self):
        import ibis.expr.operations as ops

        deps = [x.op() for x in self.args]

        if util.all_of(deps, ops.Literal):
            return _smallest_int_containing(
                [self.op(deps[0].value, deps[1].value)])
        elif util.any_of(deps, ops.Literal):
            if isinstance(deps[0], ops.Literal):
                val = deps[0].value
                atype = self.args[1].type()
            else:
                val = deps[1].value
                atype = self.args[0].type()
            return _int_one_literal_promotion(atype, val, self.op)
        else:
            return _int_bounds_promotion(self.left.type(),
                                         self.right.type(), self.op)

    def _check_compatibility(self):
        if (util.any_of(self.args, ir.StringValue) and
                not util.all_of(self.args, ir.StringValue)):
            raise TypeError('String and non-string incompatible')


class IntervalPromoter(BinaryPromoter):
    """Infers the output type of the binary interval operation

    This is a slightly modified version of BinaryPromoter, it converts
    back and forth between the interval and its inner value.

    This trick reuses the numeric type promotion logics.
    Any non-integer output type raises a TypeError.
    """

    def __init__(self, left, right, op):
        left_type = left.type()
        value_type = shape_like(left, left_type.value_type)
        self.unit = left_type.unit
        super(IntervalPromoter, self).__init__(value_type(left), right, op)

    def get_result(self):
        promoted_value_type = self._get_type()
        promoted_type = dt.Interval(self.unit, promoted_value_type)
        return shape_like_args(self.args, promoted_type)


def _decimal_promoted_type(args):
    max_precision = max_scale = ~sys.maxsize
    for arg in args:
        if isinstance(arg, ir.DecimalValue):
            max_precision = max(max_precision, arg.meta.precision)
            max_scale = max(max_scale, arg.meta.scale)
    return dt.Decimal(max_precision, max_precision)


class PowerPromoter(BinaryPromoter):

    def __init__(self, left, right):
        super(PowerPromoter, self).__init__(left, right, operator.pow)

    def _get_type(self):
        if util.any_of(self.args, ir.FloatingValue):
            if util.any_of(self.args, ir.DoubleValue):
                return 'double'
            else:
                return 'float'
        elif util.any_of(self.args, ir.DecimalValue):
            return _decimal_promoted_type(self.args)
        elif util.all_of(self.args, ir.IntegerValue):
            return 'double'
        else:
            raise NotImplementedError(
                'Operands {}, {} not supported for binary operation {}'.format(
                    type(self.left).__name__, type(self.right).__name__,
                    self.op.__name__
                )
            )


def highest_precedence_type(exprs):
    # Return the highest precedence type from the passed expressions. Also
    # verifies that there are valid implicit casts between any of the types and
    # the selected highest precedence type
    if not exprs:
        raise ValueError('Must pass at least one expression')

    expr_dtypes = {expr.type() for expr in exprs}
    return dt.highest_precedence(expr_dtypes)


def _int_bounds_promotion(ltype, rtype, op):
    lmin, lmax = ltype.bounds
    rmin, rmax = rtype.bounds

    values = [op(lmin, rmin), op(lmin, rmax),
              op(lmax, rmin), op(lmax, rmax)]

    return _smallest_int_containing(values, allow_overflow=True)


def _int_one_literal_promotion(atype, lit_val, op):
    amin, amax = atype.bounds
    bound_type = _smallest_int_containing([op(amin, lit_val),
                                           op(amax, lit_val)],
                                          allow_overflow=True)
    # In some cases, the bounding type might be int8, even though neither of
    # the types are that small. We want to ensure the containing type is _at
    # least_ as large as the smallest type in the expression
    return _largest_int([bound_type, atype])


def _smallest_int_containing(values, allow_overflow=False):
    containing_types = [dt.infer(x, allow_overflow=allow_overflow)
                        for x in values]
    return _largest_int(containing_types)


def _largest_int(int_types):
    nbytes = max(t._nbytes for t in int_types)
    return dt.validate_type('int%d' % (8 * nbytes))


# ----------------------------------------------------------------------
# Input / output type rules and validation


def shape_like(arg, out_type):
    out_type = dt.validate_type(out_type)
    if isinstance(arg, ir.ScalarExpr):
        return out_type.scalar_type()
    else:
        return out_type.array_type()


def shape_like_args(args, out_type):
    out_type = dt.validate_type(out_type)
    if util.any_of(args, ir.ColumnExpr):
        return out_type.array_type()
    else:
        return out_type.scalar_type()


# class Argument(object):

#     """

#     """

#     def __init__(self, name=None, default=None, optional=False,
#                  validator=None, doc=None, as_value_expr=None):
#         self.name = name
#         self.default = default
#         self.optional = optional
#         self.validator = validator
#         self.doc = doc
#         self.as_value_expr = as_value_expr or ir.literal

#     def validate(self, args, i):
#         arg = args[i]

#         if self.validator is not None:
#             arg = args[i] = self.validator(arg)

#         if arg is None:
#             if not self.optional:
#                 return self.as_value_expr(self.default)
#             elif self.optional:
#                 return arg

#         return self._validate(args, i)

#     def _validate(self, args, i):
#         raise NotImplementedError


# def _to_argument(val):
#     if isinstance(val, dt.DataType):
#         val = value_typed_as(val)
#     elif not isinstance(val, Argument):
#         val = val()
#     return val


# class TypeSignature(object):

#     def __init__(self, type_specs):
#         types = []

#         for val in type_specs:
#             val = _to_argument(val)
#             types.append(val)

#         self.types = types

#     def __repr__(self):
#         types = '\n    '.join('arg {0}: {1}'.format(i, repr(x))
#                               for i, x in enumerate(self.types))
#         return '{0}\n    {1}'.format(type(self), types)

#     def validate(self, args):
#         n, k = len(args), len(self.types)
#         k_required = len([x for x in self.types if not x.optional])
#         if k != k_required:
#             if n < k_required:
#                 raise com.IbisError('Expected at least {0} args, got {1}'
#                                     .format(k, k_required))
#         elif n != k:
#             raise com.IbisError('Expected {0} args, got {1}'.format(k, n))

#         if n < k:
#             args = list(args) + [t.default for t in self.types[n:]]

#         return self._validate(args, self.types)

#     def _validate(self, args, types):
#         clean_args = list(args)
#         for i, validator in enumerate(types):
#             try:
#                 clean_args[i] = validator.validate(clean_args, i)
#             except IbisTypeError as e:
#                 exc = e.args[0]
#                 msg = ('Argument {0}: {1}'.format(i, exc) +
#                        '\nArgument was: {0}'.format(ir._safe_repr(args[i])))
#                 raise IbisTypeError(msg)

#         return clean_args


# class VarArgs(TypeSignature):

#     def __init__(self, arg_type, min_length=1):
#         self.arg_type = _to_argument(arg_type)
#         self.min_length = min_length

#     def __repr__(self):
#         return '{0}\n    {1}'.format(type(self), repr(self.arg_type))

#     def validate(self, args):
#         n, k = len(args), self.min_length
#         if n < k:
#             raise com.IbisError('Expected at least {0} args, got {1}'
#                                 .format(k, n))

#         return self._validate(args, [self.arg_type] * n)


# varargs = VarArgs


def scalar_output(rule):
    def f(self):
        if isinstance(rule, dt.DataType):
            t = rule
        else:
            t = dt.validate_type(rule(self))
        return t.scalar_type()
    return f


def array_output(rule):
    def f(self):
        if isinstance(rule, dt.DataType):
            t = rule
        else:
            t = dt.validate_type(rule(self))
        return t.array_type()
    return f


def shape_like_flatargs(out_type):

    def output_type(self):
        flattened = list(self.flat_args())
        return shape_like_args(flattened, out_type)

    return output_type


def shape_like_arg(i, out_type):

    def output_type(self):
        return shape_like(self.args[i], out_type)

    return output_type


def type_of_arg(i):

    def output_type(self):
        return self.args[i]._factory

    return output_type


# def signature(types):
#     if isinstance(types, TypeSignature):
#         return types

#     return TypeSignature(types)


# class ValueArgument(Argument):

#     def _validate(self, args, i):
#         arg = args[i]
#         if not isinstance(arg, ir.Expr):
#             arg = args[i] = self.as_value_expr(arg)

#         return arg


# class CastIfDecimal(ValueArgument):

#     def __init__(self, ref_j, **arg_kwds):
#         super(CastIfDecimal, self).__init__(**arg_kwds)
#         self.ref_j = ref_j

#     def _validate(self, args, i):
#         super(CastIfDecimal, self)._validate(args, i)

#         ref_arg = args[self.ref_j]
#         if isinstance(ref_arg, ir.DecimalValue):
#             return args[i].cast(ref_arg.type())

#         return args[i]


# cast_if_decimal = CastIfDecimal


# def value_typed_as(types, **arg_kwds):
#     fail_message = 'Arg was not in types {0}'.format(repr(types))
#     return ValueTyped(types, fail_message, **arg_kwds)


# def column(value_type=None, **arg_kwds):
#     array_checker = ValueTyped(ir.ColumnExpr, 'not a column expr', **arg_kwds)
#     if value_type is None:
#         return array_checker
#     else:
#         return MultipleTypes([array_checker, value_type], **arg_kwds)


# def scalar(value_type=None, **arg_kwds):
#     scalar_checker = ValueTyped(ir.ScalarExpr, 'not a scalar expr', **arg_kwds)
#     if value_type is None:
#         return scalar_checker
#     else:
#         return MultipleTypes([scalar_checker, value_type], **arg_kwds)


# def collection(**arg_kwds):
#     return ValueTyped((ir.ColumnExpr, ir.TableExpr), 'not a collection',
#                       **arg_kwds)


# def value(**arg_kwds):
#     return ValueTyped(ir.ValueExpr, 'not a value expr', **arg_kwds)


# class Number(ValueTyped):

#     def __init__(self, allow_boolean=True, **arg_kwds):
#         super(Number, self).__init__(
#             ir.NumericValue, 'not numeric', **arg_kwds
#         )
#         self.allow_boolean = allow_boolean

#     def _validate(self, args, i):
#         arg = super(Number, self)._validate(args, i)

#         if isinstance(arg, ir.BooleanValue) and not self.allow_boolean:
#             raise IbisTypeError('not implemented for boolean values')

#         return arg


# number = Number


# def one_of(args, **arg_kwds):
#     return OneOf(args, **arg_kwds)


# temporal = one_of((dt.timestamp, dt.date, dt.time))
# TODO CREATE ERROR MESSAGES


# class Enum(Argument):
#     def __init__(self, enum, **arg_kwds):
#         super(Enum, self).__init__(**arg_kwds)
#         self.enum = enum

#     def _validate(self, args, i):
#         arg = args[i]

#         # if our passed value wasn't specified directly from the enum
#         if not isinstance(arg, self.enum):
#             value_set = {}
#             for key, value in self.enum.__members__.items():
#                 value_set.setdefault(value.value, []).append(key)

#             # not in the value_set, so can't be valid
#             if arg not in value_set:
#                 raise IbisTypeError(
#                     ('Value {} is not a member of the {} enum, '
#                      'whose values are {}').format(
#                          arg, self.enum.__name__, list(self.enum)
#                     )
#                 )

#             # if it's in the value set and the value set has duplicates, then
#             # we can't validate it because we don't know which one the user
#             # meant
#             if len(value_set[arg]) > 1:
#                 raise IbisTypeError(
#                     (
#                         'Value {0} is a member of {1}, but {1} is not unique. '
#                         'Please explicitly pass the desired enum attribute.'
#                     ).format(arg, self.enum.__name__)
#                 )
#             return self.enum[value_set[arg].pop()]
#         return arg


# enum = Enum


def comparable(left, right):
    return ir.castable(left, right) or ir.castable(right, left)
