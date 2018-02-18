import pytest
import six
import ibis
from contextlib import contextmanager
from ibis.common import IbisTypeError
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.expr.datatypes as dt
from ibis.expr import rules
import ibis.expr.rlz as rlz
from toolz import identity


def mayraise(error):
    """Wrapper around pytest.raises to support None."""
    if type(error) is type and issubclass(error, Exception):
        return pytest.raises(error)
    else:
        @contextmanager
        def not_raises():
            try:
                yield
            except Exception as e:
                raise e
        return not_raises()


# def test_validator():
#     # TODO
#     pass


@pytest.mark.parametrize(('value', 'expected'), [
    (dt.int32, dt.int32),
    ('int64', dt.int64),
    ('array<string>', dt.Array(dt.string)),
    ('exception', IbisTypeError),
    ('array<cat>', IbisTypeError),
    (int, IbisTypeError),
    ([float], IbisTypeError)
])
def test_datatype(value, expected):
    with mayraise(expected):
        assert rlz.datatype(value) == expected


@pytest.mark.parametrize(('klass', 'value', 'expected'), [
    (int, 32, 32),
    (ir.TableExpr, object, IbisTypeError),
    (six.string_types, 'foo', 'foo'),
    (dt.Integer, dt.int8, dt.int8),
    (ir.IntegerValue, 4, IbisTypeError)
])
def test_instanceof(klass, value, expected):
    with mayraise(expected):
        assert rlz.instanceof(klass, value) == expected


@pytest.mark.parametrize(('dtype', 'value', 'expected'), [
    (dt.int32, 26, ir.literal(26)),
    (dt.int32, dict(), IbisTypeError),
    (dt.string, 'bar', ir.literal('bar')),
    (dt.string, 1, IbisTypeError),
    (dt.Array(dt.float), [3.4, 5.6], ir.literal([3.4, 5.6])),
    (dt.Array(dt.float), ['s'], IbisTypeError),
    (dt.Map(dt.string, dt.Array(dt.boolean)),
     {'a': [True, False], 'b': [True]},
     ir.literal({'a': [True, False], 'b': [True]})),
    (dt.Map(dt.string, dt.Array(dt.boolean)),
     {'a': [True, False], 'b': ['B']},
     IbisTypeError)
])
def test_value(dtype, value, expected):
    with mayraise(expected):
        result = rlz.value(dtype, value)
        assert result.equals(expected)


@pytest.mark.parametrize(('validator', 'value', 'expected'), [
    (rlz.optional(identity), None, None),
    (rlz.optional(identity), 'three', 'three'),
    (rlz.optional(identity, default=1), None, 1),
    (rlz.optional(identity, default=lambda: 8), 'cat', 'cat'),
    (rlz.optional(identity, default=lambda: 8), None, 8),
    (rlz.optional(rlz.instanceof(int), default=''), None, IbisTypeError),
    (rlz.optional(rlz.instanceof(int), default=11), None, 11),
    (rlz.optional(rlz.instanceof(int)), None, None),
    (rlz.optional(rlz.instanceof(int)), 18, 18),
    (rlz.optional(rlz.instanceof(int)), 'lynx', IbisTypeError),
    (rlz.optional(rlz.instanceof(str)), 'caracal', 'caracal'),
])
def test_optional(validator, value, expected):
    with mayraise(expected):
        assert validator(value) == expected


@pytest.mark.parametrize(('values', 'value', 'expected'), [
    (['a', 'b'], 'a', 'a'),
    (('a', 'b'), 'b', 'b'),
    ({'a', 'b', 'c'}, 'c', 'c'),
    (['a', 'b'], 'c', IbisTypeError),
    ({'a', 'b', 'c'}, 'd', IbisTypeError),
    ([1, 2, 'f'], 'f', 'f'),
    ({'a': 1, 'b': 2}, 'a', 1),
    ({'a': 1, 'b': 2}, 'b', 2),
    ({'a': 1, 'b': 2}, 'c', IbisTypeError)
])
def test_isin(values, value, expected):
    with mayraise(expected):
        assert rlz.isin(values, value) == expected


# def test_interval():
#     pass


# def test_column():
#     pass


# def test_scalar():
#     pass


# def test_enum_validator():
#     enum = pytest.importorskip('enum')

#     class Foo(enum.Enum):
#         a = 1
#         b = 2

#     class Bar(enum.Enum):
#         a = 1
#         b = 2

#     class MyOp(ops.Node):

#         input_type = [rules.enum(Foo, name='value')]

#         def __init__(self, value):
#             super(MyOp, self).__init__([value])

#         def output_type(self):
#             return MyExpr

#     assert MyOp(2) is not None
#     assert MyOp(Foo.b) is not None

#     with pytest.raises(IbisTypeError):
#         MyOp(3)

#     with pytest.raises(IbisTypeError):
#         MyOp(Bar.a)

#     op = MyOp(Foo.a)
#     assert op._validate_args(op.args) == [Foo.a]

#     op = MyOp(2)
#     assert op._validate_args(op.args) == [Foo.b]


# def test_duplicate_enum():
#     enum = pytest.importorskip('enum')

#     class Dup(enum.Enum):
#         a = 1
#         b = 1
#         c = 2

#     class MyOp(ops.Node):

#         input_type = [rules.enum(Dup, name='value')]

#         def __init__(self, value):
#             super(MyOp, self).__init__([value])

#         def output_type(self):
#             return MyExpr

#     with pytest.raises(IbisTypeError):
#         MyOp(1)

#     assert MyOp(2) is not None


# doc were nowhere used
# def test_argument_docstring():
#     doc = 'A wonderful integer'

#     class MyExpr(ir.Expr):
#         pass

#     class MyOp(ops.ValueOp):

#         foo = rlz.integer
#         input_type = [rules.integer(name='foo', doc=doc)]

#         def output_type(self):
#             return MyExpr

#     op = MyOp(1)
#     assert type(op).foo.__doc__ == doc
