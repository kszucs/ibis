from typing import TypeVar

import pytest

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.patterns import CoercedTo, ValidationError, coerce, GenericCoercedTo, Pattern, InstanceOf
from ibis.common.typing import CoercionError
from ibis.common.collections import frozendict

from ibis.expr.rules import Shape
# TODO(kszucs): actually we should only allow datatype classes not instances


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (1, dt.int8),
        (1.0, dt.double),
        (True, dt.boolean),
        ("foo", dt.string),
        (b"foo", dt.binary),
        ((1, 2), dt.Array(dt.int8)),
    ],
)
def test_literal_coercion_type_inference(value, dtype):
    assert ops.Literal.__coerce__(value) == ops.Literal(value, dtype)
    assert ops.Literal.__coerce__(value, dtype) == ops.Literal(value, dtype)


def test_literal_coercion_not_castable():
    msg = "Value 1 cannot be safely coerced to `string`"
    with pytest.raises(CoercionError, match=msg):
        ops.Literal.__coerce__(1, dt.string)


# def test_literal_verify_instance_dtype():
#     instance = ops.Literal(1, dt.int8)
#     assert instance.__verify__(dt.int8) is True
#     assert instance.__matches__(dt.int16) is False


def test_coerced_to_literal():
    p = CoercedTo(ops.Literal)
    one = ops.Literal(1, dt.int8)
    assert p.validate(ops.Literal(1, dt.int8), {}) == one
    assert p.validate(1, {}) == one
    assert p.validate(False, {}) == ops.Literal(False, dt.boolean)

    p = CoercedTo(ops.Literal[dt.Int8])
    assert p.validate(ops.Literal(1, dt.int8), {}) == one

    p = Pattern.from_typehint(ops.Literal[dt.Int8])
    assert p == GenericCoercedTo(ops.Literal, frozendict({"dtype": CoercedTo(dt.Int8)}))

    one = ops.Literal(1, dt.int16)
    with pytest.raises(ValidationError):
        p.validate(ops.Literal(1, dt.int16), {})

    p = CoercedTo(ops.NullLiteral)
    one = ops.Literal(1, dt.int16)
    assert p.validate(ops.NullLiteral(), {}) == ops.NullLiteral()
    with pytest.raises(ValidationError):
        p.validate(one, {})


# TODO(kszucs): test filtering for other dtypes and shapes
def test_coerced_to_value():
    one = ops.Literal(1, dt.int8)

    p = Pattern.from_typehint(ops.Value)
    assert p.validate(1, {}) == one

    p = Pattern.from_typehint(ops.Value[dt.Int8, ...])
    assert p.validate(1, {}) == one

    p = Pattern.from_typehint(ops.Value[dt.Int8, Shape.SCALAR])
    assert p.validate(1, {}) == one

    p = Pattern.from_typehint(ops.Value[dt.Int8, Shape.COLUMNAR])
    with pytest.raises(ValidationError):
        p.validate(1, {})


    # p = CoercedTo(ops.Value[dt.Int8 | dt.Int16, ...])
    # assert p.validate(1, {}) == one


# perhaps use Any instead of ...
# ops.Value[dt.Int8 | dt.Int16, ...]
# ops.Value[dt.Int8, ...] | ops.Value[dt.Int16, ...]
