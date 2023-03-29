import pytest

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.patterns import CoercionError, CoercedTo


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


def test_literal_verify_instance_dtype():
    instance = ops.Literal(1, dt.int8)
    assert instance.__verify__(dt.int8) is True
    assert instance.__verify__(dt.int16) is False



def test_coerced_to():
    p = CoercedTo(ops.Literal)
    one = ops.Literal(1, dt.int8)
    assert p.validate(ops.Literal(1, dt.int8), {}) == one
    assert p.validate(1, {}) == one
    assert p.validate(False, {}) == ops.Literal(False, dt.boolean)

    p = CoercedTo(ops.Literal[dt.int8])
    assert p.validate(ops.Literal(1, dt.int8), {}) == one


from typing import TypeVar
