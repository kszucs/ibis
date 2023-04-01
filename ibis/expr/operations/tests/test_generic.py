from typing import TypeVar

import pytest

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.collections import frozendict
from ibis.common.patterns import (
    CoercedTo,
    GenericCoercedTo,
    Pattern,
    ValidationError,
)
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


def test_coerced_to_literal():
    p = CoercedTo(ops.Literal)
    one = ops.Literal(1, dt.int8)
    assert p.validate(ops.Literal(1, dt.int8), {}) == one
    assert p.validate(1, {}) == one
    assert p.validate(False, {}) == ops.Literal(False, dt.boolean)

    p = CoercedTo(ops.Literal[dt.Int8])
    assert p.validate(ops.Literal(1, dt.int8), {}) == one

    p = Pattern.from_typehint(ops.Literal[dt.Int8])
    assert p == GenericCoercedTo(
        ops.Literal,
        frozendict({'T': dt.Int8}),
        frozendict({"dtype": CoercedTo(dt.Int8)}),
    )

    one = ops.Literal(1, dt.int16)
    with pytest.raises(ValidationError):
        p.validate(ops.Literal(1, dt.int16), {})

    p = CoercedTo(ops.NullLiteral)
    one = ops.Literal(1, dt.int16)
    assert p.validate(ops.NullLiteral(), {}) == ops.NullLiteral()
    with pytest.raises(ValidationError):
        p.validate(one, {})


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

    # dt.Integer is not instantiable so it will be only used for checking
    # that the produced literal has any integer datatype
    p = Pattern.from_typehint(ops.Value[dt.Integer, ...])
    assert p.validate(1, {}) == one

    # same applies here, the coercion itself will use only the inferred datatype
    # but then the result is checked against the given typehint
    p = Pattern.from_typehint(ops.Value[dt.Int8 | dt.Int16, ...])
    assert p.validate(1, {}) == one
    assert p.validate(128, {}) == ops.Literal(128, dt.int16)

    p1 = Pattern.from_typehint(ops.Value[dt.Int8, ...])
    p2 = Pattern.from_typehint(ops.Value[dt.Int16, Shape.SCALAR])
    assert p1.validate(1, {}) == one
    assert p2.validate(1, {}) == ops.Literal(1, dt.int16)

    p = p1 | p2
    assert p.validate(1, {}) == one
