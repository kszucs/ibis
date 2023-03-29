import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.patterns import CoercedTo, NoMatch

# from ibis.expr.patterns import Literal

one = ops.Literal(1, dt.int64)


def test_literal_coercion():
    assert ops.Literal.__coerce__(1) == ops.Literal(1, dt.int8)


# def test_literal_pattern():
#     p = Literal(dt.int64)

#     assert p.match(1, {}) is NoMatch
#     assert p.match(1.0, {}) is NoMatch
#     assert p.match(one, {}) == one
