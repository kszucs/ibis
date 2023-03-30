import pytest

from ibis.common.enums import IntervalUnit
from ibis.common.patterns import CoercedTo

interval_units = pytest.mark.parametrize(
    ["singular", "plural", "short"],
    [
        ("year", "years", "Y"),
        ("quarter", "quarters", "Q"),
        ("month", "months", "M"),
        ("week", "weeks", "W"),
        ("day", "days", "D"),
        ("hour", "hours", "h"),
        ("minute", "minutes", "m"),
        ("second", "seconds", "s"),
        ("millisecond", "milliseconds", "ms"),
        ("microsecond", "microseconds", "us"),
        ("nanosecond", "nanoseconds", "ns"),
    ],
)


@interval_units
def test_interval_units(singular, plural, short):
    u = IntervalUnit[singular.upper()]
    assert u.singular == singular
    assert u.plural == plural
    assert u.short == short


@interval_units
def test_interval_unit_coercions(singular, plural, short):
    u = IntervalUnit[singular.upper()]
    v = CoercedTo(IntervalUnit)
    assert v.validate(singular, {}) == u
    assert v.validate(plural, {}) == u
    assert v.validate(short, {}) == u


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("HH24", "h"),
        ("J", "D"),
        ("MI", "m"),
        ("SYYYY", "Y"),
        ("YY", "Y"),
        ("YYY", "Y"),
        ("YYYY", "Y"),
    ],
)
def test_interval_unit_aliases(alias, expected):
    v = CoercedTo(IntervalUnit)
    assert v.validate(alias, {}) == IntervalUnit(expected)
