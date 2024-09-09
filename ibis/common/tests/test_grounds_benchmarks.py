from __future__ import annotations

import pytest

from koerce import attribute, Annotable
from ibis.common.collections import frozendict


pytestmark = pytest.mark.benchmark


class MyObject(Annotable, immutable=True, hashable=True):
    a: int
    b: str
    c: tuple[int, ...]
    d: frozendict[str, int]

    @attribute
    def e(self):
        return self.a * 2

    @attribute
    def f(self):
        return self.b * 2

    @attribute
    def g(self):
        return self.c * 2


def test_concrete_construction(benchmark):
    benchmark(MyObject, 1, "2", c=(3, 4), d=frozendict(e=5, f=6))


def test_concrete_isinstance(benchmark):
    def check(obj):
        for _ in range(100):
            assert isinstance(obj, MyObject)
            assert isinstance(obj, Concrete)
            assert isinstance(obj, Annotable)

    obj = MyObject(1, "2", c=(3, 4), d=frozendict(e=5, f=6))
    benchmark(check, obj)
