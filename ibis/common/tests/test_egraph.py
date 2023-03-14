from typing import Any

from rich.pretty import pprint
from ibis.util import promote_tuple
# from pprint import pprint
import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.egraph import EGraph, Pattern, Rewrite, Variable
from ibis.common.grounds import Annotable, Concrete
from ibis.common.graph import Node

one = ibis.literal(1)
two = one * 2
two_ = one + one
three = one + two
six = three * two_
seven = six + 1
seven_ = seven * 1
eleven = seven_ + 4

a, b, c = Variable('a'), Variable('b'), Variable('c')

# e = seven_
# for i in range(10):
#     e = e + two
#     e = e - two * 1

# seven_ = e


def test_simple():
    op = eleven.op()
    print()
    pprint(op)

    eg = EGraph()
    eg.add(op)

    # p = ops.Add[a, ops.Multiply[b, ops.Literal[c, dt.int8]]]
    # print(eg.match(p))

    # p = ops.Multiply[a, 1]
    # result = eg.match(p)
    # print(result)

    # print(ops.Multiply[a, 1] >> a)

    # r = ops.Multiply[a, 1] >> ops.Multiply[1, a]
    # r2 = ops.Multiply[a, 1] >> a
    # eg.apply(r)

    # print(eg)
    # for i in range(1000):
    #     eg.apply([r, r2])
    # print(eg)
    # print(eg._etables)
    p = ops.Multiply[a, ops.Literal[1, dt.int8]]

    r3 = ops.Multiply[a, ops.Literal[1, dt.int8]] >> a
    print(eg.match(ops.Multiply[a, ops.Literal[1, dt.int8]]))

    eg.apply(r3)
    # eg.apply(r3)
    # eg.apply(r3)
    # eg.apply(r3)

    print()
    pprint(eg.extract(op))


class Base(Concrete, Node):
    def __class_getitem__(self, args):
        args = promote_tuple(args)
        return Pattern(self, args)


class Lit(Base):
    value: Any


class Add(Base):
    x: Any
    y: Any


class Mul(Base):
    x: Any
    y: Any


# Rewrite rules
a, b = Variable("a"), Variable("b")
rules = [
    Add[a, b] >> Add[b, a],
    Mul[a, b] >> Mul[b, a],
    Add[a, Lit[0]] >> a,
    Mul[a, Lit[0]] >> Lit[0],
    Mul[a, Lit[1]] >> a
]


def simplify(expr, rules, iters=7):
    egraph = EGraph()
    egraph.add(expr)

    for i in range(iters):
        egraph.apply(rules)

    print("================")
    print(egraph._etables)

    best = egraph.extract(expr)
    return best


def test_simple_1():
    assert simplify(Mul(Lit(0), Lit(42)), rules) == Lit(0)


def test_simple_2():
    assert simplify(Add(Lit(0), Mul(Lit(1), Lit(2))), rules, iters=1) == Lit(2)
