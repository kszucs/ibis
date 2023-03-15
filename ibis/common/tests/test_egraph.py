from typing import Any

from rich.pretty import pprint

# from pprint import pprint
import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.egraph import EGraph, Pattern, Rewrite, Variable
from ibis.common.graph import Node
from ibis.common.grounds import Annotable, Concrete
from ibis.util import promote_tuple

one = ibis.literal(1)
two = one * 2
two_ = one + one
three = one + two
six = three * two_
seven = six + 1
seven_ = seven * 1
eleven = seven_ + 4

a, b, c = Variable('a'), Variable('b'), Variable('c')
x, y, z = Variable('x'), Variable('y'), Variable('z')


def test_simple():
    op = eleven.op()
    eg = EGraph()
    eg.add(op)
    print(eg.match(ops.Multiply[a, ops.Literal[1, dt.int8]]))

    r3 = ops.Multiply[a, ops.Literal[1, dt.int8]] >> a
    eg.apply(r3)

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
    Mul[a, Lit[1]] >> a,
]


def simplify(expr, rules, iters=7):
    egraph = EGraph()
    egraph.add(expr)
    egraph.run(rules, iters)
    print(egraph._etables)
    print(egraph._eclasses)
    best = egraph.extract(expr)
    return best


from pprint import pprint


def is_equal(a, b, rules, iters=7):
    egraph = EGraph()
    id_a = egraph.add(a)
    id_b = egraph.add(b)
    egraph.run(rules, iters)
    #pprint(egraph._etables)
    #pprint(egraph._eclasses)
    return egraph.equivalent(id_a, id_b)


def test_simple_1():
    assert simplify(Mul(Lit(0), Lit(42)), rules) == Lit(0)


def test_simple_2():
    assert simplify(Add(Lit(0), Mul(Lit(1), Lit(2))), rules, iters=1) == Lit(2)


def test_simple_3():
    rules = [
        Mul[a, b] >> Mul[b, a],
        Mul[a, Lit[1]] >> a,
    ]

    node = Mul(Lit(2), Mul(Lit(1), Lit(3)))
    expected = Mul(Lit(2), Lit(3))
    assert simplify(node, rules, iters=20000) == expected


def test_math_associate_adds():
    math_rules = [
        Add[a, b] >> Add[b, a],
        Add[a, Add[b, c]] >> Add[Add[a, b], c]
    ]

    expr_a = Add(1, Add(2, Add(3, Add(4, Add(5, Add(6, 7))))))
    expr_b = Add(7, Add(6, Add(5, Add(4, Add(3, Add(2, 1))))))
    assert is_equal(expr_a, expr_b, math_rules, iters=500)

    expr_a = Add(6, Add(Add(1, 5), Add(0, Add(4, Add(2, 3)))))
    expr_b = Add(6, Add(Add(4, 5), Add(Add(0, 2), Add(3, 1))))
    assert is_equal(expr_a, expr_b, math_rules, iters=500)


def replace_add(egraph, id, subst):
    node = egraph.extract(id)
    id = egraph.add(node)
    return id



def test_dynamic_rewrite():
    rules = [
        Rewrite(Add[x, Mul[z, y]], replace_add, name="replace-add"),
    ]

    simplify(Add(1, Mul(2, 3)), rules) == Add(1, Mul(2, 3))


def test_dynamic_condition():
    pass
