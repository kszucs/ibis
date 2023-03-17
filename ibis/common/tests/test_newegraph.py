import functools
import itertools
from pprint import pprint
from typing import Any, Tuple

import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.collections import DisjointSet
from ibis.common.graph import Node
from ibis.common.grounds import Concrete
from ibis.common.newegraph import EGraph, ENode, Pattern, Rewrite, Variable
from ibis.util import promote_tuple


def test_enode():
    node = ENode(1, (2, 3))
    assert node == ENode(1, (2, 3))
    assert node != ENode(1, [2, 4])
    assert node != ENode(1, [2, 3, 4])
    assert node != ENode(1, [2])
    assert hash(node) == hash(ENode(1, (2, 3)))
    assert hash(node) != hash(ENode(1, (2, 4)))

    with pytest.raises(AttributeError, match="immutable"):
        node.head = 2
    with pytest.raises(AttributeError, match="immutable"):
        node.args = (2, 3)


def test_enode_roundtrip():
    class MyNode(Concrete, Node):
        a: int
        b: int
        c: str

    # create e-node from node
    node = MyNode(a=1, b=2, c="3")
    enode = ENode.from_node(node)
    assert enode == ENode(MyNode, (1, 2, "3"))

    # reconstruct node from e-node
    node_ = enode.to_node()
    assert node_ == node


def test_enode_roundtrip_with_variadic_arg():
    class MyNode(Concrete, Node):
        a: int
        b: Tuple[int, ...]

    # leaf = "leaf"
    # enode = ENode.from_node(leaf)
    # assert enode == ELeaf(leaf)

    # create e-node from node
    node = MyNode(a=1, b=(2, 3))
    enode = ENode.from_node(node)
    assert enode == ENode(MyNode, (1, (2, 3)))

    # reconstruct node from e-node
    node_ = enode.to_node()
    assert node_ == node


def test_enode_roundtrip_with_nested_arg():
    class MyInt(Concrete, Node):
        value: int

    class MyNode(Concrete, Node):
        a: int
        b: Tuple[MyInt, ...]

    # create e-node from node
    node = MyNode(a=1, b=(MyInt(value=2), MyInt(value=3)))
    enode = ENode.from_node(node)

    # reconstruct node from e-node
    node_ = enode.to_node()
    assert node_ == node


def test_disjoint_set_with_enode():
    class MyNode(Concrete, Node):
        pass

    class MyLit(MyNode):
        value: int

    class MyAdd(MyNode):
        a: MyNode
        b: MyNode

    class MyMul(MyNode):
        a: MyNode
        b: MyNode

    # number postfix highlights the depth of the node
    one = MyLit(value=1)
    two = MyLit(value=2)
    two1 = MyAdd(a=one, b=one)
    three1 = MyAdd(a=one, b=two)
    six2 = MyMul(a=three1, b=two1)
    seven2 = MyAdd(a=six2, b=one)

    # expected enodes postfixed with an underscore
    one_ = ENode(MyLit, (1,))
    two_ = ENode(MyLit, (2,))
    three_ = ENode(MyLit, (3,))
    two1_ = ENode(MyAdd, (one_, one_))
    three1_ = ENode(MyAdd, (one_, two_))
    six2_ = ENode(MyMul, (three1_, two1_))
    seven2_ = ENode(MyAdd, (six2_, one_))

    enode = ENode.from_node(seven2)
    assert enode == seven2_
    assert enode.to_node() == seven2

    ds = DisjointSet()
    for enode in seven2_.traverse():
        ds.add(enode)
        assert ds.find(enode) == enode

    # merging identical nodes should return False
    assert ds.union(three1_, three1_) is False
    assert ds.find(three1_) == three1_
    assert ds[three1_] == {three1_}

    # now merge a (1 + 2) and (3) nodes, but first add `three_` to the set
    ds.add(three_)
    assert ds.union(three1_, three_) is True
    assert ds.find(three1_) == three1_
    assert ds.find(three_) == three1_
    assert ds[three_] == {three_, three1_}


def test_pattern():
    Pattern._counter = itertools.count()

    p = Pattern(ops.Literal, (1, dt.int8))
    assert p.head == ops.Literal
    assert p.args == (1, dt.int8)
    assert p.name == "_0"

    p = "name" @ Pattern(ops.Literal, (1, dt.int8))
    assert p.head == ops.Literal
    assert p.args == (1, dt.int8)
    assert p.name == "name"


def test_pattern_flatten():
    Pattern._counter = itertools.count()

    # using auto-generated names
    one = Pattern(ops.Literal, (1, dt.int8))
    two = Pattern(ops.Literal, (2, dt.int8))
    three = Pattern(ops.Add, (one, two))

    result = dict(three.flatten())
    expected = {
        Variable("_2"): Pattern(ops.Add, (Variable("_0"), Variable("_1"))),
        Variable("_1"): Pattern(ops.Literal, (2, dt.int8)),
        Variable("_0"): Pattern(ops.Literal, (1, dt.int8)),
    }
    assert result == expected

    # using user-provided names which helps capturing variables
    one = "one" @ Pattern(ops.Literal, (1, dt.int8))
    two = "two" @ Pattern(ops.Literal, (2, dt.int8))
    three = "three" @ Pattern(ops.Add, (one, two))

    result = tuple(three.flatten())
    expected = (
        (Variable("one"), Pattern(ops.Literal, (1, dt.int8))),
        (Variable("two"), Pattern(ops.Literal, (2, dt.int8))),
        (Variable("three"), Pattern(ops.Add, (Variable("one"), Variable("two")))),
    )
    assert result == expected


class PatternNamespace:
    __slots__ = '_module'

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        klass = getattr(self._module, name)

        def pattern(*args):
            return Pattern(klass, args)

        return pattern


p = PatternNamespace(ops)
a = Variable('a')

one = ibis.literal(1)
two = one * 2
two_ = one + one
two__ = ibis.literal(2)
three = one + two
six = three * two_
seven = six + 1
seven_ = seven * 1
eleven = seven_ + 4


def test_egraph_simple_match():
    eg = EGraph()
    eg.add(eleven.op())

    pat = p.Multiply(a, "lit" @ p.Literal(1, dt.int8))
    res = eg.match(pat)

    enode = ENode.from_node(seven_.op())
    matches = res[enode]
    assert matches['a'] == ENode.from_node(seven.op())
    assert matches['lit'] == ENode.from_node(one.op())


def test_egraph_extract():
    eg = EGraph()
    eg.add(eleven.op())

    res = eg.extract(one.op())
    assert res == one.op()


# def test_egraph_extract_minimum_cost():
#     assert ENode.from_node(two.op()).cost == 4
#     assert ENode.from_node(two_.op()).cost == 4
#     assert ENode.from_node(two__.op()).cost == 2

#     eg = EGraph()
#     eg.add(two.op())
#     eg.add(two_.op())
#     eg.add(two__.op())

#     eg.union(two.op(), two_.op())
#     assert eg.extract(two.op()) in {two.op(), two_.op()}

#     eg.union(two.op(), two__.op())
#     assert eg.extract(two.op()) == two__.op()


def test_egraph_rewrite_to_variable():
    eg = EGraph()
    eg.add(eleven.op())

    # rule with a variable on the right-hand side
    rule = Rewrite(p.Multiply(a, "lit" @ p.Literal(1, dt.int8)), a)
    eg.apply(rule)
    assert eg.equivalent(seven_.op(), seven.op())


def test_egraph_rewrite_to_constant():
    node = (one * 0).op()

    eg = EGraph()
    eg.add(node)

    # rule with a constant on the right-hand side
    rule = Rewrite(p.Multiply(a, "lit" @ p.Literal(0, dt.int8)), 0)
    eg.apply(rule)
    assert eg.equivalent(node, 0)


def test_egraph_rewrite_to_pattern():
    eg = EGraph()
    eg.add(three.op())

    # rule with a pattern on the right-hand side
    rule = Rewrite(p.Multiply(a, "lit" @ p.Literal(2, dt.int8)), p.Add(a, a))
    eg.apply(rule)
    assert eg.equivalent(two.op(), two_.op())


def test_egraph_rewrite_dynamic():
    def applier(match, subst):
        return p.Add(subst['a'], subst['a']).to_enode()

    node = (one * 2).op()

    eg = EGraph()
    eg.add(node)

    # rule with a dynamic pattern on the right-hand side
    rule = Rewrite(
        "mul" @ p.Multiply(a, p.Literal(Variable("times"), dt.int8)), applier
    )
    eg.apply(rule)

    assert eg.extract(node) in {two.op(), two_.op()}


class Base(Concrete, Node):
    def __class_getitem__(cls, args):
        args = promote_tuple(args)
        return Pattern(cls, args)


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

from pprint import pprint


def simplify(expr, rules, iters=7):
    egraph = EGraph()
    egraph.add(expr)
    egraph.run(rules, iters)
    print()
    print("ERelations:")
    print("-----------")
    pprint(egraph._erelations)
    print()
    print("EClasses:")
    print("---------")
    pprint(egraph._eclasses._classes)
    print()
    print("ECosts:")
    print("-------")
    pprint(egraph._ecosts)
    print()
    print("EBests:")
    print("-------")
    pprint(egraph._ebests)

    best = egraph.extract(expr)
    return best


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


def test_simple_4():
    rules = [
        Mul[a, b] >> Mul[b, a],
        Mul[a, 1] >> a,
    ]

    node = Mul(2, Mul(1, 3))
    expected = Mul(2, 3)

    assert simplify(node, rules, iters=20000) == expected
