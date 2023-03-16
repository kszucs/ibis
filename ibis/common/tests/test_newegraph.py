from pprint import pprint
from typing import Tuple

import pytest

from ibis.common.collections import DisjointSet
from ibis.common.graph import Node
from ibis.common.grounds import Concrete
from ibis.common.newegraph import ENode


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
