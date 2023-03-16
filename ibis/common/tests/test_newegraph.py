from pprint import pprint

import pytest

from ibis.common.collections import DisjointSet
from ibis.common.graph import Node
from ibis.common.grounds import Annotable
from ibis.common.newegraph import ELeaf, ENode


def test_eleaf():
    leaf = ELeaf(1)
    assert leaf == ELeaf(1)
    assert leaf != ELeaf(2)
    assert hash(leaf) == hash(ELeaf(1))
    assert hash(leaf) != hash(ELeaf(2))

    with pytest.raises(AttributeError, match="immutable"):
        leaf.value = 2


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


def test_enode_from_node():
    class MyNode(Annotable, Node):
        a: int
        b: int
        c: str

    leaf = "leaf"
    enode = ENode.from_node(leaf)
    assert enode == ELeaf(leaf)

    node = MyNode(a=1, b=2, c="3")
    enode = ENode.from_node(node)
    assert enode == ENode(MyNode, (ELeaf(1), ELeaf(2), ELeaf("3")))
    assert list(enode.traverse()) == [ELeaf(1), ELeaf(2), ELeaf("3"), enode]


def test_disjoint_set_with_enode():
    class MyNode(Annotable, Node):
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
    three = MyLit(value=3)
    two1 = MyAdd(a=one, b=one)
    three1 = MyAdd(a=one, b=two)
    six2 = MyMul(a=three1, b=two1)
    seven2 = MyAdd(a=six2, b=one)

    # expected enodes postfixed with an underscore
    one_ = ENode(MyLit, (ELeaf(1),))
    two_ = ENode(MyLit, (ELeaf(2),))
    three_ = ENode(MyLit, (ELeaf(3),))
    two1_ = ENode(MyAdd, (one_, one_))
    three1_ = ENode(MyAdd, (one_, two_))
    six2_ = ENode(MyMul, (three1_, two1_))
    seven2_ = ENode(MyAdd, (six2_, one_))

    enode = ENode.from_node(seven2)
    assert enode == seven2_

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
