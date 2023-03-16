import pytest

from ibis.common.graph import Node
from ibis.common.grounds import Annotable
from ibis.common.newegraph import ELeaf, ENode


class MyNode(Annotable, Node):
    a: int
    b: int
    c: str


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
    node = MyNode(a=1, b=2, c="3")
    enode = ENode.from_node(node)
    assert enode == ENode(MyNode, (ELeaf(1), ELeaf(2), ELeaf("3")))
