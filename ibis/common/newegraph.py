import collections
import itertools
from typing import Any, NamedTuple

from bidict import bidict

from ibis.common.graph import Node
from ibis.util import promote_list


class ENode:
    __slots__ = ("head", "args", "hash")

    def __init__(self, head, args):
        object.__setattr__(self, "head", head)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "hash", hash((self.__class__, self.head, self.args)))

    @classmethod
    def from_node(cls, node: Any):
        if isinstance(node, Node):
            args = tuple(map(cls.from_node, node.__args__))
            return ENode(node.__class__, args)
        else:
            return ELeaf(node)

    def traverse(self):
        """Traverse the tree in a depth-first manner."""
        for arg in self.args:
            yield from arg.traverse()
        yield self

    def __eq__(self, other):
        return self.head == other.head and self.args == other.args

    def __repr__(self) -> str:
        return f"ENode({self.head.__name__}, {self.args})"

    def __hash__(self):
        return self.hash

    def __setattr__(self, name, value):
        raise AttributeError("Can't set attributes on immutable ENode instance")


class ELeaf:
    __slots__ = ("value", "hash")

    def __init__(self, value):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "hash", hash((self.__class__, self.value)))

    def traverse(self):
        yield self

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return f"ELeaf({self.value})"

    def __hash__(self):
        return self.hash

    def __setattr__(self, name, value):
        raise AttributeError("Can't set attributes on immutable ELeaf instance")


# class EPattern(ETerm):
#     __slots__ = ()
# Pattern is almost identical with ENode it just can hold Variables
