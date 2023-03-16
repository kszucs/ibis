import collections
import itertools
from typing import NamedTuple

from bidict import bidict

from ibis.common.graph import Node
from ibis.util import promote_list


class ENode:
    __slots__ = ('head', 'args', 'hash')

    def __init__(self, head, args):
        object.__setattr__(self, "head", head)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "hash", hash((self.__class__, self.head, self.args)))

    @classmethod
    def from_node(cls, node: Node):
        args = tuple(
            cls.from_node(arg) if isinstance(arg, Node) else ELeaf(arg)
            for arg in node.__args__
        )
        return cls(node.__class__, args)

    def __eq__(self, other):
        return self.head == other.head and self.args == other.args

    def __hash__(self):
        return self.hash

    def __setattr__(self, name, value):
        raise AttributeError("Can't set attributes on immutable ENode instance")


class ELeaf:
    __slots__ = ('value', 'hash')

    def __init__(self, value):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "hash", hash((self.__class__, self.value)))

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return self.hash

    def __setattr__(self, name, value):
        raise AttributeError("Can't set attributes on immutable ELeaf instance")


# class EPattern(ETerm):
#     __slots__ = ()
# Pattern is almost identical with ENode it just can hold Variables
