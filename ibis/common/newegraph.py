import collections
import itertools
from typing import Any, NamedTuple

from bidict import bidict

from ibis.common.graph import Node
from ibis.util import promote_list

# TODO(kszucs): cost could be directly assigned to ENodes during construction
# based on the number of nodes in the ENode's tree


class ENode(Node):
    __slots__ = ("head", "args", "__precomputed_hash__")

    def __init__(self, head, args):
        object.__setattr__(self, "head", head)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(
            self, "__precomputed_hash__", hash((self.__class__, self.head, self.args))
        )

    @property
    def __args__(self):
        return self.args

    @property
    def __argnames__(self):
        return self.head.__argnames__

    # we can maybe spare this conversion if we don't try to recreate the original type
    # but rather use a Term[head, args] along with the original inputs, during
    # substitution a term would be produced but it would be nice to have the same hash
    # as the original
    @classmethod
    def from_node(cls, node: Any):
        def mapper(node, _, **kwargs):
            return cls(node.__class__, kwargs.values())

        return node.map(mapper)[node]

    def to_node(self):
        def mapper(node, _, **kwargs):
            return node.head(**kwargs)

        return self.map(mapper)[self]

    def __eq__(self, other):
        return self.head == other.head and self.args == other.args

    def __repr__(self) -> str:
        return f"ENode({self.head.__name__}, {self.args})"

    def __hash__(self):
        return self.__precomputed_hash__

    def __setattr__(self, name, value):
        raise AttributeError("Can't set attributes on immutable ENode instance")


# class EPattern:
#     __slots__ = ("head", "args", "hash")

#     def __init__(self, head, args):
#         object.__setattr__(self, "head", head)
#         object.__setattr__(self, "args", tuple(args))
#         object.__setattr__(self, "hash", hash((self.__class__, self.head, self.args)))

#     @classmethod
#     def from_node(cls, node: Any):
#         if isinstance(node, Node):
#             args = tuple(map(cls.from_node, node.__args__))
#             return EPattern(node.__class__, args)
#         elif isinstance(node, Variable):
#             return EVariable(node.name)
#         else:
#             return ELeaf(node)

#     def to_node(self):
#         args = (arg.to_node() for arg in self.args)
#         return self.head(*args)

#     def traverse(self):
#         """Traverse the tree in a depth-first manner."""
#         for arg in self.args:
#             yield from arg.traverse()
#         yield self

#     def __eq__(self, other):
#         return self.head == other.head and self.args == other.args

#     def __repr__(self) -> str:
#         return f"EPattern({self.head.__name__}, {self.args})"

#     def __hash__(self):
#         return self.hash

#     def __setattr__(self, name, value):
#         raise AttributeError("Can't set attributes on immutable EPattern instance")

# class Variable


# class EPattern(ETerm):
#     __slots__ = ()
# Pattern is almost identical with ENode it just can hold Variables
