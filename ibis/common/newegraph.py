import collections
import itertools
from typing import Any, NamedTuple

from bidict import bidict

from ibis.common.collections import DisjointSet
from ibis.common.graph import Node
from ibis.util import promote_list

# TODO(kszucs): cost could be directly assigned to ENodes during construction
# based on the number of nodes in the ENode's tree


# TODO(kszucs): consider to implement Comparable for ETerms in order to have
# more performant equality checks


class Term:
    __slots__ = ("head", "args", "__precomputed_hash__")

    def __init__(self, head, args):
        object.__setattr__(self, "head", head)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(
            self, "__precomputed_hash__", hash((self.__class__, self.head, self.args))
        )

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.head == other.head
            and self.args == other.args
        )

    def __hash__(self):
        return self.__precomputed_hash__

    def __setattr__(self, name, value):
        raise AttributeError("Can't set attributes on immutable ENode instance")


class ENode(Term, Node):
    __slots__ = ()

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

    def __repr__(self) -> str:
        return f"ENode({self.head.__name__}, {self.args})"


class EGraph:
    __slots__ = ("_nodes", "_eclasses", "_erelations")

    def __init__(self):
        self._nodes = bidict()
        self._eclasses = DisjointSet()
        self._erelations = collections.defaultdict(dict)

    def add(self, node):
        enode = ENode.from_node(node)
        self._nodes[node] = enode
        self._eclasses.add(enode)
        self._erelations[enode.head][enode] = enode.args

    def match(self, pattern):
        pass


class Variable:
    __slots__ = ("name", "__precomputed_hash__")

    def __init__(self, name):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "__precomputed_hash__", hash((self.__class__, name)))

    def __repr__(self) -> str:
        return f"${self.name}"

    def __hash__(self):
        return self.__precomputed_hash__

    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name

    def __setattr__(self, name, value):
        raise AttributeError("Can't set attributes on immutable ENode instance")


class Pattern(Term):
    __slots__ = ("name",)
    _counter = itertools.count()

    def __init__(self, head, args, name=None):
        super().__init__(head, args)
        name = name or f'_{next(self._counter)}'
        object.__setattr__(self, "name", name)

    def __repr__(self):
        argstring = ", ".join(map(repr, self.args))
        return f"{self.head.__name__}({argstring})"

    def __rmatmul__(self, name):
        return self.__class__(self.head, self.args, name)

    def flatten(self, var=None):
        args = []
        for arg in self.args:
            if isinstance(arg, Pattern):
                aux = Variable(arg.name)
                yield from arg.flatten(aux)
                args.append(aux)
            else:
                args.append(arg)
        yield (var, Pattern(self.head, args))




# class EPattern(ETerm):
#     __slots__ = ()
# Pattern is almost identical with ENode it just can hold Variables
