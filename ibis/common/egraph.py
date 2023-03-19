import collections
import itertools
import math
from typing import Any, NamedTuple, Tuple

from bidict import bidict

from ibis.common.collections import DisjointSet
from ibis.common.graph import Node
from ibis.util import promote_list

# consider to use an EClass(id, nodes) dataclass
# TODO(kszucs): using ENode ids instead of integer ids makes the egraph slightly
# slower (22ms -> 25ms)


class Slotted:
    __slots__ = ('__precomputed_hash__',)

    def __init__(self, *args):
        for name, value in itertools.zip_longest(self.__slots__, args):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "__precomputed_hash__", hash(args))

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        for name in self.__slots__:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self):
        return self.__precomputed_hash__

    def __setattr__(self, name, value):
        raise AttributeError("Can't set attributes on immutable ENode instance")


class ENode:
    __slots__ = ("head", "args", "__precomputed_hash__")

    def __init__(self, head, args):
        object.__setattr__(self, "head", head)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(
            self, "__precomputed_hash__", hash((self.__class__, self.head, self.args))
        )

    # @property
    # def __argnames__(self):
    #     return ()

    # @property
    # def __args__(self):
    #     return self.args

    @property
    def id(self):
        return self.__precomputed_hash__

    def __repr__(self):
        argstring = ", ".join(map(repr, self.args))
        return f"ENode({self.head.__name__}, {argstring})"

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        return self.head == other.head and self.args == other.args

    def __hash__(self):
        return self.__precomputed_hash__

    def __setattr__(self, name, value):
        raise AttributeError("immutable")

    def __lt__(self, other):
        return False

    @classmethod
    def from_node(cls, node: Any):
        def mapper(node, _, **kwargs):
            return cls(node.__class__, kwargs.values())

        return node.map(mapper)[node]
        head = type(node)
        args = tuple(
            cls.from_node(arg) if isinstance(arg, Node) else arg
            for arg in node.__args__
        )
        return cls(head, args)

    def to_node(self):
        return self.head(*self.args)


# TODO: move every E* into the Egraph so its API only uses Nodes
# TODO: track whether the egraph is saturated or not
# TODO: support parent classes in etables (Join <= InnerJoin)


class EGraph:
    __slots__ = ("_nodes", "_eclasses", "_etables", "_classes")

    def __init__(self):
        # store the nodes before converting them to enodes, so we can spare the initial
        # node traversal and omit the creation of enodes
        self._nodes = {}

        self._eclasses = DisjointSet()
        # map enode heads to their eclass ids and their arguments, this is required for
        # the relational e-matching
        self._etables = collections.defaultdict(dict)

    def __repr__(self):
        return f"EGraph({self._eclasses})"

    # TODO(kszucs): this should be done during `union` operation
    def pina(self, enode):
        head_costs = collections.defaultdict(lambda: 100)

        enode = self._eclasses.find(enode)

        # set each cost to infinity
        costs = {
            en: (math.inf, None) for en in self._eclasses.keys()
        }  # eclass -> (cost, enode)
        changed = True

        def enode_cost(enode):
            if isinstance(enode, ENode):
                cost = head_costs[enode.head]
                for arg in enode.args:
                    if isinstance(arg, ENode):
                        cost += costs[arg][0]
                    else:
                        cost += 1
            else:
                cost = 1
            return cost

        # iterate until we settle, taking the lowest cost option
        while changed:
            changed = False
            for en, enodes in self._eclasses.items():
                new_cost = min((enode_cost(en), en) for en in enodes)
                if costs[en][0] != new_cost[0]:
                    changed = True
                costs[en] = new_cost

        def extract(en):
            if not isinstance(en, ENode):
                return en

            best = costs[en][1]
            print("BEST OF", en, best, costs[best][1])

            if isinstance(best, ENode):
                return best.head(*tuple(extract(a) for a in best.args))
            else:
                return best

        print("==== BEST ====")
        return extract(enode)

    def add(self, node):
        if isinstance(node, ENode):
            args = []
            for arg in node.args:
                if isinstance(arg, ENode):
                    args.append(self.add(arg))
                else:
                    args.append(arg)

            enode = ENode(node.head, args)
        elif isinstance(node, Node):
            if node in self._nodes:
                enode = self._nodes[node]
            else:
                head = type(node)
                args = tuple(
                    self.add(arg) if isinstance(arg, Node) else arg
                    for arg in node.__args__
                )
                enode = ENode(head, args)
            self._nodes[node] = enode
        else:
            raise TypeError(
                f"`node` must be an instance of ibis.common.graph.Node but got {type(node)}"
            )

        if enode in self._eclasses:
            return self._eclasses.find(enode)

        self._eclasses.add(enode)
        self._etables[enode.head][enode] = tuple(enode.args)

        return enode

    def union(self, node1, node2):
        enode1 = ENode.from_node(node1)
        enode2 = ENode.from_node(node2)
        return self._eclasses.union(enode1, enode2)

    def _match_args(self, args, patargs):
        if len(args) != len(patargs):
            return None

        subst = {}
        for arg, patarg in zip(args, patargs):
            if isinstance(patarg, Variable):
                if isinstance(arg, ENode):
                    subst[patarg.name] = self._eclasses.find(arg)
                else:
                    subst[patarg.name] = arg
            elif isinstance(arg, ENode):
                # perhaps need another branch here
                if self._eclasses.find(arg) != self._eclasses.find(arg):
                    return None
            else:
                if patarg != arg:
                    return None

        return subst

    def match(self, pattern):
        # patterns could be reordered to match on the most selective one first
        (auxvar, pattern), *rest = reversed(list(pattern.flatten()))
        matches = {}

        rel = self._etables[pattern.head]
        for enode, args in rel.items():
            if (subst := self._match_args(args, pattern.args)) is not None:
                subst[auxvar.name] = enode
                matches[enode] = subst

        for auxvar, pattern in rest:
            rel = self._etables[pattern.head]
            tmp = {}
            for enode, subst in matches.items():
                if args := rel.get(subst[auxvar.name]):
                    if (newsubst := self._match_args(args, pattern.args)) is not None:
                        tmp[enode] = {**subst, **newsubst}
            matches = tmp

        return matches

    def apply(self, rewrites):
        n_changes = 0
        for rewrite in promote_list(rewrites):
            for id, subst in self.match(rewrite.lhs).items():
                # MOVE this check to specific subclasses to avoid the isinstance check
                if isinstance(rewrite.rhs, (Variable, Pattern)):
                    enode = rewrite.rhs.substitute(subst)
                    if isinstance(enode, ENode):
                        otherid = self.add(enode)
                    else:
                        otherid = enode
                elif isinstance(rewrite.rhs, ENode):
                    otherid = self.add(rewrite.rhs)
                elif isinstance(rewrite.rhs, Node):
                    otherid = self.add(rewrite.rhs)
                elif callable(rewrite.rhs):
                    enode = rewrite.rhs(self, id, subst)
                    if isinstance(enode, ENode):
                        otherid = self.add(enode)
                    else:
                        otherid = enode

                n_changes += self._eclasses.union(id, otherid)

        return n_changes

    def run(self, rewrites, n=1):
        for _i in range(n):
            if not self.apply(rewrites):
                print(f"Saturated after {_i} iterations")
                return True
        return False

    def extract(self, node):
        if isinstance(node, Node):
            enode = self._nodes.get(node)
        elif isinstance(node, ENode):
            enode = node
        elif isinstance(node, int):
            raise TypeError(node)

        return self.pina(enode)

    def equivalent(self, node1, node2):
        if isinstance(node1, Node):
            enode1 = ENode.from_node(node1)
        else:
            enode1 = node1
        if isinstance(node2, Node):
            enode2 = ENode.from_node(node2)
        else:
            enode2 = node2
        enode1 = self._eclasses.find(enode1)
        enode2 = self._eclasses.find(enode2)
        return enode1 == enode2


class Variable(Slotted):
    __slots__ = ("name",)

    def __repr__(self):
        return f"${self.name}"

    def substitute(self, subst):
        return subst[self.name]


class Pattern(Slotted):
    __slots__ = ("head", "args", "name")

    def __init__(self, head, args, name=None):
        super().__init__(head, tuple(args), name)

    def __repr__(self):
        argstring = ", ".join(map(repr, self.args))
        return f"{self.head.__name__}({argstring})"

    def __rshift__(self, rhs):
        return Rewrite(self, rhs)

    def __rmatmul__(self, name):
        return self.__class__(self.head, self.args, name)

    def flatten(self, var=None, counter=None):
        counter = counter or itertools.count()

        if var is None:
            if self.name is None:
                var = Variable(next(counter))
            else:
                var = Variable(self.name)

        args = []
        for arg in self.args:
            if isinstance(arg, Pattern):
                if arg.name is None:
                    aux = Variable(next(counter))
                else:
                    aux = Variable(arg.name)
                yield from arg.flatten(aux, counter)
                args.append(aux)
            else:
                args.append(arg)

        yield (var, Pattern(self.head, args))

    def substitute(self, subst):
        args = []
        for arg in self.args:
            if isinstance(arg, (Variable, Pattern)):
                args.append(arg.substitute(subst))
            else:
                args.append(arg)
        return ENode(self.head, args)

    @classmethod
    def namespace(cls, module):
        return PatternNamespace(module)


class PatternNamespace(Slotted):
    __slots__ = ('module',)

    def __getattr__(self, name):
        klass = getattr(self.module, name)

        def pattern(*args):
            return Pattern(klass, args)

        return pattern


# USE SEARCHER AND APPLIER NOTATIONS


class Rewrite:
    __slots__ = ("lhs", "rhs", "name")

    def __init__(self, lhs, rhs, name=None):
        assert isinstance(lhs, Pattern)
        # TODO: use a substitutable mixin
        # assert isinstance(rhs, (Variable, Pattern, Node))
        self.lhs = lhs
        self.rhs = rhs
        self.name = name

    def __repr__(self):
        return f"{self.lhs} >> {self.rhs}"

    def __eq__(self, other):
        return self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self):
        return hash((self.__class__, self.lhs, self.rhs))


# ops.Multiply[a, b] => ops.Add[ops.Multiply[a, b], ops.Multiply[a, b]]
# SyntacticPattern
# DynamicPattern
