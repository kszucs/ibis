import collections
import itertools
from typing import NamedTuple

from bidict import bidict
import math
from ibis.common.collections import DisjointSet
from ibis.common.graph import Node
from ibis.util import promote_list

# consider to use an EClass(id, nodes) dataclass


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
        return type(self) is type(other) and self.head == other.head and self.args == other.args

    def __hash__(self):
        return self.__precomputed_hash__

    def __setattr__(self, name, value):
        raise AttributeError("immutable")

    @classmethod
    def from_node(cls, node):
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
    __slots__ = ("_nodes", "_counter", "_enodes", "_eclasses", "_etables", "_classes")

    def __init__(self):
        # store the nodes before converting them to enodes, so we can spare the initial
        # node traversal and omit the creation of enodes
        self._nodes = {}
        # counter for generating new eclass ids
        self._counter = itertools.count()

        # map enodes to eclass ids so we can check if an enode is already in the egraph
        self._enodes = bidict()
        # map eclass ids to their parent eclass id, this is required for the union-find

        self._eclasses = DisjointSet()
        # map enode heads to their eclass ids and their arguments, this is required for
        # the relational e-matching
        self._etables = collections.defaultdict(dict)

    def __repr__(self):
        return f"EGraph({self._eclasses})"

    # TODO(kszucs): this should be done during `union` operation
    def pina(self, id):
        print(id)
        print(self._eclasses._parents)
        print(self._eclasses._classes)
        head_costs = collections.defaultdict(lambda: 100)

        id = self._eclasses.find(id)


        #eclasses = eg.eclasses()
        # set each cost to infinity
        costs = {eid: (math.inf, None) for eid in self._eclasses.keys()} # eclass -> (cost, enode)
        changed = True
        def enode_cost(id):

            if isinstance(id, Atom):
                return 1

            enode = self._enodes.inverse[id]
            cost = head_costs[enode.head]
            for arg in enode.args:
                if isinstance(arg, Atom):
                    cost += 1
                else:
                    cost += costs[arg][0]
            return cost


        # iterate until we settle, taking the lowest cost option
        while changed:
            changed = False
            for eid, enodes in self._eclasses.items():
                #print(enodes)
                new_cost = min((enode_cost(enode), enode) for enode in enodes)
                if costs[eid][0] != new_cost[0]:
                    changed = True
                costs[eid] = new_cost

        print(costs)

        def extract(eid):
            #print(eid)
            if isinstance(eid, Atom):
                return eid.value

            best = costs[eid][1]
            print("BEST OF", eid, best, costs[best][1])

            enode = self._enodes.inverse[best]

            if isinstance(enode, ENode):
                return enode.head(*tuple(extract(a) for a in enode.args))
            else:
                return enode

        print("==== BEST ====")
        return extract(id)

    # def _extract_best(self, id):
    #     id = self._eclasses.find(id)
    #     eclass = self._eclasses[id]
    #     enodes = [self._enodes.inverse[id] for id in eclass]
    #     best = enodes[0]
    #     return best

    # def _create_node(self, id):
    #     print(id)
    #     enode = self._extract_best(id)

    #     args = []
    #     for arg in enode.args:
    #         if isinstance(arg, Atom):
    #             args.append(arg.value)
    #         else:
    #             arg = self._create_node(arg)
    #             args.append(arg)

    #     return enode.head(*args)

    def add(self, node):
        if isinstance(node, ENode):
            args = []
            for arg in node.args:
                if isinstance(arg, ENode):
                    args.append(self.add(arg))
                elif isinstance(arg, (Atom, int)):
                    args.append(arg)
                else:
                    raise TypeError(arg)
            enode = ENode(node.head, args)
        elif isinstance(node, Node):
            if node in self._nodes:
                enode = self._nodes[node]
            else:
                head = type(node)
                args = tuple(
                    self.add(arg) if isinstance(arg, Node) else Atom(arg)
                    for arg in node.__args__
                )
                enode = ENode(head, args)
            self._nodes[node] = enode
        else:
            raise TypeError(
                f"`node` must be an instance of ibis.common.graph.Node but got {type(node)}"
            )

        if (id := self._enodes.get(enode)) is not None:
            return id

        assert enode not in self._enodes

        id = next(self._counter)
        #print(id, enode.id)
        id = enode.id

        self._enodes[enode] = id
        self._eclasses.add(id)
        self._etables[enode.head][id] = tuple(enode.args)

        return id

    def _match_args(self, args, patargs):
        if len(args) != len(patargs):
            return None

        subst = {}
        for arg, patarg in zip(args, patargs):
            if isinstance(patarg, Variable):
                if isinstance(arg, Atom):
                    subst[patarg.name] = arg
                elif isinstance(arg, ENode):
                    subst[patarg.name] = arg
                else:
                    subst[patarg.name] = self._eclasses.find(arg)
            elif isinstance(arg, Atom):
                if patarg != arg.value:
                    return None
            else:
                if self._eclasses.find(arg) != self._eclasses.find(arg):
                    return None

        return subst

    def match(self, pattern):
        # patterns could be reordered to match on the most selective one first
        patterns = dict(reversed(list(pattern.flatten())))

        matches = {}
        for auxvar, pattern in patterns.items():
            table = self._etables[pattern.head]

            if auxvar is None:
                for id, args in table.items():
                    if (subst := self._match_args(args, pattern.args)) is not None:
                        matches[id] = subst
            else:
                newmatches = {}
                for id, subst in matches.items():
                    sid = subst[auxvar.name]
                    if args := table.get(sid):
                        if (
                            newsubst := self._match_args(args, pattern.args)
                        ) is not None:
                            newmatches[id] = {**subst, **newsubst}

                matches = newmatches

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
        print(node)
        if isinstance(node, Node):
            enode = self._nodes.get(node)
            id = self._enodes[enode]
        elif isinstance(node, ENode):
            id = self._enodes[node]
        elif isinstance(node, int):
            id = node

        return self.pina(id)
        # return self._create_node(id)

    def equivalent(self, id1, id2):
        id1 = self._eclasses.find(id1)
        id2 = self._eclasses.find(id2)
        return id1 == id2


class Atom:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"`{self.value}`"

    def __hash__(self):
        return hash((self.__class__, self.value))

    def __eq__(self, other):
        return self.value == other.value


class Variable:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"${self.name}"

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash((self.__class__, self.name))

    def substitute(self, subst):
        return subst[self.name]


class Pattern:
    __slots__ = ("head", "args")
    _counter = itertools.count()

    def __init__(self, head, args):
        self.head = head
        self.args = tuple(args)

    def __repr__(self):
        argstring = ", ".join(map(repr, self.args))
        return f"{self.head.__name__}({argstring})"

    def __rshift__(self, rhs):
        return Rewrite(self, rhs)

    def __eq__(self, other):
        return self.head == other.head and self.args == other.args

    def __hash__(self):
        return hash((self.__class__, self.head, self.args))

    def flatten(self, var=None):
        args = []
        for arg in self.args:
            if isinstance(arg, Pattern):
                aux = Variable(f'_{next(self._counter)}')
                yield from arg.flatten(aux)
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
                args.append(Atom(arg))
        return ENode(self.head, args)


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
