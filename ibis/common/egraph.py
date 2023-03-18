import collections
import itertools
from typing import NamedTuple

from bidict import bidict

from ibis.common.graph import Node
from ibis.util import promote_list

# consider to use an EClass(id, nodes) dataclass


class ENode:
    __slots__ = ("head", "args")

    def __init__(self, head, args):
        self.head = head
        self.args = tuple(args)

    def __repr__(self):
        argstring = ", ".join(map(repr, self.args))
        return f"ENode({self.head.__name__}, {argstring})"

    def __eq__(self, other):
        return self.head == other.head and self.args == other.args

    def __hash__(self):
        return hash((self.__class__, self.head, self.args))

    @classmethod
    def from_node(cls, egraph, node):
        head = type(node)
        args = tuple(
            egraph.add(arg) if isinstance(arg, Node) else Atom(arg)
            for arg in node.__args__
        )
        return cls(head, args)

    def to_node(self, egraph):
        return self.head(*self.args)


# TODO: move every E* into the Egraph so its API only uses Nodes
# TODO: track whether the egraph is saturated or not
# TODO: support parent classes in etables (Join <= InnerJoin)


class EGraph:
    __slots__ = (
        "_nodes",
        "_counter",
        "_enodes",
        "_eparents",
        "_eclasses",
        "_etables",

    )

    def __init__(self):
        # store the nodes before converting them to enodes, so we can spare the initial
        # node traversal and omit the creation of enodes
        self._nodes = {}
        # counter for generating new eclass ids
        self._counter = itertools.count()

        # map enodes to eclass ids so we can check if an enode is already in the egraph
        self._enodes = bidict()
        # map eclass ids to their parent eclass id, this is required for the union-find
        self._eparents = {}
        # map eclass ids to their eclass
        self._eclasses = {}
        # map enode heads to their eclass ids and their arguments, this is required for
        # the relational e-matching
        self._etables = collections.defaultdict(dict)

    def __repr__(self):
        return f"EGraph({self._eclasses})"

    def _create_node(self, id):
        id = self._eparents[id]
        eclass = self._eclasses[id]
        enodes = [self._enodes.inverse[id] for id in eclass]
        enode = enodes[0]

        args = []
        for arg in enode.args:
            if isinstance(arg, Atom):
                args.append(arg.value)
            else:
                arg = self._create_node(arg)
                args.append(arg)

        return enode.head(*args)

    def add(self, node):
        if isinstance(node, Node):
            enode = self._nodes.get(node) or ENode.from_node(self, node)
            self._nodes[node] = enode
        elif isinstance(node, ENode):
            args = []
            for arg in node.args:
                if isinstance(arg, ENode):
                    args.append(self.add(arg))
                elif isinstance(arg, (Atom, int)):
                    args.append(arg)
                else:
                    raise TypeError(arg)
            enode = ENode(node.head, args)
        else:
            raise TypeError(
                f"`node` must be an instance of ibis.common.graph.Node but got {type(node)}"
            )

        if (id := self._enodes.get(enode)) is not None:
            return id

        assert enode not in self._enodes
        id = next(self._counter)

        self._enodes[enode] = id
        self._eparents[id] = id
        self._eclasses[id] = {id}
        self._etables[enode.head][id] = tuple(enode.args)

        return id

    def find(self, id):
        return self._eparents[id]

    def union(self, id1, id2):
        assert isinstance(id1, int)
        assert isinstance(id2, int)
        id1 = self._eparents[id1]
        id2 = self._eparents[id2]
        if id1 == id2:
            return False

        # Merge the smaller eclass into the larger one
        class1 = self._eclasses[id1]
        class2 = self._eclasses[id2]
        if len(class1) >= len(class2):  # >= is important
            id1, id2 = id2, id1
            class1, class2 = class2, class1

        # Update the parent pointer
        for id in class1:
            self._eparents[id] = id2

        # Do the actual merging and clear the other eclass
        class2 |= class1
        class1.clear()

        return True

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
                    subst[patarg.name] = self._eparents[arg]
            elif isinstance(arg, Atom):
                if patarg != arg.value:
                    return None
            else:
                if self._eparents[arg] != self._eparents[patarg]:
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
                elif isinstance(rewrite.rhs, Node):
                    otherid = self.add(rewrite.rhs)
                elif callable(rewrite.rhs):
                    enode = rewrite.rhs(self, id, subst)
                    if isinstance(enode, ENode):
                        otherid = self.add(enode)
                    else:
                        otherid = enode

                n_changes += self.union(id, otherid)

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

        return self._create_node(id)

    def equivalent(self, id1, id2):
        id1 = self._eparents[id1]
        id2 = self._eparents[id2]
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
