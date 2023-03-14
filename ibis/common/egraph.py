import collections
import itertools

# class EClass(set[int]):
#     pass
# from rich.pretty import pprint
from pprint import pprint

from ibis.common.graph import Node
from ibis.util import promote_list
from bidict import bidict
# consider to use an EClass(id, nodes) dataclass

class ENode:
    __slots__ = ("head", "args")

    def __init__(self, head, args):
        self.head = head
        self.args = args

    def __repr__(self):
        argstring = ", ".join(map(repr, self.args))
        return f"ENode({self.head.__name__}, {argstring})"

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



class EGraph:
    __slots__ = ("_nodes", "_counter", "_enodes", "_eparents", "_eclasses", "_etables")

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
            enode = node
        else:
            raise TypeError("`node` must be an instance of ibis.common.graph.Node")

        if (id := self._enodes.get(enode)) is not None:
            return id
        assert enode not in self._enodes

        id = next(self._counter)

        self._enodes[enode] = id
        self._eparents[id] = id
        self._eclasses[id] = {id}
        self._etables[enode.head][id] = enode.args
        return id

    def find(self, id):
        return self._eparents[id]

    def union(self, id1, id2):
        id1 = self._eparents[id1]
        id2 = self._eparents[id2]
        if id1 == id2:
            return id1

        # Merge the smaller eclass into the larger one
        class1 = self._eclasses[id1]
        class2 = self._eclasses[id2]
        if len(class1) > len(class2):
            id1, id2 = id2, id1
            class1, class2 = class2, class1

        # Do the actual merging and clear the other eclass
        class2 |= class1
        class1.clear()

        # Update the parent pointer
        self._eparents[id1] = id2

        # Remove the eclass from the eclasses dict
        del self._eclasses[id1]

        # Remove the enode from the corresponding etable
        enode = self._enodes.inverse[id1]
        del self._etables[enode.head][id1]

        return id2

    def _match_args(self, args, patargs):
        if len(args) != len(patargs):
            return None

        subst = {}
        for arg, patarg in zip(args, patargs):
            if isinstance(patarg, Variable):
                if isinstance(arg, Atom):
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
        # print()
        # pprint(self._etables)
        # patterns could be reordered to match on the most selective one first
        patterns = dict(reversed(list(pattern.flatten())))
        # print()
        # print("PATTERNS:")
        # pprint(patterns)

        matches = {}
        for auxvar, pattern in patterns.items():
            # print()
            # print(auxvar, "<~", pattern)
            table = self._etables[pattern.head]

            if auxvar is None:
                for id, args in table.items():
                    if (subst := self._match_args(args, pattern.args)) is not None:
                        matches[id] = subst
                #print(matches)
            else:
                newmatches = {}
                for id, subst in matches.items():
                    sid = subst[auxvar.name]
                    if args := table.get(sid):
                        if (newsubst := self._match_args(args, pattern.args)) is not None:
                            newmatches[id] = {**subst, **newsubst}

                matches = newmatches
                #print(matches)

        #print('----------')
        return matches

    def apply(self, rewrites):
        for rewrite in promote_list(rewrites):
            for id, subst in self.match(rewrite.lhs).items():
                enode = rewrite.rhs.substitute(subst)
                if isinstance(enode, ENode):
                    otherid = self.add(enode)
                else:
                    otherid = enode
                self.union(id, otherid)

    def extract(self, node):
        if isinstance(node, Node):
            enode = self._nodes.get(node)
            id = self._enodes[enode]
        elif isinstance(node, ENode):
            id = self._enodes[node]

        return self._create_node(id)


class Atom:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Atom({self.value})"

    def __hash__(self):
        return hash((self.__class__, self.value))


class Variable:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"${self.name}"

    def __hash__(self):
        return hash((self.__class__, self.name))

    def substitute(self, subst):
        return subst[self.name]


class Pattern:
    __slots__ = ("head", "args")
    _counter = itertools.count()

    def __init__(self, head, args):
        self.head = head
        self.args = args

    def __repr__(self):
        argstring = ", ".join(map(repr, self.args))
        return f"{self.head.__name__}({argstring})"

    def __rshift__(self, rhs):
        return Rewrite(self, rhs)

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
                args.append(arg)
        return ENode(self.head, args)


class Rewrite:
    __slots__ = ("lhs", "rhs")

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"{self.lhs} >> {self.rhs}"




# ops.Multiply[a, b] => ops.Add[ops.Multiply[a, b], ops.Multiply[a, b]]
# SyntacticPattern
# DynamicPattern
