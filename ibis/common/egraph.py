import collections
import itertools

# class EClass(set[int]):
#     pass
# from rich.pretty import pprint
from pprint import pprint

from ibis.common.graph import Node

# consider to use an EClass(id, nodes) dataclass

class ENode:
    __slots__ = ("head", "args")

    def __init__(self, head, args):
        self.head = head
        self.args = args


class EGraph:
    __slots__ = ("_nodes", "_counter", "_enodes", "_eparents", "_eclasses", "_etables")

    def __init__(self):
        # store the nodes before converting them to enodes, so we can spare the initial
        # node traversal and omit the creation of enodes
        self._nodes = {}
        # counter for generating new eclass ids
        self._counter = itertools.count()
        # map enodes to eclass ids so we can check if an enode is already in the egraph
        self._enodes = {}
        # map eclass ids to their parent eclass id, this is required for the union-find
        self._eparents = {}
        # map eclass ids to their eclass
        self._eclasses = {}
        # map enode heads to their eclass ids and their arguments, this is required for
        # the relational e-matching
        self._etables = collections.defaultdict(dict)

    def __repr__(self):
        return f"EGraph({self._eclasses})"

    def _create_enode(self, node):
        head = type(node)
        args = tuple(
            self.add(arg) if isinstance(arg, Node) else Atom(arg)
            for arg in node.__args__
        )
        return ENode(head, args)

    def add(self, node):
        if isinstance(node, Node):
            enode = self._nodes.get(node) or self._create_enode(node)
            self._nodes[node] = enode
        elif isinstance(node, ENode):
            enode = node
        else:
            raise TypeError("`node` must be an instance of ibis.common.graph.Node")

        if id := self._enodes.get(enode):
            return id

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
        print()
        pprint(self._etables)
        patterns = dict(reversed(list(pattern.flatten())))
        print()
        print("PATTERNS:")
        pprint(patterns)

        matches = {}
        for auxvar, pattern in patterns.items():
            print()
            print(auxvar, "<~", pattern)
            table = self._etables[pattern.head]

            if auxvar is None:
                for id, args in table.items():
                    if subst := self._match_args(args, pattern.args):
                        matches[id] = subst
                print(matches)
            else:
                newmatches = {}
                for id, subst in matches.items():
                    sid = subst[auxvar.name]
                    if args := table.get(sid):
                        if newsubst := self._match_args(args, pattern.args):
                            newmatches[id] = {**subst, **newsubst}

                matches = newmatches
                print(matches)

        print('----------')
        return matches

    def apply(self, rewrite):
        for id, subst in self.match(rewrite.lhs).items():
            new = rewrite.rhs.substitute(subst)
            newid = self._add(new.head, new.args)
            self.union(id, newid)


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


class Pattern:
    __slots__ = ("head", "args")
    _counter = itertools.count()

    def __init__(self, head, args):
        self.head = head
        self.args = args

    def __repr__(self):
        argstring = ", ".join(map(repr, self.args))
        return f"{self.head.__name__}({argstring})"

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
            if isinstance(arg, Variable):
                args.append(subst[arg.name])
            elif isinstance(arg, Pattern):
                args.append(arg.substitute(subst))
            else:
                args.append(arg)
        return ENode(self.head, args)


class Rewrite:
    __slots__ = ("lhs", "rhs")

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs



# ops.Multiply[a, b] => ops.Add[ops.Multiply[a, b], ops.Multiply[a, b]]
# SyntacticPattern
# DynamicPattern
