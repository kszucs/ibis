import collections
import itertools

# class EClass(set[int]):
#     pass
# from rich.pretty import pprint
from pprint import pprint

from ibis.common.graph import Node

# consider to use an EClass(id, nodes) dataclass


class EGraph:
    __slots__ = ("_tables", "_counter", "_nodes", "_parents", "_classes")

    def __init__(self):
        self._nodes = {}
        self._tables = collections.defaultdict(dict)
        self._counter = itertools.count()
        self._parents = {}
        self._classes = {}

    def __repr__(self):
        return f"EGraph({self._classes})"

    def add(self, node):
        if id := self._nodes.get(node):
            return id

        head = type(node)
        args = tuple(
            self.add(arg) if isinstance(arg, Node) else Atom(arg)
            for arg in node.__args__
        )

        id = next(self._counter)
        self._nodes[node] = id
        self._parents[id] = id
        self._classes[id] = {id}
        self._tables[head][id] = args

        return id

    def find(self, id):
        return self._parents[id]

    def union(self, id1, id2):
        id1 = self._parents[id1]
        id2 = self._parents[id2]
        if id1 == id2:
            return id1

        # Merge the smaller eclass into the larger one
        class1 = self._classes[id1]
        class2 = self._classes[id2]
        if len(class1) > len(class2):
            id1, id2 = id2, id1
            class1, class2 = class2, class1

        # Do the actual merging and clear the other eclass
        class2 |= class1
        class1.clear()

        # Update the parent pointer
        self._parents[id1] = id2

        return id2

    def _match_args(self, args, patargs):
        if len(args) != len(patargs):
            return None

        subst = {}
        for arg, patarg in zip(args, patargs):
            print(arg, patarg)
            if isinstance(patarg, Variable):
                if isinstance(arg, Atom):
                    subst[patarg.name] = arg
                else:
                    subst[patarg.name] = self._parents[arg]
            elif isinstance(arg, Atom):
                if patarg != arg.value:
                    return None
            else:
                if self._parents[arg] != self._parents[patarg]:
                    return None

        return subst

    def match(self, pattern):
        print()
        pprint(self._tables)
        patterns = dict(reversed(list(pattern.flatten())))
        print()
        print("PATTERNS:")
        pprint(patterns)

        matches = {}
        for auxvar, pattern in patterns.items():
            print()
            print(auxvar, "<~", pattern)
            table = self._tables[pattern.head]

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


class Rewrite:
    __slots__ = ("pattern", "rewrite")

    def __init__(self, pattern, rewrite):
        self.pattern = pattern
        self.rewrite = rewrite


# SyntacticPattern
# DynamicPattern
