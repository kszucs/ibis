from typing import Any

from rich.pretty import pprint

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.egraph import EGraph, Pattern, Variable
from ibis.common.grounds import Annotable

one = ibis.literal(1)
two = one * 2
two_ = one + one
three = one + two
six = three * two_
seven = six + 1
seven_ = seven * 1

a, b, c = Variable('a'), Variable('b'), Variable('c')


def test_simple():
    print()
    pprint(seven_.op())

    eg = EGraph()
    eg.add(seven_.op())
    print(eg)

    p = ops.Add[a, ops.Multiply[b, ops.Literal[c, dt.int8]]]
    print(eg.match(p))

    p = ops.Multiply[a, 1]
    result = eg.match(p)
    print(result)
