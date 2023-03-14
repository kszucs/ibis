from typing import Any

from rich.pretty import pprint
#from pprint import pprint

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.egraph import EGraph, Pattern, Variable, Rewrite, Atom
from ibis.common.grounds import Annotable

one = ibis.literal(1)
two = one * 2
two_ = one + one
three = one + two
six = three * two_
seven = six + 1
seven_ = seven * 1
eleven = seven_ + 4

a, b, c = Variable('a'), Variable('b'), Variable('c')

# e = seven_
# for i in range(10):
#     e = e + two
#     e = e - two * 1

# seven_ = e

def test_simple():
    op = eleven.op()
    print()
    pprint(op)

    eg = EGraph()
    eg.add(op)



    # p = ops.Add[a, ops.Multiply[b, ops.Literal[c, dt.int8]]]
    # print(eg.match(p))

    # p = ops.Multiply[a, 1]
    # result = eg.match(p)
    # print(result)

    # print(ops.Multiply[a, 1] >> a)


    # r = ops.Multiply[a, 1] >> ops.Multiply[1, a]
    # r2 = ops.Multiply[a, 1] >> a
    # eg.apply(r)

    # print(eg)
    # for i in range(1000):
    #     eg.apply([r, r2])
    # print(eg)
    # print(eg._etables)
    p = ops.Multiply[a, ops.Literal[1, dt.int8]]

    r3 = ops.Multiply[a, ops.Literal[1, dt.int8]] >> a
    print(eg.match(ops.Multiply[a, ops.Literal[1, dt.int8]]))


    eg.apply(r3)
    #eg.apply(r3)
    # eg.apply(r3)
    # eg.apply(r3)

    print()
    pprint(eg.extract(op))
