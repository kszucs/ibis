from matchpy import (
    Arity,
    Operation,
    Pattern,
    Symbol,
    Wildcard,
    match,
    substitute,
)

import ibis
import ibis.expr.operations as ops

f = Operation.new('f', Arity.binary)

a = Symbol('a')
b = Symbol('b')
subject = f(a, b)

x = Wildcard.dot('x')
y = Wildcard.dot('y')
pattern = Pattern(f(x, y))

substitution = next(match(subject, pattern))

#####################################

x = Wildcard.dot('x')
y = Wildcard.dot('y')

subject = ops.Add(ibis.literal(1), ibis.literal(2))
pattern = ops.Add.pattern(x, right=y)

substitution = next(match(subject, Pattern(pattern)))
print(substitution)
print(substitute(pattern, substitution))
