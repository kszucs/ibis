import functools
import operator

import datafusion as df
import datafusion.functions
import pyarrow as pa

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.datafusion.datatypes import to_pyarrow_type


@functools.singledispatch
def translate(expr, **kwargs):
    raise NotImplementedError(expr)


@translate.register(ops.Node)
def operation(op, **kwargs):
    raise com.OperationNotDefinedError(f'No translation rule for {type(op)}')


@translate.register(ops.UnboundTable)
def unbound_table(op, context, **kwargs):
    return context.table(op.name)


@translate.register(ops.DatabaseTable)
def table(op, **kwargs):
    name, _, client = op.args
    return client._context.table(name)


@translate.register(ops.Alias)
def alias(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return arg.alias(op.name)


@translate.register(ops.Literal)
def literal(op, **kwargs):
    if isinstance(op.value, (set, frozenset)):
        value = list(op.value)
    else:
        value = op.value

    arrow_type = to_pyarrow_type(op.dtype)
    arrow_scalar = pa.scalar(value, type=arrow_type)

    return df.literal(arrow_scalar)


@translate.register(ops.Cast)
def cast(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    typ = to_pyarrow_type(op.to)
    return arg.cast(to=typ)


@translate.register(ops.TableColumn)
def column(op, **kwargs):
    table_op = op.table

    if hasattr(table_op, "name"):
        return df.column(f'{table_op.name}."{op.name}"')
    else:
        return df.column(op.name)


@translate.register(ops.SortKey)
def sort_key(op, **kwargs):
    arg = translate(op.expr, **kwargs)
    return arg.sort(ascending=op.ascending)


@translate.register(ops.Selection)
def selection(op, **kwargs):
    plan = translate(op.table, **kwargs)

    selections = []
    for arg in op.selections or [op.table]:
        # TODO(kszucs) it would be nice if we wouldn't need to handle the
        # specific cases in the backend implementations, we could add a
        # new operator which retrieves all of the Table columns
        # (.e.g. Asterisk) so the translate() would handle this
        # automatically
        if isinstance(arg, ops.TableNode):
            for name in arg.schema.names:
                column = ops.TableColumn(table=arg, name=name)
                field = translate(column, **kwargs)
                selections.append(field)
        elif isinstance(arg, ops.Value):
            field = translate(arg, **kwargs)
            selections.append(field)
        else:
            raise com.TranslationError(
                "DataFusion backend is unable to compile selection with "
                f"operation type of {type(arg)}"
            )

    plan = plan.select(*selections)

    if op.predicates:
        predicates = map(translate, op.predicates)
        predicate = functools.reduce(operator.and_, predicates)
        plan = plan.filter(predicate)

    if op.sort_keys:
        sort_keys = map(translate, op.sort_keys)
        plan = plan.sort(*sort_keys)

    return plan


@translate.register(ops.Aggregation)
def aggregation(op, **kwargs):
    table = translate(op.table)
    group_by = [translate(arg, **kwargs) for arg in op.by]
    metrics = [translate(arg, **kwargs) for arg in op.metrics]

    if op.predicates:
        table = table.filter(
            functools.reduce(
                operator.and_,
                map(translate, op.predicates),
            )
        )

    return table.aggregate(group_by, metrics)


@translate.register(ops.Not)
def invert(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return ~arg


@translate.register(ops.And)
def and_(op, **kwargs):
    left = translate(op.left, **kwargs)
    right = translate(op.right, **kwargs)
    return left & right


@translate.register(ops.Or)
def or_(op, **kwargs):
    left = translate(op.left, **kwargs)
    right = translate(op.right, **kwargs)
    return left | right


@translate.register(ops.Abs)
def abs(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.abs(arg)


@translate.register(ops.Ceil)
def ceil(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.ceil(arg).cast(pa.int64())


@translate.register(ops.Floor)
def floor(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.floor(arg).cast(pa.int64())


@translate.register(ops.Round)
def round(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    if op.digits is not None:
        raise com.UnsupportedOperationError(
            'Rounding to specific digits is not supported in datafusion'
        )
    return df.functions.round(arg).cast(pa.int64())


@translate.register(ops.Ln)
def ln(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.ln(arg)


@translate.register(ops.Log2)
def log2(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.log2(arg)


@translate.register(ops.Log10)
def log10(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.log10(arg)


@translate.register(ops.Sqrt)
def sqrt(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.sqrt(arg)


@translate.register(ops.Strip)
def strip(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.trim(arg)


@translate.register(ops.LStrip)
def lstrip(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.ltrim(arg)


@translate.register(ops.RStrip)
def rstrip(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.rtrim(arg)


@translate.register(ops.Lowercase)
def lower(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.lower(arg)


@translate.register(ops.Uppercase)
def upper(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.upper(arg)


@translate.register(ops.Reverse)
def reverse(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.reverse(arg)


@translate.register(ops.StringLength)
def strlen(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.character_length(arg)


@translate.register(ops.Capitalize)
def capitalize(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.initcap(arg)


@translate.register(ops.Substring)
def substring(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    start = translate(ops.Add(left=op.start, right=1), **kwargs)
    length = translate(op.length, **kwargs)
    return df.functions.substr(arg, start, length)


@translate.register(ops.RegexExtract)
def regex_extract(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    pattern = translate(op.pattern, **kwargs)
    return df.functions.regexp_match(arg, pattern)


@translate.register(ops.Repeat)
def repeat(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    times = translate(op.times, **kwargs)
    return df.functions.repeat(arg, times)


@translate.register(ops.LPad)
def lpad(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    length = translate(op.length, **kwargs)
    pad = translate(op.pad, **kwargs)
    return df.functions.lpad(arg, length, pad)


@translate.register(ops.RPad)
def rpad(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    length = translate(op.length, **kwargs)
    pad = translate(op.pad, **kwargs)
    return df.functions.rpad(arg, length, pad)


@translate.register(ops.GreaterEqual)
def ge(op, **kwargs):
    return translate(op.left, **kwargs) >= translate(op.right, **kwargs)


@translate.register(ops.LessEqual)
def le(op, **kwargs):
    return translate(op.left, **kwargs) <= translate(op.right, **kwargs)


@translate.register(ops.Greater)
def gt(op, **kwargs):
    return translate(op.left, **kwargs) > translate(op.right, **kwargs)


@translate.register(ops.Less)
def lt(op, **kwargs):
    return translate(op.left, **kwargs) < translate(op.right, **kwargs)


@translate.register(ops.Equals)
def eq(op, **kwargs):
    return translate(op.left, **kwargs) == translate(op.right, **kwargs)


@translate.register(ops.NotEquals)
def ne(op, **kwargs):
    return translate(op.left, **kwargs) != translate(op.right, **kwargs)


@translate.register(ops.Add)
def add(op, **kwargs):
    return translate(op.left, **kwargs) + translate(op.right, **kwargs)


@translate.register(ops.Subtract)
def sub(op, **kwargs):
    return translate(op.left, **kwargs) - translate(op.right, **kwargs)


@translate.register(ops.Multiply)
def mul(op, **kwargs):
    return translate(op.left, **kwargs) * translate(op.right, **kwargs)


@translate.register(ops.Divide)
def div(op, **kwargs):
    return translate(op.left, **kwargs) / translate(op.right, **kwargs)


@translate.register(ops.FloorDivide)
def floordiv(op, **kwargs):
    return df.functions.floor(
        translate(op.left, **kwargs) / translate(op.right, **kwargs)
    )


@translate.register(ops.Modulus)
def mod(op, **kwargs):
    return translate(op.left, **kwargs) % translate(op.right, **kwargs)


@translate.register(ops.Count)
def count(op, **kwargs):
    return df.functions.count(translate(op.arg, **kwargs))


@translate.register(ops.CountStar)
def count_star(_, **kwargs):
    return df.functions.count(df.literal(1))


@translate.register(ops.Sum)
def sum(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.sum(arg)


@translate.register(ops.Min)
def min(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.min(arg)


@translate.register(ops.Max)
def max(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.max(arg)


@translate.register(ops.Mean)
def mean(op, **kwargs):
    arg = translate(op.arg, **kwargs)
    return df.functions.avg(arg)


@translate.register(ops.NodeList)
def value_list(op, **kwargs):
    return [translate(value, **kwargs) for value in op.values]


@translate.register(ops.Contains)
def contains(op, **kwargs):
    value = translate(op.value, **kwargs)
    options = translate(op.options, **kwargs)
    return df.functions.in_list(value, options, negated=False)


@translate.register(ops.NotContains)
def not_contains(op, **kwargs):
    value = translate(op.value, **kwargs)
    options = translate(op.options, **kwargs)
    return df.functions.in_list(value, options, negated=True)


@translate.register(ops.Negate)
def negate(op, **kwargs):
    return df.lit(-1) * translate(op.arg, **kwargs)


@translate.register(ops.Acos)
@translate.register(ops.Asin)
@translate.register(ops.Atan)
@translate.register(ops.Cos)
@translate.register(ops.Sin)
@translate.register(ops.Tan)
def trig(op, **kwargs):
    func_name = op.__class__.__name__.lower()
    func = getattr(df.functions, func_name)
    return func(translate(op.arg, **kwargs))


@translate.register(ops.Atan2)
def atan2(op, **kwargs):
    y, x = (translate(arg, **kwargs) for arg in op.args)
    return df.functions.atan(y / x)


@translate.register(ops.Cot)
def cot(op, **kwargs):
    x = translate(op.arg, **kwargs)
    return df.functions.cos(x) / df.functions.sin(x)


@translate.register(ops.ElementWiseVectorizedUDF)
def elementwise_udf(op, **kwargs):
    udf = df.udf(
        op.func,
        input_types=list(map(to_pyarrow_type, op.input_type)),
        return_type=to_pyarrow_type(op.return_type),
        volatility="volatile",
    )
    args = map(translate, op.func_args)

    return udf(*args)
