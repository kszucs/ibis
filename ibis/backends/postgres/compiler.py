from __future__ import annotations

import string
from functools import partial, reduce, singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
from public import public
from sqlglot.dialects import Postgres
from sqlglot.dialects.dialect import rename_func

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis.backends.base.sqlglot.compiler import NULL, STAR, SQLGlotCompiler, paren
from ibis.backends.base.sqlglot.datatypes import PostgresType
from ibis.expr.rewrites import rewrite_sample

Postgres.Generator.TRANSFORMS |= {
    sge.Map: rename_func("hstore"),
    sge.Split: rename_func("string_to_array"),
    sge.RegexpSplit: rename_func("regexp_split_to_array"),
    sge.DateFromParts: rename_func("make_date"),
    sge.ArraySize: rename_func("cardinality"),
    sge.Pow: rename_func("pow"),
}


class PostgresUDFNode(ops.Value):
    shape = rlz.shape_like("args")


@public
class PostgresCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "postgres"
    type_mapper = PostgresType
    rewrites = rewrite_sample, *SQLGlotCompiler.rewrites
    quoted = True

    NAN = sge.Literal.number("'NaN'::double precision")
    POS_INF = sge.Literal.number("'Inf'::double precision")
    NEG_INF = sge.Literal.number("'-Inf'::double precision")

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    @singledispatchmethod
    def visit_node(self, op, **kwargs):
        return super().visit_node(op, **kwargs)

    @visit_node.register(ops.Mode)
    def visit_Mode(self, op, *, arg, where):
        expr = self.f.mode()
        expr = sge.WithinGroup(
            this=expr,
            expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
        )
        if where is not None:
            expr = sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    def visit_ArgMinMax(self, op, *, arg, key, where, desc: bool):
        conditions = [arg.is_(sg.not_(NULL)), key.is_(sg.not_(NULL))]

        if where is not None:
            conditions.append(where)

        agg = self.agg.array_agg(
            sge.Ordered(this=sge.Order(this=arg, expressions=[key]), desc=desc),
            where=sg.and_(*conditions),
        )
        return paren(agg)[0]

    @visit_node.register(ops.ArgMin)
    def visit_ArgMin(self, op, *, arg, key, where):
        return self.visit_ArgMinMax(op, arg=arg, key=key, where=where, desc=False)

    @visit_node.register(ops.ArgMax)
    def visit_ArgMax(self, op, *, arg, key, where):
        return self.visit_ArgMinMax(op, arg=arg, key=key, where=where, desc=True)

    @visit_node.register(ops.Sum)
    def visit_Sum(self, op, *, arg, where):
        arg = (
            self.cast(self.cast(arg, dt.int32), op.dtype)
            if op.arg.dtype.is_boolean()
            else arg
        )
        return self.agg.sum(arg, where=where)

    @visit_node.register(ops.IsNan)
    def visit_IsNan(self, op, *, arg):
        return arg.eq(self.cast(sge.convert("NaN"), op.arg.dtype))

    @visit_node.register(ops.IsInf)
    def visit_IsInf(self, op, *, arg):
        return arg.isin(self.POS_INF, self.NEG_INF)

    @visit_node.register(ops.CountDistinctStar)
    def visit_CountDistinctStar(self, op, *, where, arg):
        # use a tuple because postgres doesn't accept COUNT(DISTINCT a, b, c, ...)
        #
        # this turns the expression into COUNT(DISTINCT ROW(a, b, c, ...))
        row = sge.Tuple(
            expressions=list(
                map(partial(sg.column, quoted=self.quoted), op.arg.schema.keys())
            )
        )
        return self.agg.count(sge.Distinct(expressions=[row]), where=where)

    @visit_node.register(ops.Correlation)
    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "sample":
            raise com.UnsupportedOperationError(
                f"{self.dialect} only implements `pop` correlation coefficient"
            )

        # TODO: rewrite rule?
        if (left_type := op.left.dtype).is_boolean():
            left = self.cast(left, dt.Int32(nullable=left_type.nullable))

        if (right_type := op.right.dtype).is_boolean():
            right = self.cast(right, dt.Int32(nullable=right_type.nullable))

        return self.agg.corr(left, right, where=where)

    @visit_node.register(ops.ApproxMedian)
    def visit_ApproxMedian(self, op, *, arg, where):
        return self.visit_Median(op, arg=arg, where=where)

    @visit_node.register(ops.Median)
    def visit_Median(self, op, *, arg, where):
        return self.visit_Quantile(op, arg=arg, quantile=sge.convert(0.5), where=where)

    @visit_node.register(ops.ApproxCountDistinct)
    def visit_ApproxCountDistinct(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[arg]), where=where)

    def array_func(self, *args):
        return sge.Anonymous(this=sg.to_identifier("array"), expressions=list(args))

    @visit_node.register(ops.IntegerRange)
    @visit_node.register(ops.TimestampRange)
    def visit_Range(self, op, *, start, stop, step):
        def zero_value(dtype):
            if dtype.is_interval():
                return self.f.make_interval()
            return 0

        def interval_sign(v):
            zero = self.f.make_interval()
            return sge.Case(
                ifs=[
                    self.if_(v.eq(zero), 0),
                    self.if_(v < zero, -1),
                    self.if_(v > zero, 1),
                ],
                default=NULL,
            )

        def _sign(value, dtype):
            if dtype.is_interval():
                return interval_sign(value)
            return self.f.sign(value)

        step_dtype = op.step.dtype
        return self.if_(
            sg.and_(
                self.f.nullif(step, zero_value(step_dtype)).is_(sg.not_(NULL)),
                _sign(step, step_dtype).eq(_sign(stop - start, step_dtype)),
            ),
            self.f.array_remove(
                self.array_func(
                    sg.select(STAR).from_(self.f.generate_series(start, stop, step))
                ),
                stop,
            ),
            self.cast(self.f.array(), op.dtype),
        )

    @visit_node.register(ops.ArrayConcat)
    def visit_ArrayConcat(self, op, *, arg):
        return reduce(self.f.array_cat, map(partial(self.cast, to=op.dtype), arg))

    @visit_node.register(ops.ArrayContains)
    def visit_ArrayContains(self, op, *, arg, other):
        return sge.ArrayContains(
            this=arg, expression=self.f.array(self.cast(other, op.arg.dtype.value_type))
        )

    @visit_node.register(ops.ArrayFilter)
    def visit_ArrayFilter(self, op, *, arg, body, param):
        return self.array_func(
            sg.select(sg.column(param, quoted=self.quoted))
            .from_(sge.Unnest(expressions=[arg], alias=param))
            .where(body)
        )

    @visit_node.register(ops.ArrayMap)
    def visit_ArrayMap(self, op, *, arg, body, param):
        return self.array_func(
            sg.select(body).from_(sge.Unnest(expressions=[arg], alias=param))
        )

    @visit_node.register(ops.ArrayPosition)
    def visit_ArrayPosition(self, op, *, arg, other):
        t = sge.Unnest(expressions=[arg], alias="value", offset=True)
        idx = sg.column("ordinality")
        value = sg.column("value")
        return self.f.coalesce(
            sg.select(idx).from_(t).where(value.eq(other)).limit(1).subquery(), 0
        )

    @visit_node.register(ops.ArraySort)
    def visit_ArraySort(self, op, *, arg):
        return self.array_func(
            sg.select("x").from_(sge.Unnest(expressions=[arg], alias="x")).order_by("x")
        )

    @visit_node.register(ops.ArrayRepeat)
    def visit_ArrayRepeat(self, op, *, arg, times):
        i = sg.to_identifier("i")
        length = self.f.cardinality(arg)
        return self.array_func(
            sg.select(arg[i % length + 1]).from_(
                self.f.generate_series(0, length * times - 1).as_(i.name)
            )
        )

    @visit_node.register(ops.ArrayDistinct)
    def visit_ArrayDistinct(self, op, *, arg):
        return self.if_(
            arg.is_(NULL),
            NULL,
            self.array_func(sg.select(sge.Explode(this=arg)).distinct()),
        )

    @visit_node.register(ops.ArrayUnion)
    def visit_ArrayUnion(self, op, *, left, right):
        return self.array_func(
            sg.union(
                sg.select(sge.Explode(this=left)), sg.select(sge.Explode(this=right))
            )
        )

    @visit_node.register(ops.ArrayIntersect)
    def visit_ArrayIntersect(self, op, *, left, right):
        return self.array_func(
            sg.intersect(
                sg.select(sge.Explode(this=left)), sg.select(sge.Explode(this=right))
            )
        )

    @visit_node.register(ops.Log2)
    def visit_Log2(self, op, *, arg):
        return self.cast(
            self.f.log(
                self.cast(sge.convert(2), dt.decimal),
                arg if op.arg.dtype.is_decimal() else self.cast(arg, dt.decimal),
            ),
            op.dtype,
        )

    @visit_node.register(ops.Log)
    def visit_Log(self, op, *, arg, base):
        if base is not None:
            if not op.base.dtype.is_decimal():
                base = self.cast(base, dt.decimal)
        else:
            base = self.cast(sge.convert(self.f.exp(1)), dt.decimal)

        if not op.arg.dtype.is_decimal():
            arg = self.cast(arg, dt.decimal)
        return self.cast(self.f.log(base, arg), op.dtype)

    @visit_node.register(ops.StructField)
    def visit_StructField(self, op, *, arg, field):
        idx = op.arg.dtype.names.index(field) + 1
        # postgres doesn't have anonymous structs :(
        #
        # this works around ibis not having a way to tell sqlglot to transform
        # an exploded array(row) into the equivalent unnest(t) _ (col1, ..., colN)
        # element
        #
        # but also postgres should really support anonymous structs
        return self.cast(
            self.f.json_extract(self.f.to_jsonb(arg), sge.convert(f"f{idx:d}")),
            op.dtype,
        )

    @visit_node.register(ops.StructColumn)
    def visit_StructColumn(self, op, *, names, values):
        return self.f.row(*map(self.cast, values, op.dtype.types))

    @visit_node.register(ops.ToJSONArray)
    def visit_ToJSONArray(self, op, *, arg):
        return self.if_(
            self.f.json_typeof(arg).eq(sge.convert("array")),
            self.array_func(sg.select(STAR).from_(self.f.json_array_elements(arg))),
            NULL,
        )

    @visit_node.register(ops.Map)
    def visit_Map(self, op, *, keys, values):
        return self.f.map(self.f.array(*keys), self.f.array(*values))

    @visit_node.register(ops.MapLength)
    def visit_MapLength(self, op, *, arg):
        return self.f.cardinality(self.f.akeys(arg))

    @visit_node.register(ops.MapGet)
    def visit_MapGet(self, op, *, arg, key, default):
        return self.if_(self.f.exist(arg, key), self.f.json_extract(arg, key), default)

    @visit_node.register(ops.MapMerge)
    def visit_MapMerge(self, op, *, left, right):
        return sge.DPipe(this=left, expression=right)

    @visit_node.register(ops.TypeOf)
    def visit_TypeOf(self, op, *, arg):
        typ = self.cast(self.f.pg_typeof(arg), dt.string)
        return self.if_(
            typ.eq(sge.convert("unknown")),
            "null" if op.arg.dtype.is_null() else "text",
            typ,
        )

    @visit_node.register(ops.Round)
    def visit_Round(self, op, *, arg, digits):
        if digits is None:
            return self.f.round(arg)

        result = self.f.round(self.cast(arg, dt.decimal), digits)
        if op.arg.dtype.is_decimal():
            return result
        return self.cast(result, dt.float64)

    @visit_node.register(ops.Modulus)
    def visit_Modulus(self, op, *, left, right):
        # postgres doesn't allow modulus of double precision values, so upcast and
        # then downcast later if necessary
        if not op.dtype.is_integer():
            left = self.cast(left, dt.decimal)
            right = self.cast(right, dt.decimal)

        result = left % right
        if op.dtype.is_float64():
            return self.cast(result, dt.float64)
        else:
            return result

    @visit_node.register(ops.RegexExtract)
    def visit_RegexExtract(self, op, *, arg, pattern, index):
        pattern = self.f.concat("(", pattern, ")")
        matches = self.f.regexp_match(arg, pattern)
        return self.if_(arg.rlike(pattern), paren(matches)[index], NULL)

    @visit_node.register(ops.FindInSet)
    def visit_FindInSet(self, op, *, needle, values):
        return self.f.coalesce(
            self.f.array_position(self.f.array(*values), needle),
            0,
        )

    @visit_node.register(ops.StringContains)
    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.strpos(haystack, needle) > 0

    @visit_node.register(ops.EndsWith)
    def visit_EndsWith(self, op, *, arg, end):
        return self.f.right(arg, self.f.length(end)).eq(end)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_binary():
            return self.cast("".join(map(r"\x{:0>2x}".format, value)), dt.binary)
        elif dtype.is_time():
            to_int32 = partial(self.cast, to=dt.int32)
            to_float64 = partial(self.cast, to=dt.float64)

            return self.f.make_time(
                to_int32(value.hour),
                to_int32(value.minute),
                to_float64(value.second + value.microsecond / 1e6),
            )
        elif dtype.is_json():
            return self.cast(value, dt.json)
        return None

    @visit_node.register(ops.TimestampFromYMDHMS)
    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds
    ):
        to_int32 = partial(self.cast, to=dt.int32)
        return self.f.make_timestamp(
            to_int32(year),
            to_int32(month),
            to_int32(day),
            to_int32(hours),
            to_int32(minutes),
            self.cast(seconds, dt.float64),
        )

    @visit_node.register(ops.DateFromYMD)
    def visit_DateFromYMD(self, op, *, year, month, day):
        to_int32 = partial(self.cast, to=dt.int32)
        return self.f.datefromparts(to_int32(year), to_int32(month), to_int32(day))

    @visit_node.register(ops.TimestampBucket)
    def visit_TimestampBucket(self, op, *, arg, interval, offset):
        origin = self.f.make_timestamp(
            *map(partial(self.cast, to=dt.int32), (1970, 1, 1, 0, 0, 0))
        )

        if offset is not None:
            origin += offset

        return self.f.date_bin(interval, arg, origin)

    @visit_node.register(ops.DayOfWeekIndex)
    def visit_DayOfWeekIndex(self, op, *, arg):
        return self.cast(self.f.extract("dow", arg) + 6, dt.int16) % 7

    @visit_node.register(ops.DayOfWeekName)
    def visit_DayOfWeekName(self, op, *, arg):
        return self.f.trim(self.f.to_char(arg, "Day"), string.whitespace)

    @visit_node.register(ops.ExtractSecond)
    def visit_ExtractSecond(self, op, *, arg):
        return self.cast(self.f.floor(self.f.extract("second", arg)), op.dtype)

    @visit_node.register(ops.ExtractMillisecond)
    def visit_ExtractMillisecond(self, op, *, arg):
        return self.cast(
            self.f.floor(self.f.extract("millisecond", arg)) % 1_000, op.dtype
        )

    @visit_node.register(ops.ExtractMicrosecond)
    def visit_ExtractMicrosecond(self, op, *, arg):
        return self.f.extract("microsecond", arg) % 1_000_000

    @visit_node.register(ops.ExtractDayOfYear)
    def visit_ExtractDayOfYear(self, op, *, arg):
        return self.f.extract("doy", arg)

    @visit_node.register(ops.ExtractWeekOfYear)
    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.extract("week", arg)

    @visit_node.register(ops.ExtractEpochSeconds)
    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.f.extract("epoch", arg)

    @visit_node.register(ops.ArrayIndex)
    def visit_ArrayIndex(self, op, *, arg, index):
        index = self.if_(index < 0, self.f.cardinality(arg) + index, index)
        return paren(arg)[index + 1]

    @visit_node.register(ops.ArraySlice)
    def visit_ArraySlice(self, op, *, arg, start, stop):
        neg_to_pos_index = lambda n, index: self.if_(index < 0, n + index, index)

        arg_length = self.f.cardinality(arg)

        if start is None:
            start = 0
        else:
            start = self.f.least(arg_length, neg_to_pos_index(arg_length, start))

        if stop is None:
            stop = arg_length
        else:
            stop = neg_to_pos_index(arg_length, stop)

        slice_expr = sge.Slice(this=start + 1, expression=stop)
        return paren(arg)[slice_expr]

    @visit_node.register(ops.IntervalFromInteger)
    def visit_IntervalFromInteger(self, op, *, arg, unit):
        plural = unit.plural
        if plural == "minutes":
            plural = "mins"
            arg = self.cast(arg, dt.int32)
        elif plural == "seconds":
            plural = "secs"
            arg = self.cast(arg, dt.float64)
        elif plural == "milliseconds":
            plural = "secs"
            arg /= 1e3
        elif plural == "microseconds":
            plural = "secs"
            arg /= 1e6
        elif plural == "nanoseconds":
            plural = "secs"
            arg /= 1e9
        else:
            arg = self.cast(arg, dt.int32)

        key = sg.to_identifier(plural)

        return self.f.make_interval(sge.Kwarg(this=key, expression=arg))

    @visit_node.register(ops.Cast)
    @visit_node.register(ops.TryCast)
    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype

        if from_.is_timestamp() and to.is_integer():
            return self.f.extract("epoch", arg)
        elif from_.is_integer() and to.is_timestamp():
            arg = self.f.to_timestamp(arg)
            if (timezone := to.timezone) is not None:
                arg = self.f.timezone(timezone, arg)
            return arg
        elif from_.is_integer() and to.is_interval():
            unit = to.unit
            return self.visit_IntervalFromInteger(
                ops.IntervalFromInteger(op.arg, unit), arg=arg, unit=unit
            )
        elif from_.is_string() and to.is_binary():
            # Postgres and Python use the words "decode" and "encode" in
            # opposite ways, sweet!
            return self.f.decode(arg, "escape")
        elif from_.is_binary() and to.is_string():
            return self.f.encode(arg, "escape")

        return self.cast(arg, op.to)

    @visit_node.register(ops.RowID)
    @visit_node.register(ops.TimeDelta)
    @visit_node.register(ops.ArrayFlatten)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)


_SIMPLE_OPS = {
    ops.ArrayCollect: "array_agg",
    ops.ArrayRemove: "array_remove",
    ops.BitAnd: "bit_and",
    ops.BitOr: "bit_or",
    ops.BitXor: "bit_xor",
    ops.GeoArea: "st_area",
    ops.GeoAsBinary: "st_asbinary",
    ops.GeoAsEWKB: "st_asewkb",
    ops.GeoAsEWKT: "st_asewkt",
    ops.GeoAsText: "st_astext",
    ops.GeoAzimuth: "st_azimuth",
    ops.GeoBuffer: "st_buffer",
    ops.GeoCentroid: "st_centroid",
    ops.GeoContains: "st_contains",
    ops.GeoContainsProperly: "st_contains",
    ops.GeoCoveredBy: "st_coveredby",
    ops.GeoCovers: "st_covers",
    ops.GeoCrosses: "st_crosses",
    ops.GeoDFullyWithin: "st_dfullywithin",
    ops.GeoDWithin: "st_dwithin",
    ops.GeoDifference: "st_difference",
    ops.GeoDisjoint: "st_disjoint",
    ops.GeoDistance: "st_distance",
    ops.GeoEndPoint: "st_endpoint",
    ops.GeoEnvelope: "st_envelope",
    ops.GeoEquals: "st_equals",
    ops.GeoGeometryN: "st_geometryn",
    ops.GeoGeometryType: "st_geometrytype",
    ops.GeoIntersection: "st_intersection",
    ops.GeoIntersects: "st_intersects",
    ops.GeoIsValid: "st_isvalid",
    ops.GeoLength: "st_length",
    ops.GeoLineLocatePoint: "st_linelocatepoint",
    ops.GeoLineMerge: "st_linemerge",
    ops.GeoLineSubstring: "st_linesubstring",
    ops.GeoNPoints: "st_npoints",
    ops.GeoOrderingEquals: "st_orderingequals",
    ops.GeoOverlaps: "st_overlaps",
    ops.GeoPerimeter: "st_perimeter",
    ops.GeoSRID: "st_srid",
    ops.GeoSetSRID: "st_setsrid",
    ops.GeoSimplify: "st_simplify",
    ops.GeoStartPoint: "st_startpoint",
    ops.GeoTouches: "st_touches",
    ops.GeoTransform: "st_transform",
    ops.GeoUnaryUnion: "st_union",
    ops.GeoUnion: "st_union",
    ops.GeoWithin: "st_within",
    ops.GeoX: "st_x",
    ops.GeoY: "st_y",
    ops.MapContains: "exist",
    ops.MapKeys: "akeys",
    ops.MapValues: "avals",
    ops.RegexSearch: "regexp_like",
    ops.TimeFromHMS: "make_time",
}


for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @PostgresCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @PostgresCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(PostgresCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _name, _fmt
