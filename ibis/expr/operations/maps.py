from public import public

from ...common import exceptions as com
from .. import datatypes as dt
from .. import rules as rlz
from .core import UnaryOp, ValueOp


@public
class MapLength(UnaryOp):
    arg = rlz.mapping
    output_dtype = dt.int64


@public
class MapValueForKey(ValueOp):
    arg = rlz.mapping
    key = rlz.one_of([rlz.string, rlz.integer])

    @property
    def output_dtype(self):
        return self.arg.type().value_type


@public
class MapValueOrDefaultForKey(ValueOp):
    arg = rlz.mapping
    key = rlz.one_of([rlz.string, rlz.integer])
    default = rlz.any

    @property
    def output_dtype(self):
        value_type = self.arg.type().value_type
        default_type = self.default.type()

        if not dt.same_kind(default_type, value_type):
            raise com.IbisTypeError(
                "Default value\n{}\nof type {} cannot be cast to map's value "
                "type {}".format(self.default, default_type, value_type)
            )

        return dt.highest_precedence((default_type, value_type))


@public
class MapKeys(UnaryOp):
    arg = rlz.mapping

    @property
    def output_dtype(self):
        return dt.Array(self.arg.type().key_type)


@public
class MapValues(UnaryOp):
    arg = rlz.mapping

    @property
    def output_dtype(self):
        return dt.Array(self.arg.type().value_type)


# TODO(kszucs): BinaryOp could be more appropiate
@public
class MapConcat(ValueOp):
    left = rlz.mapping
    right = rlz.mapping

    output_shape = rlz.shape_like("args")
    output_dtype = rlz.dtype_like("args")
