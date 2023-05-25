from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import ibis.expr.datatypes as dt
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict, MapSet
from ibis.common.dispatch import lazy_singledispatch
from ibis.common.exceptions import InputTypeError, IntegrityError
from ibis.common.grounds import Concrete
from ibis.common.validators import Coercible
from ibis.util import deprecated, indent

if TYPE_CHECKING:
    import pandas as pd


class Schema(Concrete, Coercible, MapSet):
    """An object for holding table schema information."""

    fields: FrozenDict[str, dt.DataType]
    """A mapping of [`str`][str] to [`DataType`][ibis.expr.datatypes.DataType] objects
    representing the type of each column."""

    def __repr__(self) -> str:
        space = 2 + max(map(len, self.names), default=0)
        return "ibis.Schema {{{}\n}}".format(
            indent(
                ''.join(
                    f'\n{name.ljust(space)}{type!s}' for name, type in self.items()
                ),
                2,
            )
        )

    def __rich_repr__(self):
        for name, dtype in self.items():
            yield name, str(dtype)

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def __getitem__(self, name: str) -> dt.DataType:
        return self.fields[name]

    @classmethod
    def __coerce__(cls, value) -> Schema:
        return schema(value)

    @attribute.default
    def names(self):
        return tuple(self.keys())

    @attribute.default
    def types(self):
        return tuple(self.values())

    @attribute.default
    def _name_locs(self) -> dict[str, int]:
        return {v: i for i, v in enumerate(self.names)}

    def equals(self, other: Schema) -> bool:
        """Return whether `other` is equal to `self`.

        Parameters
        ----------
        other
            Schema to compare `self` to.

        Examples
        --------
        >>> import ibis
        >>> first = ibis.schema({"a": "int"})
        >>> second = ibis.schema({"a": "int"})
        >>> assert first.equals(second)
        >>> third = ibis.schema({"a": "array<int>"})
        >>> assert not first.equals(third)
        """
        if not isinstance(other, Schema):
            raise TypeError(
                f"invalid equality comparison between Schema and {type(other)}"
            )
        return self.__cached_equals__(other)

    @classmethod
    def from_tuples(
        cls,
        values: Iterable[tuple[str, str | dt.DataType]],
    ) -> Schema:
        """Construct a `Schema` from an iterable of pairs.

        Parameters
        ----------
        values
            An iterable of pairs of name and type.

        Returns
        -------
        Schema
            A new schema

        Examples
        --------
        >>> import ibis
        >>> ibis.Schema.from_tuples([("a", "int"), ("b", "string")])
        ibis.Schema {
          a  int64
          b  string
        }
        """
        pairs = list(values)
        if len(pairs) == 0:
            return cls({})

        names, types = zip(*pairs)

        # validate unique field names
        name_locs = {v: i for i, v in enumerate(names)}
        if len(name_locs) < len(names):
            duplicate_names = list(names)
            for v in name_locs:
                duplicate_names.remove(v)
            raise IntegrityError(f'Duplicate column name(s): {duplicate_names}')

        # construct the schema
        return cls(dict(zip(names, types)))

    @classmethod
    def from_numpy(cls, numpy_schema):
        """Return the equivalent ibis schema."""
        from ibis.formats.numpy import schema_from_numpy

        return schema_from_numpy(numpy_schema)

    @classmethod
    def from_pandas(cls, pandas_schema):
        """Return the equivalent ibis schema."""
        from ibis.formats.pandas import schema_from_pandas

        return schema_from_pandas(pandas_schema)

    @classmethod
    def from_pyarrow(cls, pyarrow_schema):
        """Return the equivalent ibis schema."""
        from ibis.formats.pyarrow import schema_from_pyarrow

        return schema_from_pyarrow(pyarrow_schema)

    @classmethod
    def from_dask(cls, dask_schema):
        """Return the equivalent ibis schema."""
        return cls.from_pandas(dask_schema)

    def to_numpy(self):
        """Return the equivalent numpy dtypes."""
        from ibis.formats.numpy import schema_to_numpy

        return schema_to_numpy(self)

    def to_pandas(self):
        """Return the equivalent pandas datatypes."""
        from ibis.formats.pandas import schema_to_pandas

        return schema_to_pandas(self)

    def to_pyarrow(self):
        """Return the equivalent pyarrow schema."""
        from ibis.formats.pyarrow import schema_to_pyarrow

        return schema_to_pyarrow(self)

    def to_dask(self):
        """Return the equivalent dask dtypes."""
        return self.to_pandas()

    def as_struct(self) -> dt.Struct:
        return dt.Struct(self)

    def name_at_position(self, i: int) -> str:
        """Return the name of a schema column at position `i`.

        Parameters
        ----------
        i
            The position of the column

        Returns
        -------
        str
            The name of the column in the schema at position `i`.

        Examples
        --------
        >>> import ibis
        >>> sch = ibis.Schema({"a": "int", "b": "string"})
        >>> sch.name_at_position(0)
        'a'
        >>> sch.name_at_position(1)
        'b'
        """
        return self.names[i]

    @deprecated(
        as_of="6.0",
        instead="use ibis.formats.pandas.convert_pandas_dataframe() instead",
    )
    def apply_to(self, df: pd.DataFrame) -> pd.DataFrame:
        from ibis.formats.pandas import convert_pandas_dataframe

        return convert_pandas_dataframe(df, self)


@lazy_singledispatch
def schema(value: Any) -> Schema:
    """Construct ibis schema from schema-like python objects."""
    raise InputTypeError(value)


@lazy_singledispatch
def infer(value: Any, schema=None) -> Schema:
    """Infer the corresponding ibis schema for a python object."""
    raise InputTypeError(value)


@schema.register(Schema)
def from_schema(s):
    return s


@schema.register(Mapping)
def from_mapping(d):
    return Schema(d)


@schema.register(Iterable)
def from_pairs(lst):
    return Schema.from_tuples(lst)


@schema.register(type)
def from_class(cls):
    return Schema(dt.dtype(cls))


@schema.register("pandas.Series")
def from_pandas_series(s):
    from ibis.formats.pandas import schema_from_pandas

    return schema_from_pandas(s)


@schema.register("pyarrow.Schema")
def from_pyarrow_schema(schema):
    from ibis.formats.pyarrow import schema_from_pyarrow

    return schema_from_pyarrow(schema)


@infer.register("pandas.DataFrame")
def infer_pandas_dataframe(df, schema=None):
    from ibis.formats.pandas import schema_from_pandas_dataframe

    return schema_from_pandas_dataframe(df, schema)


# TODO(kszucs): do we really need the schema kwarg?
@infer.register("pyarrow.Table")
def infer_pyarrow_table(table, schema=None):
    from ibis.formats.pyarrow import schema_from_pyarrow

    schema = schema if schema is not None else table.schema
    return schema_from_pyarrow(schema)


# lock the dispatchers to avoid adding new implementations
del infer.register
del schema.register
