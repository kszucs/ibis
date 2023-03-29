from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    ForwardRef,
    Iterable,
    Mapping,
    Tuple,
    TypeVar,
    Union,
)

import toolz

# TODO(kszucs): try to use inspect.get_annotations() backport instead

if sys.version_info >= (3, 9):

    @toolz.memoize
    def evaluate_typehint(hint, module_name=None) -> Any:
        if isinstance(hint, str):
            hint = ForwardRef(hint)
        if isinstance(hint, ForwardRef):
            if module_name is None:
                globalns = {}
            else:
                globalns = sys.modules[module_name].__dict__
            return hint._evaluate(globalns, locals(), frozenset())
        else:
            return hint

else:

    @toolz.memoize
    def evaluate_typehint(hint, module_name) -> Any:
        if isinstance(hint, str):
            hint = ForwardRef(hint)
        if isinstance(hint, ForwardRef):
            if module_name is None:
                globalns = {}
            else:
                globalns = sys.modules[module_name].__dict__
            return hint._evaluate(globalns, locals())
        else:
            return hint


if TYPE_CHECKING:
    import ibis.expr.datatypes as dt
    import ibis.expr.schema as sch

    SupportsSchema = TypeVar(
        "SupportsSchema",
        Iterable[Tuple[str, Union[str, dt.DataType]]],
        Mapping[str, Union[str, dt.DataType]],
        sch.Schema,
    )


class CoercionError(Exception):
    ...


# TODO(kszucs): consider to move this to ibis.common.typing
class Coercible(ABC):
    """Protocol for defining coercible types.

    Coercible types define a special ``__coerce__`` method that accepts an object
    with an instance of the type. Used in conjunction with the ``coerced_to``
    pattern to coerce arguments to a specific type.
    """

    __slots__ = ()

    @classmethod
    @abstractmethod
    def __coerce__(cls, value, *typevars):
        # not typevars but patterns from now on
        ...

    # # similar to __instancecheck__ but takes an instance and the type vars
    # @classmethod
    # def __verify__(cls, instance, *typevars):
    #     # perhaps call it __matches__?
    #     # not typevars but patterns from now on
    #     return True
