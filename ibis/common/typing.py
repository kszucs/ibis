from __future__ import annotations

import collections.abc
import io
import sys
import typing
from abc import ABC, abstractmethod
from itertools import zip_longest
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
from typing_extensions import get_args, get_origin

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


def get_class_annotations(obj):
    annotations = getattr(obj, '__annotations__', {})
    module_name = getattr(obj, '__module__', None)
    return evaluate_annotations(annotations, module_name, localns=vars(obj))


def evaluate_annotations(annots, module_name, localns=None):
    module = sys.modules.get(module_name, None)
    globalns = getattr(module, '__dict__', None)
    return {
        k: eval(v, globalns, localns) if isinstance(v, str) else v
        for k, v in annots.items()
    }


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


# def coerce()

# NoneType = type(None)

# _normalize_mapping = {
#     typing.List: list,
#     typing.Dict: dict,
#     typing.Set: set,
#     typing.Tuple: tuple,
#     typing.Sequence: collections.abc.Sequence,
#     None: NoneType,
#     typing.ByteString: bytes,
# }


# def normalize(t):
#     return _normalize_mapping.get(t, t)


# empty = object()


# def _are_args_subtypes(type, parent):
#     type_args = get_args(type)
#     parent_args = get_args(parent)
#     for type_arg, parent_arg in zip_longest(type_args, parent_args, fillvalue=empty):
#         if type_arg is empty:
#             return False
#         if parent_arg is not empty and not issubtype(type_arg, parent_arg):
#             return False
#     return True


# def issubtype(type, parent):
#     print("========")
#     print(type, parent)
#     type, parent = normalize(type), normalize(parent)

#     if type is parent:
#         return True

#     if parent is Any:
#         return True

#     type_origin, parent_origin = get_origin(type), get_origin(parent)

#     if type_origin is None and parent_origin is None:
#         return issubclass(type, parent)
#     elif type_origin is None:
#         return False
#     elif parent_origin is None:
#         return False
#     else:
#         return _are_args_subtypes(type, parent)

#     #  return False
#     # if type_origin is None and parent_origin is None:
#     #     return issubclass(type, parent)
#     # elif type_origin is None:
#     #     return False
#     # elif parent_origin is None:
#     #     return False

#     # # if parent_origin is typing.Union:
#     # #     return any(issubtype(type, arg) for arg in get_args(parent))
#     # if issubtype(type_origin, parent_origin):
#     #     type_args = get_args(type)
#     #     parent_args = get_args(parent)
#     #     return _are_args_subtypes(type_args, parent_args)
#     # else:
#     #     return False
