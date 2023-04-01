from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TypeVar


def get_type_hints(obj):
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


def discover_typehints(obj):
    annotations = {}
    for name in dir(obj):
        attr = getattr(obj, name)
        if isinstance(attr, property):
            if annot := attr.fget.__annotations__.get("return"):
                annotations[name] = annot

    module_name = getattr(obj, '__module__', None)
    return evaluate_annotations(annotations, module_name, localns=vars(obj))

# raise is parameter cannot be identified (missing type annotation from the discovery)
def bind_typevars(origin, args):
    names = (p.__name__ for p in getattr(origin, "__parameters__", ()))
    params = dict(zip(names, args))

    typehints = get_type_hints(origin)
    extrahints = discover_typehints(origin)

    allhints = {**typehints, **extrahints}

    fields = {}
    for name, typehint in allhints.items():
        if isinstance(typehint, TypeVar):
            fields[name] = params[typehint.__name__]
    return fields


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

    # TODO(kszucs): remove typevars
    @classmethod
    @abstractmethod
    def __coerce__(cls, value, *typevars):
        # not typevars but patterns from now on
        ...
