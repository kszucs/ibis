from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, overload

from koerce import Builder, Call, Deferred, Var, _


def _contains_deferred(obj: Any) -> bool:
    if isinstance(obj, (Builder, Deferred)):
        return True
    elif (typ := type(obj)) in (tuple, list, set):
        return any(_contains_deferred(o) for o in obj)
    elif typ is dict:
        return any(_contains_deferred(o) for o in obj.values())
    return False


F = TypeVar("F", bound=Callable)


@overload
def deferrable(*, repr: str | None = None) -> Callable[[F], F]: ...


@overload
def deferrable(func: F) -> F: ...


def deferrable(func=None, *, repr=None):
    """Wrap a top-level expr function to support deferred arguments.

    When a deferrable function is called, the args & kwargs are traversed to
    look for `Deferred` values (through builtin collections like
    `list`/`tuple`/`set`/`dict`). If any `Deferred` arguments are found, then
    the result is also `Deferred`. Otherwise the function is called directly.

    Parameters
    ----------
    func
        A callable to make deferrable
    repr
        An optional fixed string to use when repr-ing the deferred expression,
        instead of the usual. This is useful for complex deferred expressions
        where the arguments don't necessarily make sense to be user facing
        in the repr.

    """

    def wrapper(func):
        # Parse the signature of func so we can validate deferred calls eagerly,
        # erroring for invalid/missing arguments at call time not resolve time.
        sig = inspect.signature(func)

        @functools.wraps(func)
        def inner(*args, **kwargs):
            if _contains_deferred((args, kwargs)):
                # Try to bind the arguments now, raising a nice error
                # immediately if the function was called incorrectly
                sig.bind(*args, **kwargs)
                builder = Call(func, *args, **kwargs)
                return Deferred(builder)  # , repr=repr)
            return func(*args, **kwargs)

        return inner  # type: ignore

    return wrapper if func is None else wrapper(func)


class _Variable(Var):
    def __repr__(self):
        return self.name


# reserved variable name for the value being matched
_ = Deferred(_Variable("_"))
