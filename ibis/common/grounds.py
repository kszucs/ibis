from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from functools import partial
from operator import add, attrgetter, mul, sub
from typing import Any, Hashable
from weakref import WeakValueDictionary

from etuples.core import ExpressionTuple, etuple
from kanren import conde, eq, lall, lany, run
from kanren.constraints import isinstanceo
from kanren.graph import mapo, reduceo, walko
from kanren.term import applyo
from unification import var
from unification.core import _reify, _unify, construction_sentinel

from ibis.common.caching import WeakCache
from ibis.common.validators import ImmutableProperty, Optional, Validator
from ibis.util import frozendict

EMPTY = inspect.Parameter.empty  # marker for missing argument


class BaseMeta(ABCMeta):

    __slots__ = ()

    def __call__(cls, *args, **kwargs):
        return cls.__create__(*args, **kwargs)


class Base(metaclass=BaseMeta):

    __slots__ = ()

    @classmethod
    def __create__(cls, *args, **kwargs):
        return type.__call__(cls, *args, **kwargs)


class Immutable:

    __slots__ = ()

    def __setattr__(self, name: str, _: Any) -> None:
        raise TypeError(
            f"Attribute {name!r} cannot be assigned to immutable instance of "
            f"type {type(self)}"
        )


class Parameter(inspect.Parameter):
    """
    Augmented Parameter class to additionally hold a validator object.
    """

    __slots__ = ('_validator',)

    def __init__(
        self,
        name,
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        *,
        validator=EMPTY,
    ):
        super().__init__(
            name,
            kind,
            default=None if isinstance(validator, Optional) else EMPTY,
        )
        self._validator = validator

    @property
    def validator(self):
        return self._validator

    def validate(self, this, arg):
        if self.validator is EMPTY:
            return arg
        else:
            # TODO(kszucs): use self._validator
            return self.validator(arg, this=this)


class AnnotableMeta(BaseMeta):
    """
    Metaclass to turn class annotations into a validatable function signature.
    """

    __slots__ = ()

    def __new__(metacls, clsname, bases, dct):
        # inherit from parent signatures
        params = {}
        properties = {}
        for parent in bases:
            try:
                params.update(parent.__signature__.parameters)
            except AttributeError:
                pass
            try:
                properties.update(parent.__properties__)
            except AttributeError:
                pass

        # store the originally inherited keys so we can reorder later
        inherited = set(params.keys())

        # collect the newly defined parameters
        slots = list(dct.pop('__slots__', []))
        attribs = {}
        for name, attrib in dct.items():
            if isinstance(attrib, Validator):
                # so we can set directly
                params[name] = Parameter(name, validator=attrib)
                slots.append(name)
            elif isinstance(attrib, ImmutableProperty):
                properties[name] = attrib
                slots.append(name)
            else:
                attribs[name] = attrib

        # mandatory fields without default values must preceed the optional
        # ones in the function signature, the partial ordering will be kept
        new_args, new_kwargs = [], []
        inherited_args, inherited_kwargs = [], []

        for name, param in params.items():
            if name in inherited:
                if param.default is EMPTY:
                    inherited_args.append(param)
                else:
                    inherited_kwargs.append(param)
            else:
                if param.default is EMPTY:
                    new_args.append(param)
                else:
                    new_kwargs.append(param)

        signature = inspect.Signature(
            inherited_args + new_args + new_kwargs + inherited_kwargs
        )

        attribs["__slots__"] = tuple(slots)
        attribs["__signature__"] = signature
        attribs["__properties__"] = properties
        # TODO(kszucs): rename to __argnames__
        attribs["argnames"] = tuple(signature.parameters.keys())
        return super().__new__(metacls, clsname, bases, attribs)

    def __getitem__(cls, args):
        bound = cls.__signature__.bind(*args)
        bound.apply_defaults()
        return Pattern(cls, bound.arguments)


class Annotable(Base, Hashable, Immutable, metaclass=AnnotableMeta):
    """Base class for objects with custom validation rules."""

    __slots__ = ("args", "_hash")

    @classmethod
    def __create__(cls, *args, **kwargs):
        # COULD CHECK HERE if any if the args is a unification var()/Pattern()
        # then we can bypass validation entirely
        bound = cls.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()

        # bound the signature to the passed arguments and apply the validators
        # before passing the arguments, so self.__init__() receives already
        # validated arguments as keywords
        kwargs = {}
        for name, value in bound.arguments.items():
            param = cls.__signature__.parameters[name]
            # TODO(kszucs): provide more error context on failure
            kwargs[name] = param.validate(kwargs, value)

        # construct the instance by passing the validated keyword arguments
        return super().__create__(**kwargs)

    @classmethod
    def pattern(cls, *args, **kwargs):
        bound = cls.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        return Pattern(cls, bound.arguments)

    match = pattern

    def __init__(self, **kwargs):
        # set the already validated fields using object.__setattr__ since we
        # treat the annotable instances as immutable objects
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)

        # optimizations to store frequently accessed generic properties
        args = tuple(kwargs[name] for name in self.argnames)
        # TODO(kszucs): rename to __args__
        object.__setattr__(self, "args", args)
        # TODO(kszucs): rename to __precomputed_hash__
        object.__setattr__(self, "_hash", hash((self.__class__, args)))

        # calculate special property-like objects only once due to the
        # immutable nature of annotable instances
        for name, prop in self.__properties__.items():
            object.__setattr__(self, name, prop(self))

        # any supplemental custom code provided by descendant classes
        self.__post_init__()

    def __post_init__(self):
        pass

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return super().__eq__(other)

    def __repr__(self) -> str:
        args = ", ".join(
            f"{name}={value!r}"
            for name, value in zip(self.argnames, self.args)
        )
        return f"{self.__class__.__name__}({args})"

    @classmethod
    def _reconstruct(cls, kwargs):
        # bypass Annotable.__construct__() when deserializing
        self = cls.__new__(cls)
        self.__init__(**kwargs)
        return self

    def __reduce__(self):
        kwargs = dict(zip(self.argnames, self.args))
        return (self._reconstruct, (kwargs,))

    # TODO(kszucs): consider to make a separate mixin class for this
    def copy(self, **overrides):
        kwargs = dict(zip(self.argnames, self.args))
        newargs = {**kwargs, **overrides}
        return self.__class__(**newargs)


class Weakrefable(Base):

    __slots__ = ('__weakref__',)


class Singleton(Weakrefable):
    # NOTE: this only considers the input arguments, when combined with
    # Annotable base class Singleton must come after in the MRO

    __slots__ = ()
    __instances__ = WeakValueDictionary()

    @classmethod
    def __create__(cls, *args, **kwargs):
        key = (cls, args, frozendict(kwargs))
        try:
            return cls.__instances__[key]
        except KeyError:
            instance = super().__create__(*args, **kwargs)
            cls.__instances__[key] = instance
            return instance


class Comparable(Weakrefable):

    __slots__ = ()
    __cache__ = WeakCache()

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        try:
            return self.__cached_equals__(other)
        except TypeError:
            raise NotImplemented  # noqa: F901

    @abstractmethod
    def __equals__(self, other):
        ...

    def __cached_equals__(self, other):
        if self is other:
            return True

        # type comparison should be cheap
        if type(self) != type(other):
            return False

        # reduce space required for commutative operation
        if hash(self) < hash(other):
            key = (self, other)
        else:
            key = (other, self)

        try:
            result = self.__cache__[key]
        except KeyError:
            result = self.__equals__(other)
            self.__cache__[key] = result

        return result


class Pattern:

    __slots__ = ('_cls', '_args')

    def __init__(self, cls, args):
        self._cls = cls
        self._args = args

    def __repr__(self):
        args = ", ".join(
            f"{name}={value!r}"
            for name, value in zip(self.argnames, self.args)
        )
        return f"{self._cls.__name__}({args})"

    @property
    def args(self):
        return tuple(self._args.values())

    @property
    def argnames(self):
        return tuple(self._args.keys())


@_unify.register(Annotable, Pattern, Mapping)
def _unify_Annotable(u, v, s):
    return _unify(u.args, v.args, s)


@_unify.register(Pattern, Annotable, Mapping)
def _unify_Pattern(u, v, s):
    return _unify(u.args, v.args, s)


@_reify.register(Pattern, Mapping)
def _reify_Pattern(o, s):
    kwargs = yield _reify(o._args, s)
    obj = o._cls(**kwargs)
    yield construction_sentinel
    yield obj


def reduce_identity(expanded_term, reduced_term):
    import ibis.expr.datatypes as dt
    import ibis.expr.operations as ops

    x = var()  # any operation, but see below
    type = var()

    zero = ops.Literal.pattern(0, dtype=type)
    one = ops.Literal.pattern(1, dtype=type)

    return lall(
        isinstanceo(x, ops.Value),
        isinstanceo(x, ExpressionTuple),
        isinstanceo(type, dt.Primitive),
        conde(  # similar to cond from lisp
            [eq(expanded_term, etuple(add, x, zero)), eq(reduced_term, x)],
            [eq(expanded_term, etuple(add, zero, x)), eq(reduced_term, x)],
            [eq(expanded_term, etuple(mul, x, one)), eq(reduced_term, x)],
            [eq(expanded_term, etuple(mul, one, x)), eq(reduced_term, x)],
        ),
    )


def const_fold(expanded_term, reduced_term):
    import ibis.expr.operations as ops

    x = var(prefix="x")
    y = var(prefix="y")

    x_type = var(prefix="x_type")
    y_type = var(prefix="y_type")

    left = var(prefix="left")
    right = var(prefix="right")

    return lall(
        eq(x_type, y_type),
        conde(
            [
                isinstanceo(expanded_term, ops.Literal),
                applyo(attrgetter("value"), (expanded_term,), reduced_term),
            ],
            [
                eq(expanded_term, etuple(add, x, y)),
                term_walko(const_foldo, x, left),
                term_walko(const_foldo, y, right),
                applyo(add, (left, right), reduced_term),
            ],
        ),
    )


const_foldo = partial(reduceo, const_fold)
term_walko = partial(walko, rator_goal=eq, null_type=ExpressionTuple)


def main():
    import ibis
    import ibis.expr.datatypes as dt
    import ibis.expr.operations as ops

    identity_reduceo = partial(reduceo, reduce_identity)
    #  0 + 0 + t.a * 1
    expanded_term = etuple(
        add,
        ops.Literal(0, dtype=dt.int64),
        etuple(
            add,
            ops.Literal(0, dtype=dt.int64),
            etuple(
                mul,
                ops.TableColumn(
                    table=ops.UnboundTable(
                        schema=ibis.schema({"a": "int64"}),
                        name="t",
                    ),
                    name="a",
                ),
                ops.Literal(1, dtype=dt.int64),
            ),
        ),
    )
    expanded_term = etuple(
        add,
        ops.Literal(0, dtype=dt.int64),
        etuple(
            add,
            ops.Literal(0, dtype=dt.int64),
            ops.Literal(1, dtype=dt.int64),
        ),
    )
    #  expanded_term = etuple(
    #      add,
    #      ops.Literal(0, dtype=dt.int64),
    #      ops.Literal(1, dtype=dt.int64),
    #  )
    reduced_term = var()

    res = run(
        1,
        reduced_term,
        term_walko(const_foldo, expanded_term, reduced_term),
    )
    return res


if __name__ == "__main__":
    print(main())
