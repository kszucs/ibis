"""Converted the previous validator system into a pattern matching system.

The previously used validator system had the following problems:
- Used curried functions which was error prone to missing arguments and was hard to debug.
- Used exceptions for control flow which raising exceptions from the innermost function call giving cryptic error messages.
- While it was similarly composable it was hard to see whether the validator was fully constructed or not.
- We wasn't able traverse the nested validators due to the curried functions.

In addition to those the new approach has the following advantages:
- The pattern matching system is fully composable.
- Not we support syntax sugar for combining patterns using & and |.
- We can capture values at mutiple levels using either `>>` or `@` syntax.
- Support structural pattern matching for sequences, mappings and objects.
"""
from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from enum import Enum
from inspect import Parameter
from itertools import chain, zip_longest
from typing import Any as AnyType
from typing import (
    Callable,
    Literal,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import Annotated, get_args, get_origin

from ibis.common.collections import RewindableIterator, frozendict
from ibis.common.dispatch import lazy_singledispatch
from ibis.common.typing import Coercible, CoercionError, bind_typevars, get_parameters
from ibis.util import is_function, is_iterable, promote_tuple

try:
    from types import UnionType
except ImportError:
    UnionType = object()


class ValidationError(Exception):
    ...


class Validator(ABC):
    __slots__ = ()

    @abstractmethod
    def validate(self, value, context):
        ...

    def __call__(self, value, this=None):
        this = this or {}
        return self.validate(value, this)


class MatchError(Exception):
    ...


class NoMatch:
    """Sentinel value for when a pattern doesn't match."""


class Pattern(Validator, Hashable):
    @classmethod
    def from_typehint(cls, annot: type) -> Pattern:
        """Construct a pattern from a python type annotation.

        Parameters
        ----------
        annot
            The typehint annotation to construct the pattern from. If a string is
            available then it must be evaluated before passing it to this function
            using the ``evaluate_typehint`` function.
        module
            The module to evaluate the type annotation in. This is only used
            when the first argument is a string (or ForwardRef).

        Returns
        -------
        pattern
            A pattern that matches the given type annotation.
        """
        # TODO(kszucs): cache the result of this function
        origin, args = get_origin(annot), get_args(annot)

        if origin is None:
            if annot is Ellipsis:  # TODO(kszucs): test this
                return Any()
            elif annot is None:
                return Is(None)
            elif annot is AnyType:
                return Any()
            elif isinstance(annot, TypeVar):
                # TODO(kszucs): only use coerced_to if annot.__covariant__ is True
                if annot.__bound__ is None:
                    return Any()
                return CoercedTo(annot.__bound__)
            elif isinstance(annot, Enum):
                return EqualTo(annot)
            elif issubclass(annot, Coercible):
                # CoercedTo(annot) & InstanceOf(annot)
                return CoercedTo(annot)
            else:
                return InstanceOf(annot)
        elif origin is Literal:
            return IsIn(args)
        elif origin is UnionType or origin is Union:
            inners = map(cls.from_typehint, args)
            return AnyOf(*inners)
        elif origin is Annotated:
            annot, *extras = args
            return AllOf(InstanceOf(annot), *extras)
        elif origin is Callable:
            # issubclass(origin, Callable):
            if args:
                arg_inners = tuple(map(cls.from_typehint, args[0]))
                return_inner = cls.from_typehint(args[1])
                return CallableWith(arg_inners, return_inner)
            else:
                return InstanceOf(Callable)
        elif issubclass(origin, Tuple):
            first, *rest = args
            if rest == [Ellipsis]:
                inners = cls.from_typehint(first)
            else:
                inners = tuple(map(cls.from_typehint, args))
            return TupleOf(inners)
        elif issubclass(origin, Sequence):
            (value_inner,) = map(cls.from_typehint, args)
            return SequenceOf(value_inner, type=origin)
        elif issubclass(origin, Mapping):
            key_inner, value_inner = map(cls.from_typehint, args)
            return MappingOf(key_inner, value_inner, type=origin)
        elif issubclass(origin, Coercible) and args:
            # would be nice to pass the typevars as kwargs to __coerce__
            # so the coercible can do various things with them
            params = get_parameters(origin, args)
            fields = bind_typevars(origin, args)
            field_inners = {k: cls.from_typehint(v) for k, v in fields.items()}
            # CoercedTo(origin) & GenericInstanceOf(origin, field_inners)
            return GenericCoercedTo(
                origin, frozendict(params), frozendict(field_inners)
            )
        elif isinstance(origin, type) and args:
            fields = bind_typevars(origin, args)
            field_inners = {k: cls.from_typehint(v) for k, v in fields.items()}
            return GenericInstanceOf(origin, frozendict(field_inners))
        else:
            raise NotImplementedError(
                f"Cannot create validator from annotation {annot} {origin}"
            )

    @abstractmethod
    def match(self, value, context):
        ...

    @abstractmethod
    def __eq__(self, other):
        ...

    # TODO(kszucs): consider to add match_strict() with a default implementation
    # that calls match()

    def __invert__(self):
        return Not(self)

    def __or__(self, other):
        return AnyOf(self, other)

    def __and__(self, other):
        return AllOf(self, other)

    def __rshift__(self, name):
        return Capture(self, name)

    def __rmatmul__(self, name):
        return Capture(self, name)

    def validate(self, value, context):
        result = self.match(value, context=context)
        if result is NoMatch:
            raise ValidationError(f"{value} doesn't match {self}")
        return result


# extend Singleton base class to make this a singleton
class Matcher(Pattern):
    """A lightweight alternative to `ibis.common.grounds.Concrete`.

    This class is used to create immutable dataclasses with slots and a precomputed
    hash value for quicker dictionary lookups.
    """

    __slots__ = ("__precomputed_hash__",)

    def __init__(self, *args):
        for name, value in zip_longest(self.__slots__, args):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "__precomputed_hash__", hash(args))

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        for name in self.__slots__:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self):
        return self.__precomputed_hash__

    def __setattr__(self, name, value):
        raise AttributeError("Can't set attributes on immutable ENode instance")

    def __repr__(self):
        fields = {k: getattr(self, k) for k in self.__slots__}
        fieldstring = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.__class__.__name__}({fieldstring})"

    def __rich_repr__(self):
        for name in self.__slots__:
            yield name, getattr(self, name)


class Is(Matcher):
    """Pattern that matches a value against a reference value.

    Parameters
    ----------
    value
        The reference value to match against.
    """

    __slots__ = ("value",)

    def match(self, value, context):
        if value is self.value:
            return value
        else:
            return NoMatch


class Any(Matcher):
    """Pattern that accepts any value, basically a no-op."""

    __slots__ = ()

    def match(self, value, context):
        return value


class Capture(Matcher):
    __slots__ = ("pattern", "name")

    def match(self, value, context):
        value = self.pattern.match(value, context=context)
        context[self.name] = value
        return value


class Reference(Matcher):
    """Retrieve a value from the context.

    Parameters
    ----------
    key
        The key to retrieve from the state.
    """

    __slots__ = ("key",)

    def match(self, context):
        return context[self.key]


class Check(Matcher):
    """Pattern that checks a value against a predicate.

    Parameters
    ----------
    predicate
        The predicate to use.
    """

    __slots__ = ("predicate",)

    def match(self, value, context):
        if self.predicate(value):
            return value
        else:
            return NoMatch


class Apply(Matcher):
    """Pattern that applies a function to the value.

    Parameters
    ----------
    func
        The function to apply.
    """

    __slots__ = ("func",)

    def match(self, value, context):
        return self.func(value)


class EqualTo(Matcher):
    """Pattern that checks a value equals to the given value.

    Parameters
    ----------
    value
        The value to check against.
    """

    __slots__ = ("value",)

    def match(self, value, context):
        if value == self.value:
            return value
        else:
            return NoMatch


class Option(Matcher):
    """Pattern that matches `None` or a value that passes the inner validator.

    Parameters
    ----------
    pattern
        The inner pattern to use.
    default
        The default value to use if `arg` is `None`.
    """

    __slots__ = ("pattern", "default")

    def __init__(self, pattern, default=None):
        super().__init__(pattern, default)

    def match(self, value, context):
        default = self.default
        if value is None:
            if self.default is None:
                return None
            elif is_function(default):
                value = default()
            else:
                value = default
        return self.pattern.match(value, context=context)


class TypeOf(Matcher):
    """Pattern that matches a value that is of a given type."""

    __slots__ = ("type",)

    def match(self, value, context):
        if type(value) is self.type:
            return value
        else:
            return NoMatch


class SubclassOf(Matcher):
    """Pattern that matches a value that is a subclass of a given type.

    Parameters
    ----------
    type
        The type to check against.
    """

    __slots__ = ("type",)

    def match(self, value, context):
        if issubclass(value, self.type):
            return value
        else:
            return NoMatch


# InstanceOf(dt.Array[dt.String])
# TypedAs(dt.Array[dt.String])
# TypedAs(ops.Value[dt.String, ...])


class InstanceOf(Matcher):
    """Pattern that matches a value that is an instance of a given type.

    Parameters
    ----------
    types
        The type to check against.
    """

    __slots__ = ("type",)

    # first try to support Generic types here
    def match(self, value, context):
        if isinstance(value, self.type):
            return value
        else:
            return NoMatch


class GenericInstanceOf(Matcher):
    __slots__ = ("origin", "field_patterns")

    def match(self, value, context):
        if not isinstance(value, self.origin):
            return NoMatch

        for field, pattern in self.field_patterns.items():
            attr = getattr(value, field)
            if pattern.match(attr, context) is NoMatch:
                return NoMatch

        return value


class LazyInstanceOf(Matcher):
    """A version of `InstanceOf` that accepts qualnames instead of imported classes.

    Useful for delaying imports.

    Parameters
    ----------
    types
        The types to check against.
    """

    __slots__ = ("types", "check")

    def __init__(self, types):
        types = promote_tuple(types)
        check = lazy_singledispatch(lambda x: False)
        check.register(types, lambda x: True)
        super().__init__(promote_tuple(types), check)

    def match(self, value, *, context):
        if self.check(value):
            return value
        else:
            return NoMatch


# just a shorthand
def coerce(value, type):
    if type is Ellipsis:
        return value
    p = Pattern.from_typehint(type)
    result = p.match(value, {})
    if result is NoMatch:
        raise CoercionError(f"Unable to coerce {value} to {type}")
    return result


class CoercedTo(Matcher):
    """Force a value to have a particular Python type.

    If a Coercible subclass is passed, the `__coerce__` method will be used to
    coerce the value. Otherwise, the type will be called with the value as the
    only argument.

    Parameters
    ----------
    type
        The type to coerce to.
    """

    __slots__ = ("func", "origin", "checker")

    def __init__(self, type):
        origin = get_origin(type)
        if origin is None:
            origin = type

        if issubclass(origin, Coercible):
            func = origin.__coerce__
        else:
            func = type

        checker = InstanceOf(origin)

        super().__init__(func, origin, checker)

    def match(self, value, context):
        try:
            value = self.func(value)
        except CoercionError:
            return NoMatch

        if self.checker.match(value, context) is NoMatch:
            return NoMatch

        return value

    def __repr__(self):
        return f"CoercedTo({self.origin.__name__!r})"


class GenericCoercedTo(Matcher):
    __slots__ = ("origin", "params", "field_patterns", "checker")

    def __init__(self, origin, params, field_patterns):
        checker = GenericInstanceOf(origin, field_patterns)
        super().__init__(origin, params, field_patterns, checker)

    def match(self, value, context):
        # plain coercion here
        try:
            value = self.origin.__coerce__(value, **self.params)
        except CoercionError:
            return NoMatch

        # for field, pattern in self.field_patterns.items():
        #     attr = getattr(value, field)
        #     if pattern.match(attr, context) is NoMatch:
        #         return NoMatch

        if self.checker.match(value, context) is NoMatch:
            return NoMatch

        return value


class Not(Matcher):
    __slots__ = ("pattern",)

    def match(self, value, context):
        if self.pattern.match(value, context=context) is NoMatch:
            return value
        else:
            return NoMatch


class AnyOf(Matcher):
    __slots__ = ("patterns",)

    def __init__(self, *patterns):
        super().__init__(patterns)

    def match(self, value, context):
        for pattern in self.patterns:
            result = pattern.match(value, context=context)
            if result is not NoMatch:
                return result
        return NoMatch


class AllOf(Matcher):
    __slots__ = ("patterns",)

    def __init__(self, *patterns):
        super().__init__(patterns)

    def match(self, value, context):
        for pattern in self.patterns:
            value = pattern.match(value, context=context)
            if value is NoMatch:
                return NoMatch
        return value


class Length(Matcher):
    __slots__ = ("at_least", "at_most")

    def __init__(self, exactly=None, at_least=None, at_most=None):
        if exactly is not None:
            if at_least is not None or at_most is not None:
                raise ValueError("Can't specify both exactly and at_least/at_most")
            at_least = exactly
            at_most = exactly
        super().__init__(at_least, at_most)

    def match(self, value, *, context):
        length = len(value)
        if self.at_least is not None and length < self.at_least:
            return NoMatch
        if self.at_most is not None and length > self.at_most:
            return NoMatch
        return value


class Contains(Matcher):
    __slots__ = ("needle",)

    def match(self, value, context):
        if self.needle in value:
            return value
        else:
            return NoMatch


class IsIn(Matcher):
    __slots__ = ("haystack",)

    def __init__(self, haystack):
        super().__init__(frozenset(haystack))

    def match(self, value, context):
        if value in self.haystack:
            return value
        else:
            return NoMatch


class SequenceOf(Matcher):
    __slots__ = ("item_pattern", "type_pattern", "length_pattern")

    def __init__(self, pat, type=tuple, at_least=None, at_most=None):
        item_pattern = pattern(pat)
        type_pattern = CoercedTo(type) if issubclass(type, Coercible) else Apply(type)
        length_pattern = Length(at_least=at_least, at_most=at_most)
        super().__init__(item_pattern, type_pattern, length_pattern)

    def match(self, values, context):
        if not is_iterable(values):
            return NoMatch

        result = []
        for value in values:
            value = self.item_pattern.match(value, context=context)
            if value is NoMatch:
                return NoMatch
            result.append(value)

        result = self.type_pattern.match(result, context=context)
        if result is NoMatch:
            return NoMatch

        return self.length_pattern.match(result, context=context)


class TupleOf(Matcher):
    __slots__ = ("patterns",)

    def __new__(cls, patterns):
        if isinstance(patterns, tuple):
            return super().__new__(cls)
        else:
            return SequenceOf(patterns, tuple)

    def match(self, values, context):
        if not is_iterable(values):
            return NoMatch

        if len(values) != len(self.patterns):
            return NoMatch

        result = []
        for pattern, value in zip(self.patterns, values):
            value = pattern.match(value, context=context)
            if value is NoMatch:
                return NoMatch
            result.append(value)

        return tuple(result)


class MappingOf(Matcher):
    __slots__ = ("key_pattern", "value_pattern", "type")

    def __init__(self, key_pattern, value_pattern, type=dict):
        super().__init__(pattern(key_pattern), pattern(value_pattern), type)

    def match(self, value, *, context):
        if not isinstance(value, Mapping):
            return NoMatch

        result = {}
        for k, v in value.items():
            if (k := self.key_pattern.match(k, context=context)) is NoMatch:
                return NoMatch
            if (v := self.value_pattern.match(v, context=context)) is NoMatch:
                return NoMatch
            result[k] = v

        return self.type(result)


class Object(Matcher):
    __slots__ = ("type", "patterns")

    def __init__(self, type, *args, **kwargs):
        match_args = getattr(type, "__match_args__", tuple())
        kwargs.update(dict(zip(args, match_args)))
        super().__init__(type, frozendict(kwargs))

    def match(self, value, *, context):
        if not isinstance(value, self.type):
            return NoMatch

        for attr, pattern in self.patterns.items():
            if not hasattr(value, attr):
                return NoMatch

            if match(pattern, getattr(value, attr), context=context) is NoMatch:
                return NoMatch

        return value


class CallableWith(Matcher):
    __slots__ = ("arg_patterns", "return_pattern")

    def __init__(self, arg_patterns, return_pattern=None):
        super().__init__(tuple(arg_patterns), return_pattern or Any())

    def match(self, value, context):
        from ibis.common.annotations import annotated

        if not callable(value):
            return NoMatch

        fn = annotated(self.arg_patterns, self.return_pattern, value)

        has_varargs = False
        positional, keyword_only = [], []
        for p in fn.__signature__.parameters.values():
            if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                positional.append(p)
            elif p.kind is Parameter.KEYWORD_ONLY:
                keyword_only.append(p)
            elif p.kind is Parameter.VAR_POSITIONAL:
                has_varargs = True

        if keyword_only:
            raise MatchError(
                "Callable has mandatory keyword-only arguments which cannot be specified"
            )
        elif len(positional) > len(self.arg_patterns):
            # Callable has more positional arguments than expected")
            return NoMatch
        elif len(positional) < len(self.arg_patterns) and not has_varargs:
            # Callable has less positional arguments than expected")
            return NoMatch
        else:
            return fn


class PatternSequence(Matcher):
    __slots__ = ("pattern_window",)

    def __init__(self, patterns):
        current_patterns = [
            SequenceOf(Any()) if p is Ellipsis else pattern(p) for p in patterns
        ]
        following_patterns = chain(current_patterns[1:], [Not(Any())])
        pattern_window = tuple(zip(current_patterns, following_patterns))
        super().__init__(pattern_window)

    @property
    def first_pattern(self):
        return self.pattern_window[0][0]

    def match(self, value, context):
        it = RewindableIterator(value)
        result = []

        if not self.pattern_window:
            try:
                next(it)
            except StopIteration:
                return result
            else:
                return NoMatch

        for current, following in self.pattern_window:
            original = current

            if isinstance(current, Capture):
                current = current.pattern
            if isinstance(following, Capture):
                following = following.pattern

            if isinstance(current, (SequenceOf, PatternSequence)):
                if isinstance(following, SequenceOf):
                    following = following.item_pattern
                elif isinstance(following, PatternSequence):
                    following = following.first_pattern

                matches = []
                while True:
                    it.checkpoint()
                    try:
                        item = next(it)
                    except StopIteration:
                        break

                    if match(following, item, context) is NoMatch:
                        matches.append(item)
                    else:
                        it.rewind()
                        break

                res = original.match(matches, context=context)
                if res is NoMatch:
                    return NoMatch
                else:
                    result.extend(res)
            else:
                try:
                    item = next(it)
                except StopIteration:
                    return NoMatch

                res = original.match(item, context=context)
                if res is NoMatch:
                    return NoMatch
                else:
                    result.append(res)

        return result


class PatternMapping(Matcher):
    __slots__ = ("keys_pattern", "values_pattern")

    def __init__(self, patterns):
        keys_pattern = PatternSequence(list(map(pattern, patterns.keys())))
        values_pattern = PatternSequence(list(map(pattern, patterns.values())))
        super().__init__(keys_pattern, values_pattern)

    def match(self, value, context):
        if not isinstance(value, Mapping):
            return NoMatch

        keys = value.keys()
        if (keys := self.keys_pattern.match(keys, context=context)) is NoMatch:
            return NoMatch

        values = value.values()
        if (values := self.values_pattern.match(values, context=context)) is NoMatch:
            return NoMatch

        return dict(zip(keys, values))


IsTruish = Check(lambda x: bool(x))
IsNumber = InstanceOf(numbers.Number) & ~InstanceOf(bool)
IsString = InstanceOf(str)


def NoneOf(*args) -> Pattern:
    """Match none of the passed patterns."""
    return Not(AnyOf(*args))


def ListOf(pattern):
    return SequenceOf(pattern, type=list)


def DictOf(key_pattern, value_pattern):
    return MappingOf(key_pattern, value_pattern, type=dict)


def FrozenDictOf(key_pattern, value_pattern):
    return MappingOf(key_pattern, value_pattern, type=frozendict)


def pattern(obj):
    """Create a pattern from an object."""
    if obj is Ellipsis:
        return Any()
    elif isinstance(obj, Pattern):
        return obj
    elif isinstance(obj, Mapping):
        return PatternMapping(obj)
    elif isinstance(obj, type):
        return InstanceOf(obj)
    elif get_origin(obj):
        return Pattern.from_typehint(obj)
    elif is_iterable(obj):
        return PatternSequence(obj)
    else:
        return EqualTo(obj)


def match(pat, value, context=None):
    if context is None:
        context = {}

    pat = pattern(pat)
    if pat.match(value, context=context) is NoMatch:
        return NoMatch

    return context


# TODO(kszucs): add a ChildOf matcher to match against the children of a node
