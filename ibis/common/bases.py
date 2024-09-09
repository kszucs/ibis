from __future__ import annotations

import collections.abc
from abc import abstractmethod
from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self


# @collections.abc.Hashable.register
# class Hashable(Abstract):
#     @abstractmethod
#     def __hash__(self) -> int: ...


# class Comparable(Abstract):
#     """Enable quick equality comparisons.

#     The subclasses must implement the `__equals__` method that returns a boolean
#     value indicating whether the two instances are equal. This method is called
#     only if the two instances are of the same type and the result is cached for
#     future comparisons.

#     Since the class holds a global cache of comparison results, it is important
#     to make sure that the instances are not kept alive longer than necessary.
#     """

#     __cache__ = {}

#     @abstractmethod
#     def __equals__(self, other) -> bool: ...

#     def __eq__(self, other) -> bool:
#         if self is other:
#             return True

#         # type comparison should be cheap
#         if type(self) is not type(other):
#             return False

#         id1 = id(self)
#         id2 = id(other)
#         try:
#             return self.__cache__[id1][id2]
#         except KeyError:
#             result = self.__equals__(other)
#             self.__cache__.setdefault(id1, {})[id2] = result
#             self.__cache__.setdefault(id2, {})[id1] = result
#             return result

#     def __del__(self):
#         id1 = id(self)
#         for id2 in self.__cache__.pop(id1, ()):
#             eqs2 = self.__cache__[id2]
#             del eqs2[id1]
#             if not eqs2:
#                 del self.__cache__[id2]
