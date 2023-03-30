from abc import ABC, abstractmethod


class _Negatable(ABC):
    @abstractmethod
    def negate(self):  # pragma: no cover
        ...
