import io
import typing
from typing import Generic, Optional, TypeVar, Union

from ibis.common.typing import evaluate_annotations, get_type_hints

T = TypeVar("T")
S = TypeVar("S")


class My(Generic[T, S]):
    a: T
    b: S
    c: str


def example(a: int, b: str) -> str:
    ...


def test_evaluate_annotations():
    annotations = {"a": "Union[int, str]", "b": "Optional[str]"}
    hints = evaluate_annotations(annotations, module_name=__name__)
    assert hints == {"a": Union[int, str], "b": Optional[str]}


def test_get_type_hints():
    hints = get_type_hints(My)
    assert hints == {"a": T, "b": S, "c": str}

    hints = get_type_hints(example)
    assert hints == {"a": int, "b": str, "return": str}
