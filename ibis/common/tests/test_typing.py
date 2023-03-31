import io
import typing
from typing import Generic, Optional, TypeVar, Union

from ibis.common.typing import get_type_hints

T = TypeVar("T")
S = TypeVar("S")


class My(Generic[T, S]):
    a: T
    b: S
    c: str


def example(a: int, b: str) -> str:
    ...


# TODO(kszucs): rename it to evaluate()
# def test_evaluate_typehint():
#     hint = evaluate_typehint("Union[int, str]", module_name=__name__)
#     assert hint == Union[int, str]

#     hint = evaluate_typehint(Optional[str], module_name=__name__)
#     assert hint == Optional[str]


# def test_evaluate_annotations():
#     pass


def test_get_type_hints():
    hints = get_type_hints(My)
    assert hints == {"a": T, "b": S, "c": str}

    hints = get_type_hints(example)
    assert hints == {"a": int, "b": str, "return": str}


# def test_issubtype():
#     assert issubtype(str, str)
#     assert not issubtype(int, str)

#     # Any
#     assert issubtype(typing.List, typing.Any)
#     assert issubtype(typing.Any, typing.Any)

#     # Self
#     assert issubtype(list, list)
#     assert issubtype(typing.List, typing.List)
#     assert not issubtype(list, dict)
#     assert not issubtype(typing.List, typing.Dict)

#     # None
#     assert issubtype(None, type(None))
#     assert issubtype(type(None), None)
#     assert issubtype(None, None)

#     # alias
#     assert issubtype(list, typing.List)
#     assert issubtype(typing.List, list)
#     assert issubtype(bytes, typing.ByteString)

#     # Subclass
#     assert issubtype(list, typing.Sequence)

#     # # FileLike
#     # with open("test", "wb") as file_ref:
#     #     assert issubtype(type(file_ref), typing.BinaryIO)
#     # with open("test", "rb") as file_ref:
#     #     assert issubtype(type(file_ref), typing.BinaryIO)
#     # with open("test", "w") as file_ref:
#     #     assert issubtype(type(file_ref), typing.TextIO)
#     # with open("test", "r") as file_ref:
#     #     assert issubtype(type(file_ref), typing.TextIO)

#     # assert issubtype(type(io.BytesIO(b"0")), typing.BinaryIO)
#     # assert issubtype(type(io.StringIO("0")), typing.TextIO)

#     # subscribed generic
#     assert issubtype(typing.List[int], list)
#     assert issubtype(typing.List[typing.List], list)
#     assert not issubtype(list, typing.List[int])

#     # # Union
#     # assert issubtype(list, typing.Union[typing.List, typing.Tuple])
#     # assert issubtype(typing.Union[list, tuple], typing.Union[list, tuple, None])
#     # assert issubtype(typing.Union[list, tuple], typing.Sequence)

#     # assert not issubtype(list, typing.Union[typing.Tuple, typing.Set])
#     # assert not issubtype(typing.Tuple[typing.Union[int, None]], typing.Tuple[None])

#     # # Nested containers
#     # assert issubtype(typing.List[int], typing.List[int])
#     # assert issubtype(typing.List[typing.List], typing.List[typing.Sequence])

#     # assert issubtype(typing.Dict[typing.List, int], typing.Dict[typing.Sequence, int])
#     # assert issubtype(
#     #     typing.Callable[[typing.List, int], int],
#     #     typing.Callable[[typing.Sequence, int], int],
#     # )
#     # assert not issubtype(
#     #     typing.Callable[[typing.Sequence, int], int],
#     #     typing.Callable[[typing.List, int], int],
#     # )
