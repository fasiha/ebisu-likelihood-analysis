from typing import Callable, TypeVar
from collections.abc import Iterable
T = TypeVar('T')


def split_by(split_pred: Callable[[T, list[T]], bool],
             lst: Iterable[T]) -> list[list[T]]:
    "Allows each element to decide if it wants to be in previous partition"
    lst = iter(lst)
    try:
        x = next(lst)
    except StopIteration:  # empty iterable (list, zip, etc.)
        return []
    ret: list[list[T]] = []
    ret.append([x])
    for x in lst:
        if split_pred(x, ret[-1]):
            ret.append([x])
        else:
            ret[-1].append(x)
    return ret


def partition_by(f: Callable[[T], bool], lst: Iterable[T]) -> list[list[T]]:
    "See https://clojuredocs.org/clojure.core/partition-by"
    lst = iter(lst)
    try:
        x = next(lst)
    except StopIteration:  # empty iterable (list, zip, etc.)
        return []
    ret: list[list[T]] = []
    ret.append([x])
    y = f(x)
    for x in lst:
        newy = f(x)
        if y == newy:
            ret[-1].append(x)
        else:
            ret.append([x])
        y = newy
    return ret
