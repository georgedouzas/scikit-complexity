"""Find the connected components of the active links between members.

They are found with a union-find forest.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from numba import njit

from ...types import Array, IntArray

LABEL = 0
SIZE = 1
LARGEST = 2
GIANT = 3


@njit(cache=True)
def _root(parents: IntArray, member: int) -> int:
    """Return the root of the tree of a member, halving the path to it on the way."""
    while parents[member] != member:
        parents[member] = parents[parents[member]]
        member = parents[member]
    return member


@njit(cache=True)
def _component_labels(size: int, firsts: IntArray, seconds: IntArray, count: int) -> IntArray:
    """Return the smallest member of the component of every member.

    The components are over the first `count` active links.
    """
    parents = np.arange(size)
    for link in range(count):
        first, second = _root(parents, firsts[link]), _root(parents, seconds[link])
        if first < second:
            parents[second] = first
        elif second < first:
            parents[first] = second
    labels = np.empty(size, dtype=np.int64)
    for member in range(size):
        labels[member] = _root(parents, member)
    return labels


@njit(cache=True)
def _component_values(size: int, firsts: IntArray, seconds: IntArray, count: int, kind: int) -> Array:
    """Return a value of the component of every member.

    The value is the label, the size, the largest size, or the membership of the giant component. The first `count`
    entries of `firsts` and `seconds` are the members of the active links. The label of a component is its smallest
    member, and the giant component is the largest one, the one of the smallest label among ties.
    """
    labels = _component_labels(size, firsts, seconds, count)
    sizes = np.zeros(size, dtype=np.int64)
    for member in range(size):
        sizes[labels[member]] += 1
    largest, giant = 0, 0
    for member in range(size):
        if sizes[member] > largest:
            largest, giant = sizes[member], member
    values = np.empty(size)
    for member in range(size):
        if kind == LABEL:
            values[member] = labels[member]
        elif kind == SIZE:
            values[member] = sizes[labels[member]]
        elif kind == LARGEST:
            values[member] = largest
        else:
            values[member] = 1.0 if labels[member] == giant else 0.0
    return values
