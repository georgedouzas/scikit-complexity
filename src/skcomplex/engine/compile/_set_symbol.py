"""Symbol of one element of an element set, with the same symbol of every element of the set."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ..problems import Reference
from ._observed import Observed
from ._tree import Node

_MEMBER_REFERENCE = re.compile(r'^(?P<path>.+)\[(?P<index>-?\d+)\]$')


_EDGE_NAMES = ('source', 'target', 'edge')


type Observe = Callable[[Node, object], Observed | None]


@dataclass(frozen=True)
class SetSymbol:
    """Symbol of one element of an element set, with the same symbol of every element of the set."""

    canonical: Expression
    references: tuple[Reference, ...]
    label: str
