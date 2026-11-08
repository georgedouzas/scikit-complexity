"""Define the set symbol spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import re
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ._reference_spec import ReferenceSpec

_MEMBER_REFERENCE = re.compile(r'^(?P<path>.+)\[(?P<index>-?\d+)\]$')


_EDGE_NAMES = ('source', 'target', 'edge')


@dataclass(frozen=True)
class SetSymbolSpec(BaseSpec):
    """Symbol of one element of an element set.

    It holds the same symbol of every element of the set.
    """

    canonical: Expression
    references: tuple[ReferenceSpec, ...]
    label: str
