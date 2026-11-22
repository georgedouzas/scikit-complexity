"""Define the set symbol spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import re
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec
from ._reference_spec import ReferenceSpec

MEMBER_REFERENCE = re.compile(r'^(?P<path>.+)\[(?P<index>-?\d+)\]$')
EDGE_NAMES = ('source', 'target', 'edge')


@dataclass(frozen=True)
class SetSymbolSpec(BaseSpec):
    """A spec of the one symbol that stands for every element of an element set.

    An element set is a list of elements that a law treats alike, such as every species of a reaction, and the engine
    writes the law once in one symbol and evaluates it for every element of the set. The spec holds that symbol and the
    value it stands for in each element. It writes the law of every species of a reaction, every tank of a plant, and
    every account of a bank once.

    Read more in the user guide, under Extend the library.

    Args:
        canonical:
            The symbol the law is written in.

        references:
            The value the symbol stands for in every element of the set.

        label:
            The label of the set.

    See Also:
        KernelSpec: What the laws of a component read.

    Examples:
        >>> from skcomplex.core.engine._utils import symbol
        >>> from skcomplex.core.engine.problems._set_symbol_spec import SetSymbolSpec
        >>> SetSymbolSpec(symbol('x'), (), 'species').label
        'species'
    """

    canonical: Expression
    references: tuple[ReferenceSpec, ...]
    label: str
