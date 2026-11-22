"""Define the structure state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...base import BaseSpec


@dataclass(frozen=True)
class StructureStateSpec(BaseSpec):
    """A spec of the structures of elements kept outside the state vector at a frame.

    It holds the structures by the path of their element, such as the word of a rewriting system, and the reporters that
    measure them, such as its length. It keeps the words of a grammar, the shapes of a growing plant, and the strings of
    a language.

    Read more in the user guide, under Solvers.

    Args:
        structures:
            A mapping from element path to its structure, such as a word or a patch.

        reporters:
            A mapping from dotted reporter name, such as `<path>.length`, to its value.

    See Also:
        BaseSolver: Collects it after a step.

    Examples:
        >>> from skcomplex.core.engine.solvers import StructureStateSpec
        >>> StructureStateSpec().structures
        {}
    """

    structures: Mapping[str, object] = field(default_factory=dict)
    reporters: Mapping[str, float] = field(default_factory=dict)
