"""Define the structure state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class StructureStateSpec(BaseSpec):
    """Structures of the elements a stepper keeps outside the state vector at one frame, with their measures.

    Args:
        structures:
            A mapping from element path to its structure, such as a word or a patch.

        reporters:
            A mapping from dotted reporter name, such as `<path>.length`, to its value.

    Examples:
        >>> from skcomplex.specs import StructureStateSpec
        >>> StructureStateSpec(reporters={'cantor.length': 1.0}).reporters['cantor.length']
        1.0
    """

    structures: Mapping[str, Any] = field(default_factory=dict)
    reporters: Mapping[str, float] = field(default_factory=dict)
