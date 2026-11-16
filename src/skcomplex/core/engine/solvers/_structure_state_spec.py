"""Define the structure state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...specs._base_spec import BaseSpec


@dataclass(frozen=True)
class StructureStateSpec(BaseSpec):
    """A spec of the structures of elements kept outside the state vector.

    They are held at one frame, with their measures.

    Args:
        structures:
            A mapping from element path to its structure, such as a word or a patch.

        reporters:
            A mapping from dotted reporter name, such as `<path>.length`, to its value.
    """

    structures: Mapping[str, object] = field(default_factory=dict)
    reporters: Mapping[str, float] = field(default_factory=dict)
