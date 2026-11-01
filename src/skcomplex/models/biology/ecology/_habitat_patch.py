"""Patch of space, a compartment of `n` organisms that divide and die at the same rate `alpha`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import CompartmentElement
from ....specs import ParameterSpec, StateSpec, VisualSpec


class HabitatPatchElement(CompartmentElement):
    """Patch of space, a compartment of `n` organisms that divide and die at the same rate `alpha`.

    The divisions and deaths are jumps of `neutral_clustering`, so the number of organisms is a symmetric random walk,
    and an empty patch stays empty without arrivals from its neighbours.

    Args:
        label:
            The label of the patch or the lattice of patches.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of organisms.

        alpha:
            The rate at which an organism divides, equal to the rate at which it dies.

    Examples:
        >>> from skcomplex.models.biology.ecology import HabitatPatchElement
        >>> HabitatPatchElement('patch', n=5.0).alpha
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('image', {'low': 0.0}),
        *,
        n: Annotated[float, StateSpec(domain=NN)] = 10.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual, n=n)
        self.alpha = alpha
