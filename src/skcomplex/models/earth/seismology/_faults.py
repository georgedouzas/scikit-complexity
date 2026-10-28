"""Seismology models: earthquakes on loaded faults."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Any

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, LatticeElement
from ....core.laws import BaseLaw, DriveLaw, LocalRuleLaw, RelaxationLaw
from ....core.spaces import LatticeSpace, VonNeumannNeighborhood
from ....core.system import System
from ....engine.runtime import register_model
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ....types import Array

NEAREST = VonNeumannNeighborhood(center=True)


class FaultElement(Element):
    """Block of the Olami-Feder-Christensen earthquake model, loaded uniformly and slipping at a threshold force.

    A slipping block passes the fraction `alpha` of its force to each of its four neighbours and drops to zero. With
    `alpha` below 1/4 the model is dissipative and its earthquake sizes still follow a power law.

    Args:
        label:
            The label of the block or the lattice group.

        threshold:
            The force at which a block slips.

        **values:
            The values of the declared variables, by name:

            force:
                The initial force.

            alpha:
                The fraction of force passed to each neighbour.

    Examples:
        >>> from skcomplex.models.earth.seismology import FaultElement
        >>> FaultElement('block').alpha
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        force: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        threshold: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.force = force
        self.alpha = alpha
        self.threshold = threshold

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the uniform loading and the slipping relaxation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        threshold = self.threshold

        def load(force: Array, *, random: np.random.Generator) -> Array:
            following = force + (threshold - force.max())
            following[np.unravel_index(np.argmax(force), force.shape)] = threshold
            return following

        def slip(force: Array, *around: Array, alpha: Any) -> Array:  # noqa: ANN401
            passed = sum(np.where(item >= threshold, item, 0.0) for item in around)
            return np.where(force >= threshold, 0.0, force) + alpha * passed

        return [
            RelaxationLaw(
                drive=[DriveLaw(s.force, load)],
                relax=[LocalRuleLaw(s.force, NEAREST, slip)],
                stable=lambda force: bool((force < threshold).all()),
                activity=lambda force: force >= threshold,
                label='earthquake',
            ),
        ]


@register_model(
    field='earth/seismology',
    tags=('lattice', 'earthquakes', 'power law'),
    references=('Olami, Feder, and Christensen 1992', 'Jensen 2023'),
    defaults={'until': 5000, 'dt': 100},
)
def earthquakes(
    shape: Sequence[int] = (50, 50),
    *,
    alpha: float = 0.2,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Olami-Feder-Christensen earthquake model with open edges, started from random forces.

    Its earthquakes are recorded as `earthquake` events with their sizes, the number of slips, their durations, and
    their areas.

    Args:
        shape:
            The number of rows and columns.

        alpha:
            The fraction of force passed to each neighbour.

        random_state:
            The seed or generator of the initial forces.

    Returns:
        system:
            The model, with the forces `fault.force`.

    Examples:
        >>> from skcomplex.models.earth.seismology import earthquakes
        >>> earthquakes((10, 10), alpha=0.15).elements[0].values['alpha']
        0.15
    """
    forces = np.random.default_rng(random_state).random(tuple(shape))
    space = LatticeSpace(tuple(shape), boundary='fixed')
    return System(
        elements=[
            LatticeElement('fault', element=FaultElement, space=space, values={'force': forces, 'alpha': alpha}),
        ],
    )
