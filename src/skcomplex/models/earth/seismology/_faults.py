"""Seismology models: earthquakes on loaded faults."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, parameter, state
from ....core.model import Drive, Element, LatticeSpace, Law, LocalRule, Neighborhood, Relaxation, Symbols, System
from ....core.runtime import register_model

type Array = npt.NDArray[np.float64]

NEAREST = Neighborhood.von_neumann(center=True)


class Fault(Element):
    """Block of the Olami-Feder-Christensen earthquake model, loaded uniformly and slipping at a threshold force.

    A slipping block passes the fraction `alpha` of its force to each of its four neighbours and drops to zero. With
    `alpha` below 1/4 the model is dissipative and its earthquake sizes still follow a power law.

    Args:
        label:
            The label of the block or the lattice group.

        force:
            The initial force.

        alpha:
            The fraction of force passed to each neighbour.

        threshold:
            The force at which a block slips.

    Examples:
        >>> from skcomplex.models.earth.seismology import Fault
        >>> Fault('block').alpha
        0.2
    """

    force: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    alpha: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    threshold: float = 1.0

    def laws(self, s: Symbols) -> list[Law]:
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
            Relaxation(
                drive=[Drive(s.force, load)],
                relax=[LocalRule(s.force, NEAREST, slip)],
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
    return System(components=[Fault.lattice('fault', space, force=forces, alpha=alpha)])
