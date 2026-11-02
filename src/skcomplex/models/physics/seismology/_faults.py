"""Seismology models: earthquakes on loaded faults."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, LatticeElement
from ....core.interactions import CascadeInteraction
from ....core.spaces import LatticeSpace, VonNeumannNeighborhood
from ....core.system import System
from ....specs import StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model

THRESHOLD = 1.0


def _load(force: Array, *, random: np.random.Generator) -> Array:
    """Return the forces after the uniform loading that brings the most loaded block to the threshold."""
    following = force + (THRESHOLD - force.max())
    following[np.unravel_index(np.argmax(force), force.shape)] = THRESHOLD
    return following


class FaultElement(Element):
    """Block of the Olami-Feder-Christensen earthquake model, the force on it.

    The cascade of `earthquakes` loads every block uniformly until one reaches the threshold force, and a slipping block
    passes a fraction of its force to each of its four neighbours and drops to zero.

    Args:
        label:
            The label of the block or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        force:
            The initial force.

    Examples:
        >>> from skcomplex.models.physics.seismology import FaultElement
        >>> FaultElement('block').force
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        force: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.force = force


@register_model(
    field='physics/seismology',
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

    The fault is a cascade that resets a slipping block to zero and passes the fraction `alpha` of its force to each of
    its four neighbours. With `alpha` below 1/4 the model is dissipative and its earthquake sizes still follow a power
    law. Its earthquakes are recorded as `earthquake` events with their sizes, the number of slips, their durations,
    and their areas.

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
        >>> from skcomplex.models.physics.seismology import earthquakes
        >>> earthquakes((10, 10), alpha=0.15).interactions[0].share
        0.15
    """
    forces = np.random.default_rng(random_state).random(tuple(shape))
    space = LatticeSpace(tuple(shape), boundary='fixed')
    return System(
        elements=[LatticeElement('fault', element=FaultElement, space=space, values={'force': forces})],
        interactions=[
            CascadeInteraction(
                'earthquake',
                target='fault',
                state='force',
                threshold=THRESHOLD,
                share=alpha,
                mode='reset',
                drive=_load,
                neighborhood=VonNeumannNeighborhood(center=True),
            ),
        ],
    )
