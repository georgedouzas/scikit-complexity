"""Stochastic process systems: random iterations of interval maps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import bernoulli
from ....core.components.system import System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class RandomBranchMapElement(Element):
    """Value of a random interval map, which applies `(1 + x) / 2` with probability `p1` and `mu x²` otherwise.

    Args:
        label:
            The label of the element or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, between 0 and 1.

        p1:
            The probability of the branch `(1 + x) / 2`.

        mu:
            The coefficient of the branch `mu x²`, between 0 and 1.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import RandomBranchMapElement
        >>> RandomBranchMapElement('map', p1=0.3).p1
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.5,
        p1: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.5,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.p1 = p1
        self.mu = mu

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the update of the value by the branch drawn in the step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        first = bernoulli('branch', s.p1)
        return [UpdateLaw(s.x, first * (1 + s.x) / 2 + (1 - first) * s.mu * s.x**2)]


@register_system(
    field='mathematics/stochastic_processes',
    tags=('random map', 'iterated function system', 'interval map', 'stationary distribution'),
    references=('Dilão 2023',),
    defaults={'until': 200, 'dt': 1},
)
def random_branch_map(
    n_orbits: int = 200,
    *,
    p1: float = 0.5,
    mu: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return orbits of a random map of `[0, 1]` with the two branches `(1 + x) / 2` and `mu x²`.

    At every step each orbit applies `(1 + x) / 2` with probability `p1` and `mu x²` otherwise (Dilão 2023, Exercise
    5.10). The first branch pulls towards 1 and the second towards 0, so the orbits settle on a stationary
    distribution whose shape changes with `p1` and `mu`. For `mu = 0` the second branch resets the orbit to 0, and
    the stationary mean is `p1 / (2 - p1)`. For `mu = 1` both branches fix `x = 1`, which attracts the orbits once
    `p1 > 1/2`, since there the first branch halves the distance to 1 and the second doubles it.

    Args:
        n_orbits:
            The number of independent orbits.

        p1:
            The probability of the branch `(1 + x) / 2`.

        mu:
            The coefficient of the branch `mu x²`, between 0 and 1.

        random_state:
            The seed or generator of the initial values, uniform on `[0, 1]`.

    Returns:
        system:
            The system, with the group `orbits` of state `x`.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import random_branch_map
        >>> random_branch_map(10, p1=0.3).elements[0].prepare().size_
        10
    """
    start = np.random.default_rng(random_state).uniform(0.0, 1.0, n_orbits)
    return System(
        elements=[
            GroupElement(
                'orbits',
                element=RandomBranchMapElement,
                size=n_orbits,
                values={'x': start, 'p1': p1, 'mu': mu},
            ),
        ],
    )
