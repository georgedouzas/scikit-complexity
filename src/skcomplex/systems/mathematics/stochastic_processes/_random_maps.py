"""Stochastic process systems of random iterations of interval maps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import bernoulli
from ....core.components.spaces import BaseSpace
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class RandomBranchMapElement(Element):
    """Values `x` of a random interval map, one per orbit.

    Args:
        label:
            The label of the orbits.

        x:
            The initial value, between 0 and 1, one per orbit or shared.

        size:
            The number of orbits, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import RandomBranchMapElement
        >>> RandomBranchMapElement('map', x=0.3).x
        0.3
    """

    def __init__(self, label: str, *, x: Annotated[float, RealSet.closed(0, 1)] = 0.5, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x


class RandomBranchMapInteraction(Interaction):
    """The random map that applies `(1 + x) / 2` with probability `p1` and `mu x²` otherwise.

    Args:
        label:
            The label of the interaction.

        target:
            The orbits whose values the map moves.

        p1:
            The probability of the branch `(1 + x) / 2`.

        mu:
            The coefficient of the branch `mu x²`, between 0 and 1.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import RandomBranchMapInteraction
        >>> RandomBranchMapInteraction('map', target='orbits', p1=0.3).p1
        0.3
    """

    law_classes = (UpdateLaw,)

    draws = True

    def __init__(self, label: str, *, target: Role = 'orbits', p1: float = 0.5, mu: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.p1 = p1
        self.mu = mu

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the update of the value by the branch drawn in the step as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            first = bernoulli('branch', context.p1)
            self.laws_: list[BaseComponent] = [UpdateLaw(x, first * (1 + x) / 2 + (1 - first) * context.mu * x**2)]


class RandomBranchMapSystem(BaseSystem):
    """Orbits of a random map of `[0, 1]`.

    The map has the two branches `(1 + x) / 2` and `mu x²`.

    At every step each orbit applies `(1 + x) / 2` with probability `p1` and `mu x²` otherwise (Dilão 2023, Exercise
    5.10). The first branch pulls towards 1 and the second towards 0, so the orbits settle on a stationary
    distribution whose shape changes with `p1` and `mu`. For `mu = 0` the second branch resets the orbit to 0, and
    the stationary mean is `p1 / (2 - p1)`. For `mu = 1` both branches fix `x = 1`, which attracts the orbits once
    `p1 > 1/2`, since there the first branch halves the distance to 1 and the second doubles it.

    The system has the group `orbits` of state `x`.

    Args:
        n_orbits:
            The number of independent orbits.

        p1:
            The probability of the branch `(1 + x) / 2`.

        mu:
            The coefficient of the branch `mu x²`, between 0 and 1.

        random_state:
            The seed or generator of the initial values, uniform on `[0, 1]`.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import RandomBranchMapSystem
        >>> RandomBranchMapSystem(10, p1=0.3).prepare().system_.elements[0].size
        10
    """

    field = 'mathematics/stochastic_processes'
    topics = ('random map', 'iterated function system', 'interval map', 'stationary distribution')
    references = ('Dilão 2023',)
    until = 200
    dt = 1

    def __init__(
        self,
        n_orbits: int = 200,
        *,
        p1: float = 0.5,
        mu: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_orbits = n_orbits
        self.p1 = p1
        self.mu = mu
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the orbits and their random map."""
        super()._prepare(context)
        start = np.random.default_rng(self.random_state).uniform(0.0, 1.0, self.n_orbits)
        self.system_ = System(
            self.label,
            elements=[RandomBranchMapElement('orbits', x=start, size=self.n_orbits)],
            interactions=[RandomBranchMapInteraction('map', target='orbits', p1=self.p1, mu=self.mu)],
        )
