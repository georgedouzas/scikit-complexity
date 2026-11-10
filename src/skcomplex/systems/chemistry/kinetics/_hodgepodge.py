"""Chemical kinetics systems: reactions and excitable media."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import BallNeighborhood, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array


class HodgepodgeCellElement(Element):
    """Cell of the hodgepodge machine of Gerhardt and Schuster.

    It is healthy at 0, infected below `n`, and ill at `n`.

    Healthy cells catch the infection from their infected and ill neighbours, infected cells worsen by the mean of their
    neighbourhood plus `g`, and ill cells heal, by the local rule of `HodgepodgeSystem`.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        level:
            The initial level of infection from 0 to `n`.

        n:
            The level of illness.

        k1:
            The weight of infected neighbours for healthy cells.

        k2:
            The weight of ill neighbours for healthy cells.

        g:
            The speed of the infection.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import HodgepodgeCellElement
        >>> HodgepodgeCellElement('cell').n
        100.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        level: Annotated[float, StateSpec(domain=NN)] = 0,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        k1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        k2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 34.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
        self.n = n
        self.k1 = k1
        self.k2 = k2
        self.g = g


class HodgepodgeSystem(BaseSystem):
    """The hodgepodge machine, a chemical reaction whose waves form spirals.

    Every cell of a lattice holds a level of infection from 0 to `n`. Healthy cells catch the infection from their
    infected and ill Moore neighbours, infected cells worsen by the mean of their neighbourhood plus `g`, and ill cells
    heal. The levels are `cells.level`.

    Args:
        shape:
            The number of rows and columns.

        n:
            The level of illness.

        k1:
            The weight of infected neighbours for healthy cells.

        k2:
            The weight of ill neighbours for healthy cells.

        g:
            The speed of the infection.

        random_state:
            The seed or generator of the initial levels.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import HodgepodgeSystem
        >>> HodgepodgeSystem((10, 10)).prepare().system_.elements[0].values['g']
        34.0
    """

    field = 'chemistry/kinetics'
    topics = ('lattice', 'excitable media', 'spirals')
    references = ('Gerhardt and Schuster 1989', 'Charbonneau 2017')
    until = 200
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (100, 100),
        *,
        n: float = 100.0,
        k1: float = 2.0,
        k2: float = 3.0,
        g: float = 34.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.n = n
        self.k1 = k1
        self.k2 = k2
        self.g = g
        self.random_state = random_state

    @staticmethod
    def _infect(
        level: Array,
        *around: Array,
        n: float | Array,
        k1: float | Array,
        k2: float | Array,
        g: float | Array,
    ) -> Array:
        """Return the next levels.

        Healthy cells catch the infection, infected cells worsen, and ill cells heal.
        """
        neighbours = np.stack(around)
        infected = ((neighbours > 0) & (neighbours < n)).sum(axis=0)
        ill = (neighbours >= n).sum(axis=0)
        healthy = np.floor(infected / k1) + np.floor(ill / k2)
        worsening = np.minimum(np.floor((level + neighbours.sum(axis=0)) / (infected + ill + 1)) + g, n)
        return np.where(level <= 0, np.minimum(healthy, n), np.where(level >= n, 0.0, worsening))

    def _build_system(self) -> CompositeSystem:
        """Return the lattice of cells with random levels and the rule of the infection."""
        shape = tuple(self.shape)
        levels = np.random.default_rng(self.random_state).integers(0, int(self.n) + 1, size=shape).astype(float)
        values = {'level': levels, 'n': self.n, 'k1': self.k1, 'k2': self.k2, 'g': self.g}
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'cells',
                    element=HodgepodgeCellElement,
                    space=LatticeSpace(shape),
                    values=values,
                    on='sites',
                ),
            ],
            interactions=[
                LocalRuleInteraction(
                    'infection',
                    target='cells',
                    state='level',
                    rule=self._infect,
                    neighborhood=BallNeighborhood(center='first'),
                ),
            ],
        )
