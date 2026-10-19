"""Chemical kinetics models: reactions and excitable media."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, parameter, state
from ....core.model import (
    Element,
    LatticeSpace,
    Law,
    LocalRule,
    Neighborhood,
    Symbols,
    System,
)
from ....core.runtime import register_model

type Array = npt.NDArray[np.float64]


class HodgepodgeCell(Element):
    """Cell of the hodgepodge machine of Gerhardt and Schuster, healthy at 0, infected below `n`, and ill at `n`.

    Healthy cells catch the infection from their infected and ill neighbours, infected cells worsen by the mean of their
    neighbourhood plus `g`, and ill cells heal.

    Args:
        label:
            The label of the cell or the lattice group.

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
        >>> from skcomplex.models.chemistry.kinetics import HodgepodgeCell
        >>> HodgepodgeCell('cell').n
        100.0
    """

    level: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    n: float = parameter(default=100.0, domain=Domain.POSITIVE)
    k1: float = parameter(default=2.0, domain=Domain.POSITIVE)
    k2: float = parameter(default=3.0, domain=Domain.POSITIVE)
    g: float = parameter(default=28.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule of infection, worsening, and healing.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """

        def rule(level: Array, *around: Array, n: Any, k1: Any, k2: Any, g: Any) -> Array:  # noqa: ANN401
            neighbours = np.stack(around)
            infected = ((neighbours > 0) & (neighbours < n)).sum(axis=0)
            ill = (neighbours >= n).sum(axis=0)
            healthy = np.floor(infected / k1) + np.floor(ill / k2)
            worsening = np.minimum(np.floor((level + neighbours.sum(axis=0)) / (infected + ill + 1)) + g, n)
            return np.where(level <= 0, np.minimum(healthy, n), np.where(level >= n, 0.0, worsening))

        return [LocalRule(s.level, Neighborhood.moore(center=True), rule)]


@register_model(
    field='chemistry/kinetics',
    tags=('lattice', 'excitable media', 'spirals'),
    references=('Gerhardt and Schuster 1989', 'Charbonneau 2017'),
    defaults={'until': 200, 'dt': 1},
)
def hodgepodge(
    shape: Sequence[int] = (100, 100),
    *,
    n: float = 100.0,
    k1: float = 2.0,
    k2: float = 3.0,
    g: float = 28.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the hodgepodge machine, a model of a chemical reaction whose waves form spirals.

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

    Returns:
        system:
            The model, with the levels `cells.level`.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import hodgepodge
        >>> hodgepodge((10, 10)).elements[0].values['g']
        28.0
    """
    levels = np.random.default_rng(random_state).integers(0, int(n) + 1, size=tuple(shape)).astype(float)
    space = LatticeSpace(tuple(shape))
    return System(elements=[HodgepodgeCell.lattice('cells', space, level=levels, n=n, k1=k1, k2=k2, g=g)])
