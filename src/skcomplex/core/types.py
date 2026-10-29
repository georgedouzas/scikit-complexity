"""Define the protocols of the core: the automata that analyses read, named by the spaces they live on."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
from sage.rings.polynomial.laurent_polynomial import LaurentPolynomial
from sage.structure.element import Matrix

from ..types import Indices, IntArray
from .spaces import BaseSpace, LatticeNeighborhood, MemoryNeighborhood


class AutomatonProtocol(Protocol):
    """Cellular automaton the analysis reads by its methods, such as `Automaton` of the automata models."""

    space: BaseSpace
    neighborhood: LatticeNeighborhood | MemoryNeighborhood
    rule: object
    states: int

    @property
    def reads(self) -> LatticeNeighborhood | MemoryNeighborhood:
        """Neighbourhood the rule reads."""

    @property
    def size(self) -> int:
        """Number of sites."""

    def table(self) -> Indices:
        """Return the site of every cell the rule reads at every site."""

    def local_map(self) -> IntArray:
        """Return the next value of every pattern of the cells the rule reads."""

    def step(self, configuration: npt.ArrayLike) -> IntArray:
        """Return the image of a configuration."""

    def run(self, configuration: npt.ArrayLike, steps: int) -> IntArray:
        """Return the configurations of a number of steps."""


class LinearAutomatonProtocol(Protocol):
    """Linear cellular automaton the analysis reads by its matrix and its Laurent polynomial."""

    def matrix(self) -> Matrix:
        """Return the Sage matrix of the automaton on its space."""

    def laurent_polynomial(self) -> LaurentPolynomial:
        """Return the Sage Laurent polynomial of the automaton on `Z^d`."""


class SubshiftProtocol(Protocol):
    """Subshift of finite type the analysis reads by its count of admissible configurations and its violations."""

    def count(self, space: BaseSpace) -> int:
        """Return the number of admissible configurations of a space."""

    def violations(self, configuration: npt.ArrayLike, space: BaseSpace) -> npt.NDArray[np.bool_]:
        """Return the sites at which a forbidden pattern appears."""
