"""Ecology systems: neutral birth and death of many species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import NN

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ..common.populations import NeutralBirthInteraction, NeutralDeathInteraction


class NeutralSpeciesElement(Element):
    """Species of neutral theory, a compartment of `x` individuals.

    Args:
        label:
            The label of the species or of the group of species.

        x:
            The initial number of individuals.

        size:
            The number of species, or `None` for one.

        space:
            The space with one species per site, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.ecology import NeutralSpeciesElement
        >>> NeutralSpeciesElement('trees', size=50).size
        50
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float | npt.ArrayLike, NN] = 10.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x


class NeutralMacroecologySystem(BaseSystem):
    """Many species under neutral birth, death, and immigration.

    Their abundances follow a gamma-like law.

    Without emigration, `d0 = 0`, the abundances settle on the negative binomial distribution of mean `b0 / (d1 - b1)`,
    the discrete form of the gamma distribution that the Fokker-Planck equation of macroecology predicts and that fits
    the abundances of tropical trees. The system has the abundances `species.x`.

    Args:
        species:
            The number of species.

        n:
            The initial number of individuals of every species.

        b0:
            The rate of immigration or speciation.

        b1:
            The per capita birth rate.

        d0:
            The rate of emigration or extinction.

        d1:
            The per capita death rate, above `b1`.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.ecology import NeutralMacroecologySystem
        >>> NeutralMacroecologySystem(20).prepare().system_.elements[0].size
        20
    """

    field = 'biology/ecology'
    topics = ('macroecology', 'neutral theory', 'jump process', 'species abundance')
    references = ('Hubbell 2001', 'Volkov et al. 2003', 'Gros 2015')
    until = 100.0
    dt = 1.0

    def __init__(
        self,
        species: int = 100,
        n: float = 10.0,
        *,
        b0: float = 1.0,
        b1: float = 0.9,
        d0: float = 0.0,
        d1: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.species = species
        self.n = n
        self.b0 = b0
        self.b1 = b1
        self.d0 = d0
        self.d1 = d1

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of many species under neutral birth.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[NeutralSpeciesElement('species', x=self.n, size=self.species)],
            interactions=[
                NeutralBirthInteraction('birth', target='species', b0=self.b0, b1=self.b1),
                NeutralDeathInteraction('death', target='species', d0=self.d0, d1=self.d1),
            ],
        )
