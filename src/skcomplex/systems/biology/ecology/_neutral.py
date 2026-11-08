"""Ecology systems: neutral birth and death of many species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import CompartmentElement, GroupElement
from ....core.components.interactions import JumpInteraction
from ....core.components.laws.utils import where
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class NeutralSpeciesElement(CompartmentElement):
    """Species of neutral theory, a compartment of `n` individuals.

    Its birth and death rates are shared by every species.

    The births of `NeutralMacroecologySystem` occur at the rate `b0 + b1 n`, with immigration or speciation `b0`, and
    its deaths at the rate `d0 + d1 n`, with emigration or extinction `d0` while the species has members.

    Args:
        label:
            The label of the species or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of individuals.

        b0:
            The rate of immigration or speciation.

        b1:
            The per capita birth rate.

        d0:
            The rate of emigration or extinction.

        d1:
            The per capita death rate.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import NeutralSpeciesElement
        >>> GroupElement('trees', element=NeutralSpeciesElement, size=50).size
        50
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=NN)] = 10.0,
        b0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        b1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        d0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual, n=n)
        self.b0 = b0
        self.b1 = b1
        self.d0 = d0
        self.d1 = d1


class NeutralMacroecologySystem(BaseSystem):
    """Many species under neutral birth, death, and immigration.

    Their abundances follow a gamma-like law.

    Without emigration, `d0 = 0`, the abundances settle on the negative binomial distribution of mean `b0 / (d1 - b1)`,
    the discrete form of the gamma distribution that the Fokker-Planck equation of macroecology predicts and that fits
    the abundances of tropical trees. The system has the abundances `species.n`.

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

    Examples:
        >>> from skcomplex.systems.biology.ecology import NeutralMacroecologySystem
        >>> NeutralMacroecologySystem(20).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.species = species
        self.n = n
        self.b0 = b0
        self.b1 = b1
        self.d0 = d0
        self.d1 = d1

    def _build_system(self) -> System:
        """Return the parts of many species under neutral birth."""
        return System(
            self.label,
            elements=[
                GroupElement(
                    'species',
                    element=NeutralSpeciesElement,
                    size=self.species,
                    values={'n': self.n, 'b0': self.b0, 'b1': self.b1, 'd0': self.d0, 'd1': self.d1},
                ),
            ],
            interactions=[
                JumpInteraction(
                    'birth',
                    target='species',
                    changes={'n': 1},
                    rate=lambda s: s.target.b0 + s.target.b1 * s.target.n,
                ),
                JumpInteraction(
                    'death',
                    target='species',
                    changes={'n': -1},
                    rate=lambda s: where(s.target.n > 0, s.target.d0, 0) + s.target.d1 * s.target.n,
                ),
            ],
        )
