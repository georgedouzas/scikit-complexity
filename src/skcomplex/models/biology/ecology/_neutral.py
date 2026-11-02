"""Ecology models: neutral birth and death of many species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import CompartmentElement, GroupElement
from ....core.interactions import JumpInteraction
from ....core.laws.utils import where
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class NeutralSpeciesElement(CompartmentElement):
    """Species of neutral theory, a compartment of `n` individuals with birth and death rates shared by every species.

    The births of `neutral_macroecology` occur at the rate `b0 + b1 n`, with immigration or speciation `b0`, and its
    deaths at the rate `d0 + d1 n`, with emigration or extinction `d0` while the species has members.

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
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.ecology import NeutralSpeciesElement
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


@register_model(
    field='biology/ecology',
    tags=('macroecology', 'neutral theory', 'jump process', 'species abundance'),
    references=('Hubbell 2001', 'Volkov et al. 2003', 'Gros 2015'),
    defaults={'until': 100.0, 'dt': 1.0},
)
def neutral_macroecology(
    species: int = 100,
    n: float = 10.0,
    *,
    b0: float = 1.0,
    b1: float = 0.9,
    d0: float = 0.0,
    d1: float = 1.0,
) -> System:
    """Return many species under neutral birth, death, and immigration, whose abundances follow a gamma-like law.

    Without emigration, `d0 = 0`, the abundances settle on the negative binomial distribution of mean
    `b0 / (d1 - b1)`, the discrete form of the gamma distribution that the Fokker-Planck equation of macroecology
    predicts and that fits the abundances of tropical trees.

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

    Returns:
        system:
            The model, with the abundances `species.n`.

    Examples:
        >>> from skcomplex.models.biology.ecology import neutral_macroecology
        >>> neutral_macroecology(20).elements[0].check().prepare().size_
        20
    """
    return System(
        elements=[
            GroupElement(
                'species',
                element=NeutralSpeciesElement,
                size=species,
                values={'n': n, 'b0': b0, 'b1': b1, 'd0': d0, 'd1': d1},
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
