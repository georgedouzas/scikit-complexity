"""Ecology models: neutral birth and death of many species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, JumpLaw
from ....core.laws.utils import where
from ....core.system import System
from ....engine.runtime import register_model
from ....specs import ParameterSpec, StateSpec, VisualSpec


class NeutralSpeciesElement(Element):
    """Species whose individuals are born and die at rates that do not depend on the species, of neutral theory.

    Births occur at the rate `b0 + b1 n`, with immigration or speciation `b0`, and deaths at the rate `d0 + d1 n`,
    with emigration or extinction `d0` while the species has members.

    Args:
        label:
            The label of the species or the group.

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
        super().__init__(label, visual=visual)
        self.n = n
        self.b0 = b0
        self.b1 = b1
        self.d0 = d0
        self.d1 = d1

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the birth and death events.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            JumpLaw(s.b0 + s.b1 * s.n, {s.n: 1}),
            JumpLaw(where(s.n > 0, s.d0, 0) + s.d1 * s.n, {s.n: -1}),
        ]


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
    )
