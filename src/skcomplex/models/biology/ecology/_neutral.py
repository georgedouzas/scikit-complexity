"""Ecology models: neutral birth and death of many species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.components import Element
from ....core.laws import Jump, Law, Symbols, where
from ....core.runtime import register_model
from ....core.system import System


class NeutralSpecies(Element):
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
        >>> from skcomplex.models.biology.ecology import NeutralSpecies
        >>> NeutralSpecies.group('trees', 50).size
        50
    """

    n: float = state(default=10.0, domain=Domain.NONNEGATIVE_INTEGER)
    b0: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    b1: float = parameter(default=0.9, domain=Domain.NONNEGATIVE)
    d0: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the birth and death events.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Jump(s.b0 + s.b1 * s.n, {s.n: 1}),
            Jump(where(s.n > 0, s.d0, 0) + s.d1 * s.n, {s.n: -1}),
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
        >>> neutral_macroecology(20).elements[0].size
        20
    """
    return System(components=[NeutralSpecies.group('species', species, n=n, b0=b0, b1=b1, d0=d0, d1=d1)])
