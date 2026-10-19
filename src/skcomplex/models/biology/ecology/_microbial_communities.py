"""Ecology models: microbial communities of species that grow and interact in pairs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.base import Domain, ValidationError, parameter, role, state
from ....core.model import Element, Interaction, Law, Rate, Symbols, System, total
from ....core.runtime import register_model

COMMUNITY_GROWTH = (1.0, 0.8, 0.6)
COMMUNITY_INTERACTIONS = ((-1.0, -0.3, 0.2), (-0.2, -1.0, -0.3), (0.1, -0.2, -1.0))


class Microbe(Element):
    """Species of a microbial community, with an abundance and an intrinsic growth rate.

    Args:
        label:
            The label of the species or of the group of species.

        x:
            The initial abundance.

        b:
            The intrinsic growth rate.

    Examples:
        >>> from skcomplex.models.biology.ecology import Microbe
        >>> Microbe('species', x=0.1, b=0.8).b
        0.8
    """

    x: float = state(default=0.1, domain=Domain.NONNEGATIVE)
    b: float = parameter(default=1.0, domain=Domain.REAL)


class GeneralizedLotkaVolterra(Interaction):
    """Growth of the species of a group, `dx_i/dt = x_i (b_i + sum_j a_ij x_j)`, with an interaction matrix `a`.

    The sum over the species is a total over the group weighted by row `i` of the matrix. The diagonal entries are the
    interactions of a species with itself, negative for self-limitation.

    Args:
        label:
            The label of the interaction.

        community:
            The path of the group of species.

        a:
            The interaction matrix, where `a_ij` is the effect of species `j` on the growth of species `i`.

    Examples:
        >>> from skcomplex.models.biology.ecology import GeneralizedLotkaVolterra
        >>> GeneralizedLotkaVolterra('glv', community='species', a=((-1.0, 0.5), (0.5, -1.0))).a
        ((-1.0, 0.5), (0.5, -1.0))
    """

    community: str = role()
    a: Sequence[Sequence[float]] = ((-1.0,),)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the generalized Lotka-Volterra growth of every species.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        x = s.community.x
        return [Rate(x, x * (s.community.b + total(x, weights=np.asarray(self.a, dtype=float))))]


@register_model(
    field='biology/ecology',
    tags=('microbial community', 'Lotka-Volterra', 'interaction matrix', 'ode'),
    references=('Raman 2021', 'Stein et al. 2013'),
    defaults={'until': 30.0, 'dt': 0.1},
)
def generalized_lotka_volterra(
    x: Sequence[float] | None = None,
    *,
    b: Sequence[float] = COMMUNITY_GROWTH,
    a: Sequence[Sequence[float]] = COMMUNITY_INTERACTIONS,
) -> System:
    """Return the generalized Lotka-Volterra model of a microbial community, `dx_i/dt = x_i (b_i + sum_j a_ij x_j)`.

    Every species grows at its intrinsic rate `b_i` and its growth changes with the abundance of every species through
    the pairwise interaction `a_ij`. A feasible interior fixed point solves `b + a x = 0`. The book gives no parameter
    values. The default is a choice, a community of three self-limiting species with weak competition and one weak
    mutualism, whose fixed point `x = -a^-1 b` is positive and stable, so the species coexist. Abundances and time are
    in arbitrary units.

    Args:
        x:
            The initial abundances, or `None` for 0.1 each.

        b:
            The intrinsic growth rates.

        a:
            The interaction matrix, one row per species.

    Returns:
        system:
            The model, with the abundances `species.x` of the members of the group `species`.

    Raises:
        ValidationError:
            The growth rates, the matrix, and the abundances do not have one entry per species.

    Examples:
        >>> from skcomplex.models.biology.ecology import generalized_lotka_volterra
        >>> generalized_lotka_volterra().elements[0].size
        3
    """
    size = len(b)
    abundances = [0.1] * size if x is None else [float(value) for value in x]
    if len(abundances) != size or len(a) != size or any(len(row) != size for row in a):
        error_msg = f'The abundances and the interaction matrix must have one entry per species, {size} growth rates.'
        raise ValidationError(error_msg)
    species = Microbe.group('species', size, x=abundances, b=[float(value) for value in b])
    matrix = tuple(tuple(float(value) for value in row) for row in a)
    return System(elements=[species], interactions=[GeneralizedLotkaVolterra('glv', community='species', a=matrix)])
