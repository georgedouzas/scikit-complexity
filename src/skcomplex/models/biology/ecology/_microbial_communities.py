"""Ecology models: microbial communities of species that grow and interact in pairs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.errors import ValidationError
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import total
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.system import System
from ..._catalog import register_model

COMMUNITY_GROWTH = (1.0, 0.8, 0.6)
COMMUNITY_INTERACTIONS = ((-1.0, -0.3, 0.2), (-0.2, -1.0, -0.3), (0.1, -0.2, -1.0))


class MicrobeElement(Element):
    """Species of a microbial community, with an abundance and an intrinsic growth rate.

    Args:
        label:
            The label of the species or of the group of species.

        x:
            The initial abundance.

        b:
            The intrinsic growth rate.

    Examples:
        >>> from skcomplex.models.biology.ecology import MicrobeElement
        >>> MicrobeElement('species', x=0.1, b=0.8).b
        0.8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        b: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.b = b


class GeneralizedLotkaVolterraInteraction(Interaction):
    """Growth of the species of a group, `dx_i/dt = x_i (b_i + sum_j a_ij x_j)`, with an interaction matrix `a`.

    The sum over the species is a total over the group weighted by row `i` of the matrix. The diagonal entries are the
    interactions of a species with itself, negative for self-limitation.

    Args:
        label:
            The label of the interaction.

        a:
            The interaction matrix, where `a_ij` is the effect of species `j` on the growth of species `i`.

        **values:
            The values of the declared variables, by name:

            community:
                The path of the group of species.

    Examples:
        >>> from skcomplex.models.biology.ecology import GeneralizedLotkaVolterraInteraction
        >>> GeneralizedLotkaVolterraInteraction('glv', community='species', a=((-1.0, 0.5), (0.5, -1.0))).a
        ((-1.0, 0.5), (0.5, -1.0))
    """

    def __init__(
        self,
        label: str,
        *,
        community: Annotated[str, RoleSpec()],
        a: Sequence[Sequence[float]] = ((-1.0,),),
    ) -> None:
        super().__init__(label)
        self.community = community
        self.a = a

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the generalized Lotka-Volterra growth of every species.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        x = s.community.x
        return [RateLaw(x, x * (s.community.b + total(x, weights=np.asarray(self.a, dtype=float))))]


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
        >>> generalized_lotka_volterra().elements[0].check().prepare().size_
        3
    """
    size = len(b)
    abundances = [0.1] * size if x is None else [float(value) for value in x]
    if len(abundances) != size or len(a) != size or any(len(row) != size for row in a):
        error_msg = f'The abundances and the interaction matrix must have one entry per species, {size} growth rates.'
        raise ValidationError(error_msg)
    species = GroupElement(
        'species',
        element=MicrobeElement,
        size=size,
        values={'x': abundances, 'b': [float(value) for value in b]},
    )
    matrix = tuple(tuple(float(value) for value in row) for row in a)
    return System(
        elements=[species],
        interactions=[GeneralizedLotkaVolterraInteraction('glv', community='species', a=matrix)],
    )
