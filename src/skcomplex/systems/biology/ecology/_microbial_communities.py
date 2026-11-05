"""Ecology systems: microbial communities of species that grow and interact in pairs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from ....core.components.elements import GroupElement
from ....core.components.interactions import RateInteraction
from ....core.components.system import System
from ....core.errors import ValidationError
from ..._catalog import register_system
from ._generalized_lotka_volterra import GeneralizedLotkaVolterraInteraction
from ._microbe import MicrobeElement

COMMUNITY_GROWTH = (1.0, 0.8, 0.6)


COMMUNITY_INTERACTIONS = ((-1.0, -0.3, 0.2), (-0.2, -1.0, -0.3), (0.1, -0.2, -1.0))


def _growth(namespace: SimpleNamespace) -> object:
    """Return the intrinsic growth `b x` of a species."""
    return namespace.target.b * namespace.target.x


@register_system(
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
    """Return the generalized Lotka-Volterra system of a microbial community, `dx_i/dt = x_i (b_i + sum_j a_ij x_j)`.

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
            The system, with the abundances `species.x` of the members of the group `species`, the growth `growth`, and
            the interactions `glv`, whose matrix is the parameter `glv.a`.

    Raises:
        ValidationError:
            The growth rates, the matrix, and the abundances do not have one entry per species.

    Examples:
        >>> from skcomplex.systems.biology.ecology import generalized_lotka_volterra
        >>> generalized_lotka_volterra().elements[0].prepare().size_
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
        interactions=[
            RateInteraction('growth', target='species', rate=_growth),
            GeneralizedLotkaVolterraInteraction('glv', target='species', a=matrix),
        ],
    )
