"""Ecology models: a neutral community of constant size whose species arise by speciation and vanish by drift."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.system import System
from ....specs import ObservableSpec
from ....types import AnyArray
from ..._catalog import register_model
from ._neutral_replacement import NeutralReplacementInteraction
from ._organism import OrganismElement


def _richness(species: AnyArray) -> float:
    """Return the number of distinct species among the individuals."""
    return float(len(np.unique(species)))


@register_model(
    field='biology/ecology',
    tags=('neutral theory', 'biodiversity', 'speciation', 'agents', 'species abundance'),
    references=('Bertin 2021', 'Hubbell 2001', 'Vallade and Houchmandzadeh 2003'),
    defaults={'until': 500},
    observables=lambda system: [
        ObservableSpec('richness', group='community', variable='species', reduce=_richness),
    ],
)
def neutral_biodiversity(
    individuals: int = 200,
    *,
    nu: float = 0.01,
    species: int = 1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a neutral community of constant size, where species arise by speciation and vanish by drift.

    Every species is transient, yet the number of species settles around `S_N = sum_n theta / (theta + n)` for `n`
    from 0 to `N - 1`, with `theta = (N - 1) nu / (1 - nu)`, and the mean number of species of `k` individuals around
    `phi_k = theta / k prod_j (N - j + 1) / (N - j + theta)` for `j` from 1 to `k`.

    Args:
        individuals:
            The number of individuals `N`.

        nu:
            The probability that an offspring founds a new species.

        species:
            The initial number of species, to which the individuals belong at random.

        random_state:
            The seed or generator of the initial species.

    Returns:
        system:
            The model, with the group `community` and the number `richness` of species.

    Examples:
        >>> from skcomplex.models.biology.ecology import neutral_biodiversity
        >>> neutral_biodiversity(20).elements[0].prepare().size_
        20
    """
    labels = np.random.default_rng(random_state).integers(species, size=individuals).astype(float)
    return System(
        elements=[GroupElement('community', element=OrganismElement, size=individuals, values={'species': labels})],
        interactions=[NeutralReplacementInteraction('replacement', target='community', mutation=nu)],
    )
