"""Ecology systems: random Lotka-Volterra communities with interactions among three species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from itertools import combinations

import numpy as np
import numpy.typing as npt
from scipy.special import comb

from ....core.components.elements import GroupElement
from ....core.components.system import System
from ..._catalog import register_system
from ..common.populations import LogisticGrowthInteraction, PopulationElement
from ._triplet import TripletInteraction


def _draw_tensor(
    size: int,
    mu: float,
    sigma: float,
    gamma: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Return the coupling tensor with the moments and the correlation of Sidhom and Galla, equations 4 and 5."""
    pairs = comb(size - 1, 2)
    correlation = (1 - gamma) * np.eye(3) + gamma * np.ones((3, 3))
    triples = np.array(list(combinations(range(size), 3)))
    values = rng.multivariate_normal(np.full(3, mu / pairs), 1.5 * sigma**2 / pairs * correlation, size=len(triples))
    tensor = np.zeros((size, size, size))
    for position in range(3):
        i, j, k = (triples[:, (position + shift) % 3] for shift in range(3))
        tensor[i, j, k] = tensor[i, k, j] = values[:, position]
    return tensor


@register_system(
    field='biology/ecology',
    tags=('random community', 'Lotka-Volterra', 'higher-order interactions', 'coupling tensor', 'ode'),
    references=('Sidhom and Galla 2024',),
    defaults={'until': 1000.0, 'dt': 10.0},
)
def higher_order_random_lotka_volterra(
    size: int = 200,
    *,
    mu: float = -4.0,
    sigma: float = 1.0,
    gamma: float = 0.0,
    k: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a random Lotka-Volterra community with third-order interactions.

    The abundances follow `dx_i/dt = x_i (k - x_i + sum_(j<k) a_ijk x_j x_k)` (Sidhom and Galla 2024, Eq. 2), where the
    sum runs over the pairs of other species. The coefficients are Gaussian with mean `mu / binom(N - 1, 2)` and
    variance `3 sigma^2 / (2 binom(N - 1, 2))`, and the three coefficients of a set of three species have the
    correlation `gamma` (Eqs. 4 and 5). The defaults are the competitive community of the paper's Fig. 2, 200 species
    with uncorrelated coefficients at `mu = -4` and `sigma = 1`, in its phase with a unique stable fixed point, where
    the theory gives a fraction `0.928` of surviving species and a mean abundance `0.392`. The paper does not state the
    distribution of the initial abundances, which are drawn uniformly from the interval from 0 to 1.

    Args:
        size:
            The number of species.

        mu:
            The mean of the interactions, negative for competition.

        sigma:
            The heterogeneity of the interactions.

        gamma:
            The correlation of the three coefficients of a set of three species, from `-1/2` to 1, where 1 gives every
            species of a set the same coefficient.

        k:
            The growth rate at low abundance, which is also the carrying capacity without interactions.

        random_state:
            The seed or the generator that draws the tensor and the initial abundances.

    Returns:
        system:
            The system, with the abundances `species.n` of the members of the group `species`, the growth `growth`, and
            the interactions `triplets`, whose tensor is the parameter `triplets.a`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import higher_order_random_lotka_volterra
        >>> higher_order_random_lotka_volterra(5).interactions[1].a.shape
        (5, 5, 5)
    """
    rng = np.random.default_rng(random_state)
    tensor = _draw_tensor(size, mu, sigma, gamma, rng)
    species = GroupElement('species', element=PopulationElement, size=size, values={'n': rng.random(size)})
    return System(
        elements=[species],
        interactions=[
            LogisticGrowthInteraction('growth', target='species', r=k, K=k),
            TripletInteraction('triplets', target='species', a=tensor, state='n'),
        ],
    )
