"""Random Lotka-Volterra communities with interactions among three species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from itertools import combinations
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from scipy.special import comb

from ....core.components.elements import GroupElement
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import contraction
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec
from ..common.populations import LogisticGrowthInteraction, PopulationElement


class TripletInteraction(RateInteraction):
    """Interactions of every species with the pairs of other species.

    They add `dx_i/dt += x_i sum_(j<k) a_ijk x_j x_k`.

    The coefficient `a_ijk` is the effect of the combination of the species `j` and `k` on the growth of the species
    `i` (Sidhom and Galla 2024, Eq. 2). The tensor holds it for both orders of the pair, `a_ijk = a_ikj`, so the sum
    over the sets `j < k` is half the contraction of the tensor with the abundances, and it holds 0 where indices
    coincide.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of species.

        a:
            The coupling tensor of order 3, with one index per species, or a number for every entry.

        state:
            The name of the state that holds the abundance of a species.

    Examples:
        >>> from skcomplex.systems.biology.ecology import TripletInteraction
        >>> TripletInteraction('triplets', target='species', a=0.0, state='n').state
        'n'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        a: Annotated[object, ParameterSpec(shape=('target', 'target', 'target'))] = 0.0,
        state: str = 'x',
    ) -> None:
        super().__init__(label, target=target, state=state, rate=self._triplets)
        self.a = a

    def _triplets(self, namespace: SimpleNamespace) -> object:
        """Return the growth of a species from the pairs of other species."""
        abundance = getattr(namespace.target, self.state)
        return abundance * contraction(namespace.a, abundance, abundance) / 2


class HigherOrderRandomLotkaVolterraSystem(BaseSystem):
    """A random Lotka-Volterra community with third-order interactions.

    The abundances follow `dx_i/dt = x_i (k - x_i + sum_(j<k) a_ijk x_j x_k)` (Sidhom and Galla 2024, Eq. 2), where the
    sum runs over the pairs of other species. The coefficients are Gaussian with mean `mu / binom(N - 1, 2)` and
    variance `3 sigma^2 / (2 binom(N - 1, 2))`, and the three coefficients of a set of three species have the
    correlation `gamma` (Eqs. 4 and 5). The defaults are the competitive community of the paper's Fig. 2, 200 species
    with uncorrelated coefficients at `mu = -4` and `sigma = 1`, in its phase with a unique stable fixed point, where
    the theory gives a fraction `0.928` of surviving species and a mean abundance `0.392`. The paper does not state the
    distribution of the initial abundances, which are drawn uniformly from the interval from 0 to 1. The system has the
    abundances `species.n` of the members of the group `species`, the growth `growth`, and the interactions `triplets`,
    whose tensor is the parameter `triplets.a`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import HigherOrderRandomLotkaVolterraSystem
        >>> HigherOrderRandomLotkaVolterraSystem(5).prepare().system_.interactions[1].a.shape
        (5, 5, 5)
    """

    field = 'biology/ecology'
    topics = ('random community', 'Lotka-Volterra', 'higher-order interactions', 'coupling tensor', 'ode')
    references = ('Sidhom and Galla 2024',)
    until = 1000.0
    dt = 10.0

    def __init__(
        self,
        size: int = 200,
        *,
        mu: float = -4.0,
        sigma: float = 1.0,
        gamma: float = 0.0,
        k: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.k = k
        self.random_state = random_state

    @staticmethod
    def _draw_tensor(
        size: int,
        mu: float,
        sigma: float,
        gamma: float,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """Return the coupling tensor with the moments and the correlation of the source.

        They follow equations 4 and 5 of Sidhom and Galla.
        """
        pairs = comb(size - 1, 2)
        correlation = (1 - gamma) * np.eye(3) + gamma * np.ones((3, 3))
        triples = np.array(list(combinations(range(size), 3)))
        values = rng.multivariate_normal(
            np.full(3, mu / pairs),
            1.5 * sigma**2 / pairs * correlation,
            size=len(triples),
        )
        tensor = np.zeros((size, size, size))
        for position in range(3):
            i, j, k = (triples[:, (position + shift) % 3] for shift in range(3))
            tensor[i, j, k] = tensor[i, k, j] = values[:, position]
        return tensor

    def _build_system(self) -> System:
        """Return the parts of a random Lotka-Volterra community of third order.

        Its interactions are of third order.
        """
        rng = np.random.default_rng(self.random_state)
        tensor = self._draw_tensor(self.size, self.mu, self.sigma, self.gamma, rng)
        species = GroupElement(
            'species',
            element=PopulationElement,
            size=self.size,
            values={'n': rng.random(self.size)},
        )
        return System(
            self.label,
            elements=[species],
            interactions=[
                LogisticGrowthInteraction('growth', target='species', r=self.k, K=self.k),
                TripletInteraction('triplets', target='species', a=tensor, state='n'),
            ],
        )
