"""Macroeconomics systems: linear choices of agents that herd on the past choices of the agents they are linked to."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import CouplingInteraction
from ....core.components.laws.utils import normal
from ....core.components.system import System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system


class HerdingAgentElement(Element):
    """Agent whose choice `c` is a weighted sum of the past choices of others plus its individual characteristic.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        c:
            The choice of the agent, such as its consumption.

        x:
            The mean of the individual characteristic of the agent.

        gamma:
            The weight `gamma` of the individual characteristic in the choice.

        sigma:
            The standard deviation of the shocks of the individual characteristic.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.macroeconomics import HerdingAgentElement
        >>> GroupElement('agents', element=HerdingAgentElement, size=3, values={'x': [1.0, 2.0, 3.0]}).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[float, StateSpec()] = 0.0,
        x: Annotated[float, StateSpec()] = 1.0,
        gamma: Annotated[float, ParameterSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c
        self.x = x
        self.gamma = gamma
        self.sigma = sigma


def _choice(s: SimpleNamespace, herding: Expression) -> Expression:
    """Return the next choice, the herding term plus `gamma` times the shocked individual characteristic."""
    return herding + s.member.gamma * (s.member.x + s.member.sigma * normal('shock', 0, 1))


def _random_links(n_agents: int, density: float, herding: float, rng: np.random.Generator) -> Array:
    """Return random directed weights with no self-links, every row of which sums to `herding`."""
    links = (rng.random((n_agents, n_agents)) < density) * rng.random((n_agents, n_agents))
    np.fill_diagonal(links, 0.0)
    rows = links.sum(axis=1, keepdims=True)
    return np.divide(herding * links, rows, out=np.zeros_like(links), where=rows > 0)


@register_system(
    field='economics/macroeconomics',
    tags=('herding', 'network', 'aggregation', 'agents', 'linear map'),
    references=('Caiani et al. 2016 §1.1.6',),
    defaults={'until': 50},
    observables=lambda system: [ObservableSpec('mean_choice', group='agents', variable='c', reduce='mean')],
)
def network_herding(
    theta: Sequence[Sequence[float]] | None = None,
    x: Sequence[float] | None = None,
    *,
    gamma: float = 1.0,
    sigma: float = 0.1,
    n_agents: int = 50,
    density: float = 0.1,
    herding: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return agents whose linear choices herd on the past choices of the agents they are linked to.

    The choice of agent `i` is `c_it = sum_j theta_ij c_j,t-1 + gamma x_it`, so the choices follow
    `c_t = Theta c_t-1 + gamma x_t` on the weighted network `Theta`, without self-links, Eqs. 1.10 and 1.11 of Caiani
    et al. (2016, Section 1.1.6). The individual characteristic is `x_it = x_i + sigma eps_it`, with a standard normal
    shock `eps_it`. The mean choice follows `c_t = (1/N) sum_i d_i c_i,t-1 + gamma x_t` (Eq. 1.12), where the out-degree
    `d_i` of agent `i`, the sum of column `i` of `Theta`, measures how much its choice moves the others, so the
    persistence of the aggregate depends on the out-degrees and on how they correlate with the choices. Without
    shocks the choices settle at `(I - Theta)^-1 gamma x` when the spectral radius of `Theta` is below 1.

    Args:
        theta:
            The weights, `theta[i][j]` the weight of the past choice of `j` in the choice of `i`, or `None` for a random
            network of `n_agents` agents.

        x:
            The means of the individual characteristics, or `None` for standard lognormal draws.

        gamma:
            The weight of the individual characteristic.

        sigma:
            The standard deviation of the shocks of the individual characteristics.

        n_agents:
            The number of agents of a random network.

        density:
            The probability of a link from one agent to another in a random network.

        herding:
            The sum of the weights of every agent with links in a random network, below 1 for stable choices.

        random_state:
            The seed or generator of the random network and characteristics.

    Returns:
        system:
            The system, with the group `agents` of the choices `agents.c` and the herding `herding`. The observable of
            the system is the mean choice `mean_choice`.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import network_herding
        >>> network_herding([[0.0, 0.5], [0.5, 0.0]], [1.0, 1.0]).elements[0].prepare().size_
        2
    """
    rng = np.random.default_rng(random_state)
    weights = _random_links(n_agents, density, herding, rng) if theta is None else np.asarray(theta, dtype=float)
    size = len(weights)
    means = rng.lognormal(0.0, 1.0, size) if x is None else np.asarray(x, dtype=float)
    agents = GroupElement(
        'agents',
        element=HerdingAgentElement,
        size=size,
        values={'c': gamma * means, 'x': means, 'gamma': gamma, 'sigma': sigma},
    )
    herding_map = CouplingInteraction(
        'herding',
        target='agents',
        read='c',
        write='c',
        function=lambda s: s.other.c,
        reduce='sum',
        mode='group',
        weights=weights.tolist(),
        form='map',
        combine=_choice,
    )
    return System(elements=[agents], interactions=[herding_map])
