"""Stochastic process models: random walks and diffusions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import AgentType, Element, Jump, Law, Noise, Rate, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

RandomWalker = Element.define(
    'RandomWalker',
    states={'x': state(default=0.0, domain=Domain.INTEGER)},
    parameters={'rate': parameter(1.0, domain=Domain.NONNEGATIVE), 'bias': parameter(0.5, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [Jump(s.rate * s.bias, {s.x: 1}), Jump(s.rate * (1 - s.bias), {s.x: -1})],
)

WeakeningParticle = Element.define(
    'WeakeningParticle',
    states={'x': 0.0},
    parameters={
        'k': parameter(1.0, domain=Domain.POSITIVE),
        'ell': parameter(1.0, domain=Domain.POSITIVE),
        'D': parameter(0.25, domain=Domain.POSITIVE),
    },
    laws=lambda s: [Rate(s.x, -s.k * s.x / (1 + (s.x / s.ell) ** 2)), Noise(s.x, (2 * s.D) ** (1 / 2))],
)


class AgeingWalker(AgentType):
    """Walker that steps up with a probability that decays with the number of its earlier up steps.

    After `k` up steps, the next step is up with probability `p / (1 + k)^nu` and down otherwise.

    Args:
        label:
            The label of the agent type or group.

        location:
            Unused, since walkers have no space.

        x:
            The initial position.

        ups:
            The initial number of up steps.

        p:
            The probability of the first up step.

        nu:
            The exponent of the decay of the up probability with the number of up steps.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import AgeingWalker
        >>> AgeingWalker.group('walkers', 10, nu=0.5)
        Group('walkers', element=AgeingWalker, size=10)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'x'})
    x: float = state(default=0.0, domain=Domain.INTEGER)
    ups: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    p: float = parameter(0.5, domain=Domain.NONNEGATIVE)
    nu: float = parameter(1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the stepping rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.walk)]

    def walk(self, agent: AgentView, model: ModelView) -> None:
        """Take one step up or down.

        Args:
            agent:
                The walker.

            model:
                The model.
        """
        if model.rng.random() < agent.p / (1 + agent.ups) ** agent.nu:
            agent.x += 1
            agent.ups += 1
            return
        agent.x -= 1


@register_model(
    field='mathematics/stochastic_processes',
    tags=('random walk', 'jump process', 'diffusion'),
    references=('Pearson 1905', 'Argun et al. 2021', 'Charbonneau 2017', 'Jensen 2023'),
    defaults={'until': 100.0, 'dt': 1.0},
)
def random_walk(n_walkers: int = 100, *, rate: float = 1.0, bias: float = 0.5) -> System:
    """Return independent walkers on the integers that step at random times, a continuous-time random walk.

    Every walker steps up at rate `rate * bias` and down at rate `rate * (1 - bias)`, so its mean moves with velocity
    `rate (2 bias - 1)` and its variance grows as `4 rate bias (1 - bias) t`.

    Args:
        n_walkers:
            The number of walkers.

        rate:
            The rate of steps of a walker.

        bias:
            The probability that a step goes up.

    Returns:
        system:
            The model, with the positions `walkers.x`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import random_walk
        >>> random_walk(20).elements[0].size
        20
    """
    return System(elements=[RandomWalker.group('walkers', n_walkers, rate=rate, bias=bias)])


@register_model(
    field='mathematics/stochastic_processes',
    tags=('random walk', 'sde', 'heavy tails', 'power law'),
    references=('Jensen 2023',),
    defaults={'until': 50.0, 'dt': 0.05},
)
def weakening_ornstein_uhlenbeck(
    n_particles: int = 200,
    *,
    k: float = 1.0,
    ell: float = 1.0,
    D: float = 0.25,
) -> System:
    """Return Brownian particles held by a restoring force that weakens with distance.

    The particles follow `dx = -k x / (1 + x²/ℓ²) dt + √(2D) dW`. Near the origin the force is that of an
    Ornstein-Uhlenbeck process, and far from it the force decays as `1 / x`, so the stationary density
    `(1 + x²/ℓ²)^(-k ℓ² / 2D)` has power-law tails.

    Args:
        n_particles:
            The number of independent particles, all started at the origin.

        k:
            The stiffness of the force near the origin.

        ell:
            The distance where the force starts to weaken.

        D:
            The diffusion coefficient.

    Returns:
        system:
            The model, with the positions `particles.x`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import weakening_ornstein_uhlenbeck
        >>> weakening_ornstein_uhlenbeck(10).elements[0].size
        10
    """
    return System(elements=[WeakeningParticle.group('particles', n_particles, k=k, ell=ell, D=D)])


@register_model(
    field='mathematics/stochastic_processes',
    tags=('random walk', 'agents', 'history dependence', 'ageing'),
    references=('Hanel, Thurner, and Gell-Mann 2014', 'Thurner et al. 2018'),
    defaults={'until': 100},
)
def ageing_random_walk(n_walkers: int = 50, *, p: float = 0.5, nu: float = 1.0) -> System:
    """Return ageing random walks, whose up steps become rarer the more of them a walker has taken.

    After `k` up steps a walker steps up with probability `p / (1 + k)^nu`, so the walk is history dependent and,
    for `nu > 0`, its number of up steps grows as `n^(1 / (1 + nu))` after `n` steps.

    Args:
        n_walkers:
            The number of independent walkers, all started at 0.

        p:
            The probability of the first up step.

        nu:
            The exponent of the decay of the up probability.

    Returns:
        system:
            The model, with the group `walkers`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import ageing_random_walk
        >>> ageing_random_walk(5).elements[0].size
        5
    """
    return System(elements=[AgeingWalker.group('walkers', n_walkers, p=p, nu=nu)])
