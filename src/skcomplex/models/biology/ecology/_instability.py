"""Ecology models: Jensen's coupled populations that lose the stability of their steady state from within."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import exp, sqrt

from ....core.base import Domain, Expression, parameter, state
from ....core.model import Element, Law, Symbols, System, Update, bernoulli, normal, where
from ....core.runtime import register_model


def _nonnegative(value: Expression) -> Expression:
    """Return a population size, cut at 0 from below."""
    return where(value > 0, value, 0)


class UnstablePopulations(Element):
    """Resident population `x` and invading population `y`, after Eqs. (13.3) and (13.4) of Jensen.

    At every step `x ← [x - eta (x - X)] exp(-alpha y) + √x sigma chi`, an Ornstein-Uhlenbeck step damped by the
    invader. With probability `p` a mutation from `x` adds `nu x` to `y`, and otherwise
    `y ← y (1 + beta) + √y sigma chi'`. The noises `chi` and `chi'` are independent standard normal values, scaled by
    the square root of the population sizes as in birth and death processes, and both sizes are cut at 0.

    Args:
        label:
            The label of the populations.

        x:
            The initial resident population.

        y:
            The initial invading population.

        eta:
            The restoring rate `eta` of the resident population.

        mean:
            The level `X` the resident population returns to.

        alpha:
            The suppression `alpha` of the resident population by the invader.

        beta:
            The growth rate `beta` of the invader.

        nu:
            The share `nu` of the resident population that a mutation moves to the invader.

        p:
            The probability of a mutation per step.

        sigma:
            The amplitude `sigma` of the noise.

    Examples:
        >>> from skcomplex.models.biology.ecology import UnstablePopulations
        >>> UnstablePopulations('populations').p
        0.001
    """

    x: float = state(default=1.1, domain=Domain.NONNEGATIVE)
    y: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    eta: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    mean: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    alpha: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    nu: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    p: float = parameter(default=0.001, domain=Domain.NONNEGATIVE)
    sigma: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the steps of the two populations.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        resident = (s.x - s.eta * (s.x - s.mean)) * exp(-s.alpha * s.y) + sqrt(s.x) * s.sigma * normal('chi_x')
        grown = s.y * (1 + s.beta) + sqrt(s.y) * s.sigma * normal('chi_y')
        invader = where(bernoulli('mutation', s.p) > 0, s.y + s.nu * s.x, grown)
        return [Update(s.x, _nonnegative(resident)), Update(s.y, _nonnegative(invader))]


@register_model(
    field='biology/ecology',
    tags=('tipping point', 'stochastic', 'discrete map', 'early warning'),
    references=('Jensen 2023',),
    defaults={'until': 2000, 'dt': 10},
)
def intrinsic_instability(
    *,
    x: float = 1.1,
    eta: float = 0.1,
    alpha: float = 0.01,
    beta: float = 0.01,
    nu: float = 0.01,
    p: float = 0.001,
    sigma: float = 0.01,
) -> System:
    """Return Jensen's coupled populations, whose steady state becomes unstable without a change of the surroundings.

    The resident population fluctuates about its level `X = 1` until a rare mutation seeds the invader. The invader
    then grows by the factor `1 + beta` per step and suppresses the resident, so the trajectory leaves the fixed point
    `(X, 0)` along its unstable direction, with eigenvalue `beta`, and the resident declines although no parameter has
    changed.

    Args:
        x:
            The initial resident population.

        eta:
            The restoring rate of the resident population.

        alpha:
            The suppression of the resident population by the invader.

        beta:
            The growth rate of the invader.

        nu:
            The share of the resident population that a mutation moves to the invader.

        p:
            The probability of a mutation per step.

        sigma:
            The amplitude of the noise.

    Returns:
        system:
            The model, with the populations `populations.x` and `populations.y`.

    Examples:
        >>> from skcomplex.models.biology.ecology import intrinsic_instability
        >>> intrinsic_instability().elements[0].beta
        0.01
    """
    populations = UnstablePopulations(
        'populations',
        x=x,
        eta=eta,
        alpha=alpha,
        beta=beta,
        nu=nu,
        p=p,
        sigma=sigma,
    )
    return System(components=[populations])
