"""Intermittency: bursts of a variable driven across a bifurcation by multiplicative noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import sqrt

from ....core.base import Domain, parameter, state
from ....core.model import Element, Noise, Rate, System
from ....core.runtime import register_model
from ....core.solvers import Heun

IntermittentVariable = Element.define(
    'IntermittentVariable',
    states={'x': state(default=1.0, domain=Domain.POSITIVE)},
    parameters={'epsilon': 0.0, 'D': parameter(default=0.5, domain=Domain.POSITIVE)},
    laws=lambda s: [Rate(s.x, s.epsilon * s.x - s.x**3), Noise(s.x, sqrt(2 * s.D) * s.x)],
)


@register_model(
    field='mathematics/stochastic_processes',
    tags=('noise', 'sde', 'multiplicative noise', 'intermittency', 'bifurcation'),
    references=('Bertin 2021', 'Platt, Spiegel, and Tresser 1993', 'Heagy, Platt, and Hammel 1994'),
    defaults={'until': 500.0, 'dt': 0.1, 'solver': Heun(dt=0.01)},
)
def on_off_intermittency(
    n_realizations: int = 20,
    *,
    epsilon: float = 0.0,
    D: float = 0.5,
    x: float = 1.0,
) -> System:
    """Return stochastic on-off intermittency, `dx/dt = (epsilon + xi) x - x^3` with Stratonovich white noise `xi`.

    The noise has the correlation `<xi(t) xi(t')> = 2 D delta(t - t')`. The variable `y = ln x` diffuses freely while
    `x` is small and is pushed back by `-exp(2 y)` once `x` grows, so near the bifurcation `epsilon = 0` the variable
    rests close to 0 for long laminar phases, whose durations are distributed as `tau^(-3/2)`, between bursts where
    `x` is of order 1. For `epsilon > 0` the power law is cut off beyond durations of order `D / epsilon^2`.

    Args:
        n_realizations:
            The number of independent realizations.

        epsilon:
            The control parameter, the distance from the bifurcation.

        D:
            The noise intensity.

        x:
            The initial value of every realization.

    Returns:
        system:
            The model, with the values `variables.x`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import on_off_intermittency
        >>> on_off_intermittency(5).noise
        'stratonovich'
    """
    variables = IntermittentVariable.group('variables', n_realizations, x=x, epsilon=epsilon, D=D)
    return System(elements=[variables], noise='stratonovich')
