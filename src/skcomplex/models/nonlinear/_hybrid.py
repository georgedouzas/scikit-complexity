"""Model delayed and noisy nonlinear dynamics: Mackey-Glass, delayed Kuramoto, and stochastic resonance."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from sage.all__sagemath_symbolics import cos, sin, sqrt

from ...core.base import Domain, parameter, state
from ...core.model import Element, Interaction, Noise, Rate, System, delayed
from ...core.runtime import register_model

MackeyGlass = Element.define(
    'MackeyGlass',
    states={'x': state(default=0.5, domain=Domain.NONNEGATIVE)},
    parameters={'beta': 0.2, 'gamma': 0.1, 'n': 10.0, 'tau': parameter(default=17.0, domain=Domain.POSITIVE)},
    laws=lambda s: [Rate(s.x, s.beta * delayed(s.x, s.tau) / (1 + delayed(s.x, s.tau) ** s.n) - s.gamma * s.x)],
)

PhaseOscillator = Element.define(
    'PhaseOscillator',
    states={'theta': 0.0},
    parameters={'omega': 1.0},
    laws=lambda s: [Rate(s.theta, s.omega)],
)

DelayedCoupling = Interaction.define(
    'DelayedCoupling',
    roles=['source', 'target'],
    parameters={'K': 1.0, 'tau': parameter(default=1.0, domain=Domain.POSITIVE)},
    laws=lambda s: [Rate(s.target.theta, s.K * sin(delayed(s.source.theta, s.tau) - s.target.theta))],
)

DoubleWell = Element.define(
    'DoubleWell',
    states={'x': -1.0},
    parameters={
        'amplitude': 0.1,
        'frequency': 0.01,
        'D': parameter(default=0.12, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Rate(s.x, s.x - s.x**3 + s.amplitude * cos(2 * math.pi * s.frequency * s.t)),
        Noise(s.x, sqrt(2 * s.D)),
    ],
)


@register_model(
    field='nonlinear dynamics',
    tags=('chaos', 'delay', 'physiology'),
    references=('Mackey and Glass 1977', 'Erneux 2009'),
    defaults={'until': 500.0, 'dt': 0.5},
)
def mackey_glass(
    x: float = 0.5,
    *,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: float = 10.0,
    tau: float = 17.0,
) -> System:
    """Return the Mackey-Glass equation `x' = beta x(t - tau) / (1 + x(t - tau)^n) - gamma x` of blood cell production.

    It is periodic for `tau = 6` and chaotic for `tau = 17`. The history before the start is the initial value.

    Args:
        x:
            The initial concentration, also its history.

        beta:
            The production rate.

        gamma:
            The decay rate.

        n:
            The steepness of the production feedback.

        tau:
            The delay of production.

    Returns:
        system:
            The model, with the concentration `cells.x`.

    Examples:
        >>> from skcomplex.models.nonlinear import mackey_glass
        >>> mackey_glass(tau=6.0).elements[0].tau
        6.0
    """
    return System(elements=[MackeyGlass('cells', x=x, beta=beta, gamma=gamma, n=n, tau=tau)])


@register_model(
    field='nonlinear dynamics',
    tags=('synchronization', 'delay', 'oscillators'),
    references=('Yeung and Strogatz 1999', 'Kuramoto 1984'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def delayed_kuramoto(
    n_oscillators: int = 5,
    *,
    K: float = 2.0,
    tau: float = 1.0,
    spread: float = 0.2,
    seed: int = 0,
) -> System:
    """Return Kuramoto phase oscillators coupled all to all through signals that arrive after a delay.

    Oscillator `i` follows `theta_i' = omega_i + (K / N) sum_j sin(theta_j(t - tau) - theta_i)`, with natural
    frequencies spread evenly around 1 and initial phases spread around the circle.

    Args:
        n_oscillators:
            The number of oscillators.

        K:
            The coupling strength.

        tau:
            The delay of the coupling.

        spread:
            The width of the interval of natural frequencies around 1.

        seed:
            The seed of the initial phases.

    Returns:
        system:
            The model, with the phases `oscillator_<i>.theta`.

    Examples:
        >>> from skcomplex.models.nonlinear import delayed_kuramoto
        >>> len(delayed_kuramoto(n_oscillators=3).interactions)
        6
    """
    phases = [2 * math.pi * ((seed * 0.618 + index / n_oscillators) % 1.0) for index in range(n_oscillators)]
    omegas = [1.0 + spread * (index / max(n_oscillators - 1, 1) - 0.5) for index in range(n_oscillators)]
    elements = [
        PhaseOscillator(f'oscillator_{index}', theta=phases[index], omega=omegas[index])
        for index in range(n_oscillators)
    ]
    interactions = [
        DelayedCoupling(
            f'coupling_{source}_{target}',
            source=f'oscillator_{source}',
            target=f'oscillator_{target}',
            K=K / n_oscillators,
            tau=tau,
        )
        for source in range(n_oscillators)
        for target in range(n_oscillators)
        if source != target
    ]
    return System(elements=elements, interactions=interactions)


@register_model(
    field='nonlinear dynamics',
    tags=('noise', 'sde', 'forcing', 'resonance'),
    references=('Benzi, Sutera, and Vulpiani 1981', 'Gammaitoni et al. 1998'),
    defaults={'until': 500.0, 'dt': 0.5},
)
def stochastic_resonance(
    x: float = -1.0,
    *,
    amplitude: float = 0.1,
    frequency: float = 0.01,
    D: float = 0.12,
) -> System:
    """Return a particle in a double well, `dx = (x - x^3 + A cos(2 pi f t)) dt + sqrt(2 D) dW`.

    The forcing alone is too weak to move the particle between the wells, but noise of the right intensity makes the
    hops follow the forcing.

    Args:
        x:
            The initial position.

        amplitude:
            The amplitude of the periodic forcing.

        frequency:
            The frequency of the forcing.

        D:
            The noise intensity.

    Returns:
        system:
            The model, with the position `particle.x`.

    Examples:
        >>> from skcomplex.models.nonlinear import stochastic_resonance
        >>> stochastic_resonance(D=0.2).elements[0].D
        0.2
    """
    return System(elements=[DoubleWell('particle', x=x, amplitude=amplitude, frequency=frequency, D=D)])
