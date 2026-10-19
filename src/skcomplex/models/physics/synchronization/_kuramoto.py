"""Synchronization models: coupled phase oscillators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from sage.all__sagemath_symbolics import sin

from ....core.base import Domain, parameter
from ....core.model import Element, Interaction, Rate, System, delayed
from ....core.runtime import register_model

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


@register_model(
    field='physics/synchronization',
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
        >>> from skcomplex.models.physics.synchronization import delayed_kuramoto
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
