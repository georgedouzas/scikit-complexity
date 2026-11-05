"""Climate variability: stochastic climate and the delayed oscillator of El Niño."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.components.interactions import RelaxationInteraction
from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver
from ..._catalog import register_system
from ._climate_anomaly import ClimateAnomalyElement
from ._delayed_oscillator import DelayedOscillatorElement


def _get_damping_time(s: SimpleNamespace) -> Expression:
    """Return the damping time `1 / lam` of the temperature anomaly."""
    return 1 / s.target.lam


def _get_weather(s: SimpleNamespace) -> Expression:
    """Return the amplitude `sigma` of the weather noise."""
    return s.target.sigma


@register_system(
    field='physics/climate',
    tags=('sde', 'Ornstein-Uhlenbeck', 'red noise', 'stochastic climate'),
    references=('Hasselmann 1976', 'Ghil and Lucarini 2020'),
    defaults={'until': 100.0, 'dt': 0.1, 'solver': EulerMaruyamaSolver(dt=0.01)},
)
def hasselmann(T: float = 0.0, *, lam: float = 0.5, sigma: float = 0.3) -> System:
    """Return Hasselmann's stochastic climate, where fast weather noise drives a slow, damped temperature anomaly.

    The anomaly follows `dT = -lam T dt + sigma dW`, an Ornstein-Uhlenbeck process whose red spectrum integrates white
    weather noise, with the stationary variance `sigma^2 / (2 lam)`.

    Args:
        T:
            The initial temperature anomaly.

        lam:
            The feedback rate that damps the anomaly.

        sigma:
            The amplitude of the weather noise.

    Returns:
        system:
            The system, with the anomaly `ocean.T`.

    Examples:
        >>> from skcomplex.systems.physics.climate import hasselmann
        >>> hasselmann(lam=0.1).elements[0].lam
        0.1
    """
    feedback = RelaxationInteraction(
        'feedback',
        target='ocean',
        state='T',
        goal=0.0,
        time=_get_damping_time,
        noise=_get_weather,
    )
    return System(elements=[ClimateAnomalyElement('ocean', T=T, lam=lam, sigma=sigma)], interactions=[feedback])


@register_system(
    field='physics/climate',
    tags=('delay', 'oscillation', 'El Niño', 'ENSO'),
    references=('Suarez and Schopf 1988', 'Battisti and Hirst 1989', 'Ghil and Lucarini 2020'),
    defaults={'until': 200.0, 'dt': 0.5},
)
def enso(T: float = 0.1, *, alpha: float = 0.75, delta: float = 6.0) -> System:
    """Return the delayed oscillator of El Niño of Suarez and Schopf, `dT/dt = T - T^3 - alpha T(t - delta)`.

    The sea surface temperature anomaly of the eastern Pacific grows by local feedback and is damped by the delayed
    return of oceanic waves reflected at the western boundary. For `alpha < 1` and a long enough delay the warm and cold
    states lose stability to an oscillation between El Niño and La Niña. The history before the start is the initial
    anomaly.

    Args:
        T:
            The initial temperature anomaly, also its history.

        alpha:
            The strength of the delayed negative feedback.

        delta:
            The delay of the wave transit.

    Returns:
        system:
            The system, with the anomaly `pacific.T`.

    Examples:
        >>> from skcomplex.systems.physics.climate import enso
        >>> enso(delta=4.0).elements[0].delta
        4.0
    """
    return System(elements=[DelayedOscillatorElement('pacific', T=T, alpha=alpha, delta=delta)])
