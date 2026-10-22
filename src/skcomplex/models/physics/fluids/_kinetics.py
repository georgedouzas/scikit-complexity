"""Kinetic and wave transport models: wave-number conservation with breaking, and discrete-velocity gases."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from sage.all__sagemath_symbolics import sqrt

from ....core.base import Domain, parameter, state
from ....core.model import Element, Field, Flux, LatticeSpace, Law, Rate, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import SSPRK


class WaveNumber(Element):
    """Local wave number of deep-water waves, conserved as `dk/dt + d omega(k)/dx = 0` with `omega = sqrt(g k)`.

    The wave number travels at the group speed `C(k) = sqrt(g / k) / 2`, which falls as the waves shorten, so long
    waves catch up with short ones and the wave number steepens until it breaks.

    Args:
        label:
            The label of the site or the field.

        k:
            The initial wave number.

        g:
            The gravitational acceleration.

    Examples:
        >>> from skcomplex.models.physics.fluids import WaveNumber
        >>> WaveNumber('site').g
        1.0
    """

    k: float = state(default=1.0, domain=Domain.POSITIVE)
    g: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the conservation of waves, with the frequency as the flux of the wave number.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Flux(s.k, sqrt(s.g * s.k))]


class CarlemanGas(Element):
    """Site of the Carleman discrete-velocity gas, densities `f_plus` and `f_minus` moving at the speeds `c` and `-c`.

    Collisions turn two particles of one velocity into two of the other at the rate `kappa`, so
    `df_plus/dt + c df_plus/dx = kappa (f_minus^2 - f_plus^2)` and the opposite for `f_minus`, which conserves the
    mass and relaxes to the equilibrium `f_plus = f_minus`.

    Args:
        label:
            The label of the site or the field.

        f_plus:
            The initial density of right-moving particles.

        f_minus:
            The initial density of left-moving particles.

        c:
            The particle speed.

        kappa:
            The collision rate.

    Examples:
        >>> from skcomplex.models.physics.fluids import CarlemanGas
        >>> CarlemanGas('site').kappa
        1.0
    """

    f_plus: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    f_minus: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    c: float = parameter(default=1.0, domain=Domain.POSITIVE)
    kappa: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the transport of both densities and their collisions.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        collisions = s.kappa * (s.f_minus**2 - s.f_plus**2)
        return [
            Flux(s.f_plus, s.c * s.f_plus),
            Flux(s.f_minus, -s.c * s.f_minus),
            Rate(s.f_plus, collisions),
            Rate(s.f_minus, -collisions),
        ]


@register_model(
    field='physics/fluids',
    tags=('field', 'waves', 'breaking', 'conservation law', 'finite volume', 'pde'),
    references=('Whitham 1974', 'Forrest 2018'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': SSPRK()},
)
def wave_breaking(n: int = 200, *, k0: float = 1.0, amplitude: float = 0.5, g: float = 1.0) -> System:
    """Return the kinematic wave equation of deep-water wave trains on a periodic interval of length 2 pi.

    The wave number starts as `k0 (1 + amplitude sin x)` and steepens into a jump, a break of the wavelength, at the
    time `1 / max(-C'(k0(x)) k0'(x))` of the first crossing of characteristics. Finite volumes conserve the total
    number of waves.

    Args:
        n:
            The number of cells.

        k0:
            The mean initial wave number.

        amplitude:
            The relative amplitude of the initial modulation, below 1.

        g:
            The gravitational acceleration.

    Returns:
        system:
            The model, with the wave number `train.k`.

    Examples:
        >>> from skcomplex.models.physics.fluids import wave_breaking
        >>> wave_breaking(16).elements[0].size
        16
    """
    spacing = 2 * math.pi / n
    centers = (np.arange(n) + 0.5) * spacing
    values = {'k': k0 * (1 + amplitude * np.sin(centers)), 'g': g}
    space = LatticeSpace(n, spacing=spacing)
    return System(components=[Field('train', WaveNumber, space, values=values, discretization='finite-volume')])


@register_model(
    field='physics/fluids',
    tags=('field', 'kinetic theory', 'discrete velocities', 'finite volume', 'pde'),
    references=('Carleman 1957', 'Gatignol 1975', 'Bellomo et al. 2017'),
    defaults={'until': 5.0, 'dt': 0.1, 'solver': SSPRK()},
)
def discrete_velocity_boltzmann(
    n: int = 200,
    *,
    c: float = 1.0,
    kappa: float = 1.0,
    length: float = 10.0,
) -> System:
    """Return the Carleman discrete-velocity Boltzmann equation on a periodic line, from a pulse of right movers.

    A pulse of right-moving particles on a uniform background splits into two moving pulses while collisions relax
    every site towards equal densities, and the total mass stays constant.

    Args:
        n:
            The number of cells.

        c:
            The particle speed.

        kappa:
            The collision rate.

        length:
            The length of the line.

    Returns:
        system:
            The model, with the densities `gas.f_plus` and `gas.f_minus`.

    Examples:
        >>> from skcomplex.models.physics.fluids import discrete_velocity_boltzmann
        >>> discrete_velocity_boltzmann(16).elements[0].size
        16
    """
    spacing = length / n
    centers = (np.arange(n) + 0.5) * spacing
    pulse = np.exp(-((centers - length / 2) ** 2))
    values = {'f_plus': 0.5 + pulse, 'f_minus': np.full(n, 0.5), 'c': c, 'kappa': kappa}
    space = LatticeSpace(n, spacing=spacing)
    return System(components=[Field('gas', CarlemanGas, space, values=values, discretization='finite-volume')])
