"""Kinetic and wave transport models: wave-number conservation with breaking, and discrete-velocity gases."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, sqrt

from ....core.elements import FieldElement
from ....core.interactions import ConservationInteraction
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.solvers import SSPRKSolver
from ..._catalog import register_model
from ._carleman_gas import CarlemanGasElement
from ._wave_number import WaveNumberElement


def _get_collisions(s: SimpleNamespace) -> Expression:
    """Return the net rate `kappa (f_minus^2 - f_plus^2)` at which collisions turn left movers into right movers."""
    return s.target.kappa * (s.target.f_minus**2 - s.target.f_plus**2)


@register_model(
    field='physics/fluids',
    tags=('field', 'waves', 'breaking', 'conservation law', 'finite volume', 'pde'),
    references=('Whitham 1974', 'Forrest 2018'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': SSPRKSolver()},
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
        >>> wave_breaking(16).elements[0].prepare().size_
        16
    """
    spacing = 2 * math.pi / n
    centers = (np.arange(n) + 0.5) * spacing
    values = {'k': k0 * (1 + amplitude * np.sin(centers)), 'g': g}
    space = LatticeSpace(n, spacing=spacing)
    return System(
        elements=[
            FieldElement(
                'train',
                element=WaveNumberElement,
                space=space,
                values=values,
                discretization='finite-volume',
            ),
        ],
        interactions=[
            ConservationInteraction('action', target='train', state='k', flux=lambda s: sqrt(s.target.g * s.target.k)),
        ],
    )


@register_model(
    field='physics/fluids',
    tags=('field', 'kinetic theory', 'discrete velocities', 'finite volume', 'pde'),
    references=('Carleman 1957', 'Gatignol 1975', 'Bellomo et al. 2017'),
    defaults={'until': 5.0, 'dt': 0.1, 'solver': SSPRKSolver()},
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
        >>> discrete_velocity_boltzmann(16).elements[0].prepare().size_
        16
    """
    spacing = length / n
    centers = (np.arange(n) + 0.5) * spacing
    pulse = np.exp(-((centers - length / 2) ** 2))
    values = {'f_plus': 0.5 + pulse, 'f_minus': np.full(n, 0.5), 'c': c, 'kappa': kappa}
    space = LatticeSpace(n, spacing=spacing)
    return System(
        elements=[
            FieldElement('gas', element=CarlemanGasElement, space=space, values=values, discretization='finite-volume'),
        ],
        interactions=[
            ConservationInteraction(
                'right_movers',
                target='gas',
                state='f_plus',
                flux=lambda s: s.target.c * s.target.f_plus,
                source=_get_collisions,
            ),
            ConservationInteraction(
                'left_movers',
                target='gas',
                state='f_minus',
                flux=lambda s: -s.target.c * s.target.f_minus,
                source=lambda s: -_get_collisions(s),
            ),
        ],
    )
