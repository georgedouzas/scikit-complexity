"""Kinetic and wave transport systems.

The module covers wave-number conservation with breaking, and discrete-velocity gases.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import ConservationInteraction
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import SSPRKSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class CarlemanGasElement(Element):
    """Site of the Carleman discrete-velocity gas.

    The densities `f_plus` and `f_minus` move at the speeds `c` and `-c`.

    Collisions turn two particles of one velocity into two of the other at the rate `kappa`, so
    `df_plus/dt + c df_plus/dx = kappa (f_minus^2 - f_plus^2)` and the opposite for `f_minus`, which conserves the
    mass and relaxes to the equilibrium `f_plus = f_minus`.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        f_plus:
            The initial density of right-moving particles.

        f_minus:
            The initial density of left-moving particles.

        c:
            The particle speed.

        kappa:
            The collision rate.

    Examples:
        >>> from skcomplex.systems.physics.fluids import CarlemanGasElement
        >>> CarlemanGasElement('site').kappa
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        f_plus: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        f_minus: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.f_plus = f_plus
        self.f_minus = f_minus
        self.c = c
        self.kappa = kappa


class WaveNumberElement(Element):
    """Local wave number of deep-water waves.

    The wave number is conserved as `dk/dt + d omega(k)/dx = 0` with `omega = sqrt(g k)`.

    The wave number travels at the group speed `C(k) = sqrt(g / k) / 2`, which falls as the waves shorten, so long
    waves catch up with short ones and the wave number steepens until it breaks.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        k:
            The initial wave number.

        g:
            The gravitational acceleration.

    Examples:
        >>> from skcomplex.systems.physics.fluids import WaveNumberElement
        >>> WaveNumberElement('site').g
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        k: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.k = k
        self.g = g


class WaveBreakingSystem(BaseSystem):
    """The kinematic wave equation of deep-water wave trains.

    The interval is periodic of length 2 pi.

    The wave number starts as `k0 (1 + amplitude sin x)` and steepens into a jump, a break of the wavelength, at the
    time `1 / max(-C'(k0(x)) k0'(x))` of the first crossing of characteristics. Finite volumes conserve the total number
    of waves. The system has the wave number `train.k`.

    Args:
        n:
            The number of cells.

        k0:
            The mean initial wave number.

        amplitude:
            The relative amplitude of the initial modulation, below 1.

        g:
            The gravitational acceleration.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.fluids import WaveBreakingSystem
        >>> WaveBreakingSystem(16).prepare().system_.elements[0].prepare().size_
        16
    """

    field = 'physics/fluids'
    topics = ('field', 'waves', 'breaking', 'conservation law', 'finite volume', 'pde')
    references = ('Whitham 1974', 'Forrest 2018')
    until = 20.0
    dt = 0.5
    solver = SSPRKSolver()

    def __init__(
        self,
        n: int = 200,
        *,
        k0: float = 1.0,
        amplitude: float = 0.5,
        g: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.k0 = k0
        self.amplitude = amplitude
        self.g = g

    def _build_system(self) -> System:
        """Return the kinematic wave equation of deep-water wave trains.

        The interval is periodic of length 2 pi.
        """
        spacing = 2 * math.pi / self.n
        centers = (np.arange(self.n) + 0.5) * spacing
        values = {'k': self.k0 * (1 + self.amplitude * np.sin(centers)), 'g': self.g}
        space = LatticeSpace(self.n, spacing=spacing)
        return System(
            self.label,
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
                ConservationInteraction(
                    'action',
                    target='train',
                    state='k',
                    flux=lambda s: sqrt(s.target.g * s.target.k),
                ),
            ],
        )


class DiscreteVelocityBoltzmannSystem(BaseSystem):
    """The Carleman discrete-velocity Boltzmann equation on a periodic line.

    The gas starts from a pulse of right movers.

    A pulse of right-moving particles on a uniform background splits into two moving pulses while collisions relax every
    site towards equal densities, and the total mass stays constant. The system has the densities `gas.f_plus` and
    `gas.f_minus`.

    Args:
        n:
            The number of cells.

        c:
            The particle speed.

        kappa:
            The collision rate.

        length:
            The length of the line.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.fluids import DiscreteVelocityBoltzmannSystem
        >>> DiscreteVelocityBoltzmannSystem(16).prepare().system_.elements[0].prepare().size_
        16
    """

    field = 'physics/fluids'
    topics = ('field', 'kinetic theory', 'discrete velocities', 'finite volume', 'pde')
    references = ('Carleman 1957', 'Gatignol 1975', 'Bellomo et al. 2017')
    until = 5.0
    dt = 0.1
    solver = SSPRKSolver()

    def __init__(
        self,
        n: int = 200,
        *,
        c: float = 1.0,
        kappa: float = 1.0,
        length: float = 10.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.c = c
        self.kappa = kappa
        self.length = length

    @staticmethod
    def _get_collisions(s: SimpleNamespace) -> Expression:
        """Return the net rate at which collisions turn left movers into right movers.

        The rate is `kappa (f_minus^2 - f_plus^2)`.
        """
        return s.target.kappa * (s.target.f_minus**2 - s.target.f_plus**2)

    def _build_system(self) -> System:
        """Return the Carleman discrete-velocity Boltzmann equation on a periodic line.

        The gas starts from a pulse of right movers.
        """
        spacing = self.length / self.n
        centers = (np.arange(self.n) + 0.5) * spacing
        pulse = np.exp(-((centers - self.length / 2) ** 2))
        values = {'f_plus': 0.5 + pulse, 'f_minus': np.full(self.n, 0.5), 'c': self.c, 'kappa': self.kappa}
        space = LatticeSpace(self.n, spacing=spacing)
        return System(
            self.label,
            elements=[
                FieldElement(
                    'gas',
                    element=CarlemanGasElement,
                    space=space,
                    values=values,
                    discretization='finite-volume',
                ),
            ],
            interactions=[
                ConservationInteraction(
                    'right_movers',
                    target='gas',
                    state='f_plus',
                    flux=lambda s: s.target.c * s.target.f_plus,
                    source=self._get_collisions,
                ),
                ConservationInteraction(
                    'left_movers',
                    target='gas',
                    state='f_minus',
                    flux=lambda s: -s.target.c * s.target.f_minus,
                    source=lambda s: -self._get_collisions(s),
                ),
            ],
        )
