"""Kinetic and wave transport systems.

The module covers wave-number conservation with breaking, and discrete-velocity gases.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import ConservationInteraction
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SSPRKSolver
from ....core.systems import BaseSystem, System


class CarlemanGasElement(Element):
    """Cells of the Carleman discrete-velocity gas.

    The densities `f_plus` and `f_minus` move at the speeds `c` and `-c`.

    Collisions turn two particles of one velocity into two of the other at the rate `kappa`, so
    `df_plus/dt + c df_plus/dx = kappa (f_minus^2 - f_plus^2)` and the opposite for `f_minus`, which conserves the
    mass and relaxes to the equilibrium `f_plus = f_minus`.

    Args:
        label:
            The label of the gas.

        f_plus:
            The initial density of right-moving particles, one value per cell or one for all.

        f_minus:
            The initial density of left-moving particles, one value per cell or one for all.

        c:
            The particle speed.

        kappa:
            The collision rate.

        space:
            The lattice of the cells, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.physics.fluids import CarlemanGasElement
        >>> CarlemanGasElement('site').kappa
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        f_plus: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        f_minus: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        c: Annotated[float, RealSet.open(0, oo)] = 1.0,
        kappa: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.f_plus = f_plus
        self.f_minus = f_minus
        self.c = c
        self.kappa = kappa


class WaveNumberElement(Element):
    """Cells of the local wave number of deep-water waves.

    The wave number is conserved as `dk/dt + d omega(k)/dx = 0` with `omega = sqrt(g k)`.

    The wave number travels at the group speed `C(k) = sqrt(g / k) / 2`, which falls as the waves shorten, so long
    waves catch up with short ones and the wave number steepens until it breaks.

    Args:
        label:
            The label of the wave train.

        k:
            The initial wave number, one value per cell or one for all.

        g:
            The gravitational acceleration.

        space:
            The lattice of the cells, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.physics.fluids import WaveNumberElement
        >>> WaveNumberElement('site').g
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        k: Annotated[object, RealSet.open(0, oo)] = 1.0,
        g: Annotated[float, RealSet.open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
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

        solver:
            The solver of the problem, or `'auto'` for strong stability preserving Runge-Kutta steps.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.fluids import WaveBreakingSystem
        >>> WaveBreakingSystem(16).prepare().system_.elements[0].space.prepare().size_
        16
    """

    field = 'physics/fluids'
    topics = ('field', 'waves', 'breaking', 'conservation law', 'finite volume', 'pde')
    references = ('Whitham 1974', 'Forrest 2018')
    until = 20.0
    dt = 0.5

    def __init__(
        self,
        n: int = 200,
        *,
        k0: float = 1.0,
        amplitude: float = 0.5,
        g: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n = n
        self.k0 = k0
        self.amplitude = amplitude
        self.g = g

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a modulated wave train on a periodic interval of length 2 pi.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        spacing = 2 * math.pi / self.n
        centers = (np.arange(self.n) + 0.5) * spacing
        space = LatticeSpace(self.n, spacing=spacing, discretization='finite-volume')
        train = WaveNumberElement('train', k=self.k0 * (1 + self.amplitude * np.sin(centers)), g=self.g, space=space)
        self.system_ = System(
            self.label,
            elements=[train],
            interactions=[
                ConservationInteraction(
                    'action',
                    target='train',
                    state='k',
                    flux=lambda s: sqrt(s.target.g * s.target.k),
                ),
            ],
            solver=SSPRKSolver(),
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

        solver:
            The solver of the problem, or `'auto'` for strong stability preserving Runge-Kutta steps.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.fluids import DiscreteVelocityBoltzmannSystem
        >>> DiscreteVelocityBoltzmannSystem(16).prepare().system_.elements[0].space.prepare().size_
        16
    """

    field = 'physics/fluids'
    topics = ('field', 'kinetic theory', 'discrete velocities', 'finite volume', 'pde')
    references = ('Carleman 1957', 'Gatignol 1975', 'Bellomo et al. 2017')
    until = 5.0
    dt = 0.1

    def __init__(
        self,
        n: int = 200,
        *,
        c: float = 1.0,
        kappa: float = 1.0,
        length: float = 10.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a gas on a periodic line that starts from a pulse of right movers.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        spacing = self.length / self.n
        centers = (np.arange(self.n) + 0.5) * spacing
        pulse = np.exp(-((centers - self.length / 2) ** 2))
        space = LatticeSpace(self.n, spacing=spacing, discretization='finite-volume')
        gas = CarlemanGasElement(
            'gas',
            f_plus=0.5 + pulse,
            f_minus=np.full(self.n, 0.5),
            c=self.c,
            kappa=self.kappa,
            space=space,
        )
        self.system_ = System(
            self.label,
            elements=[gas],
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
            solver=SSPRKSolver(),
        )
