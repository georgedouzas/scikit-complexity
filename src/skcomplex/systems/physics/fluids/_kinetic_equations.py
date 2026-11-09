"""Kinetic equations of gases and plasmas over position and velocity.

The equations live on the phase spaces of the core.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import FieldElement
from ....core.components.interactions import AdvectionInteraction, EncounterInteraction
from ....core.components.laws.utils import coordinate, gradient, inverse_laplacian, maximum_entropy, moment
from ....core.components.spaces import LatticeSpace, PhaseSpace
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import RK4Solver, SemiLagrangianSolver, SSPRKSolver, StrangSplittingSolver
from ....core.errors import ValidationError
from ....core.specs import AxisSpec, RoleSpec, VisualSpec
from ..common.fields import PhaseSpaceDensityElement
from ..common.particles import RelaxationTimeInteraction


class BhatnagarGrossKrookInteraction(RelaxationTimeInteraction):
    """Relaxation of Bhatnagar, Gross, and Krook of a density.

    The density relaxes towards the maximum-entropy density with its moments.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the field on a phase space.

        state:
            The name of the state of the density.

        axes:
            The names of the velocity axes.

        time:
            A function of the namespace that returns the relaxation time, or a positive number.

    Examples:
        >>> from skcomplex.systems.physics.fluids import BhatnagarGrossKrookInteraction
        >>> BhatnagarGrossKrookInteraction('bgk', target='gas', axes=('v',), time=0.1).axes
        ('v',)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'x',
        axes: Sequence[str] = ('v',),
        time: Callable[[SimpleNamespace], object] | float = 1.0,
    ) -> None:
        super().__init__(label, target=target, state=state, goal=0.0, time=time)
        self.axes = axes

    def _get_goal(self, namespace: SimpleNamespace) -> object:
        """Return the maximum-entropy density with the moments of the state of the target."""
        return maximum_entropy(getattr(namespace.target, self.state), axes=self.axes)


SETUPS = ('landau', 'two-stream')
SHOCK_TUBE = 'sod'
VELOCITY_AXES = ('vx', 'vy')
VLASOV_VISUAL = VisualSpec('image', {'moments': (0,)})


def _get_plane_streaming(_: SimpleNamespace) -> list[Expression]:
    """Return the drift of free streaming over one position and two velocity axes."""
    return [coordinate('vx'), 0, 0]


def _get_uniform_gas(positions: int, velocities: int, knudsen: float, half_width: float) -> tuple[FieldElement, float]:
    """Return a gas over one position and two velocity axes, and its collision rate.

    The gas has a density wave, and the rate is its collision frequency.
    """
    length = 2 * math.pi
    axes = [AxisSpec(name, velocities, -half_width, half_width) for name in VELOCITY_AXES]
    space = PhaseSpace(LatticeSpace((positions,), spacing=length / positions), axes).prepare()
    vx, vy = np.meshgrid(space.coordinates_['vx'], space.coordinates_['vy'], indexing='ij')
    x = np.arange(positions) * length / positions
    profile = np.exp(-(vx**2 + vy**2) / 2) / (2 * np.pi)
    values = np.multiply.outer(1.0 + 0.2 * np.sin(x), profile)
    gas = FieldElement(
        'gas',
        element=PhaseSpaceDensityElement,
        space=space,
        values={'x': values},
        discretization='finite-volume',
    )
    return gas, 1.0 / knudsen


def _get_maxwell_molecules(_speed: object, _cosine: object) -> float:
    """Return the cross-section of Maxwell molecules in two velocity dimensions."""
    return 1 / (2 * math.pi)


class BoltzmannEquationSystem(BaseSystem):
    """The Boltzmann equation of Maxwell molecules.

    The equation is over one position and two velocity axes.

    Free streaming moves a density wave along the periodic position axis, and binary collisions redistribute the density
    over the velocities at every position, by the fast spectral method. The system has the density `gas.x`.

    Args:
        positions:
            The number of positions.

        velocities:
            The number of points of each velocity axis.

        knudsen:
            The Knudsen number, the inverse of the collision frequency.

        half_width:
            The half-width of each velocity axis.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.fluids import BoltzmannEquationSystem
        >>> BoltzmannEquationSystem(4, 8).prepare().system_.elements[0].prepare().size_
        256
    """

    field = 'physics/fluids'
    topics = ('field', 'kinetic', 'phase space', 'collisions', 'pde')
    references = ('Bellomo et al. 2017', 'Mouhot and Pareschi 2006')
    until = 0.5
    dt = 0.1
    solver = StrangSplittingSolver(SemiLagrangianSolver(0.1), RK4Solver(0.05), 0.1)

    def __init__(
        self,
        positions: int = 8,
        velocities: int = 12,
        *,
        knudsen: float = 1.0,
        half_width: float = 6.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.positions = positions
        self.velocities = velocities
        self.knudsen = knudsen
        self.half_width = half_width

    def _build_system(self) -> System:
        """Return the Boltzmann equation of Maxwell molecules.

        The equation is over one position and two velocity axes.
        """
        gas, frequency = _get_uniform_gas(self.positions, self.velocities, self.knudsen, self.half_width)
        stream = AdvectionInteraction('stream', target='gas', state='x', drift=_get_plane_streaming)
        collide = EncounterInteraction(
            'collide',
            candidate='gas',
            cross_section=_get_maxwell_molecules,
            factor=lambda _: frequency,
        )
        return System(self.label, elements=[gas], interactions=[stream, collide])


class EnskogSystem(BaseSystem):
    """The local Enskog equation.

    It is the Boltzmann equation with the contact value of the pair correlation of disks.

    The collision frequency is multiplied at every position by Henderson's contact value `(1 - 7 c / 16) / (1 - c)^2` of
    the area fraction `c = pi n d^2 / 4`, with `n` the density moment and `d` the diameter of the disks. The collision
    partners sit at one position, so the shift of one diameter of the full Enskog collision is left out. The system has
    the density `gas.x`.

    Args:
        positions:
            The number of positions.

        velocities:
            The number of points of each velocity axis.

        knudsen:
            The Knudsen number, the inverse of the collision frequency of the dilute gas.

        diameter:
            The diameter of the disks.

        half_width:
            The half-width of each velocity axis.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.fluids import EnskogSystem
        >>> EnskogSystem(4, 8).prepare().system_.elements[0].prepare().size_
        256
    """

    field = 'physics/fluids'
    topics = ('field', 'kinetic', 'phase space', 'collisions', 'pde')
    references = ('Bellomo et al. 2017', 'Enskog 1922')
    until = 0.5
    dt = 0.1
    solver = StrangSplittingSolver(SemiLagrangianSolver(0.1), RK4Solver(0.05), 0.1)

    def __init__(
        self,
        positions: int = 8,
        velocities: int = 12,
        *,
        knudsen: float = 1.0,
        diameter: float = 0.2,
        half_width: float = 6.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.positions = positions
        self.velocities = velocities
        self.knudsen = knudsen
        self.diameter = diameter
        self.half_width = half_width

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        gas, frequency = _get_uniform_gas(self.positions, self.velocities, self.knudsen, self.half_width)
        area = math.pi * self.diameter**2 / 4

        def contact(s: SimpleNamespace) -> Expression:
            fraction = area * moment(s.candidate.x, 0, axes=VELOCITY_AXES)
            return frequency * (1 - 7 * fraction / 16) / (1 - fraction) ** 2

        stream = AdvectionInteraction('stream', target='gas', state='x', drift=_get_plane_streaming)
        collide = EncounterInteraction('collide', candidate='gas', cross_section=_get_maxwell_molecules, factor=contact)
        return System(self.label, elements=[gas], interactions=[stream, collide])


class BGKEquationSystem(BaseSystem):
    """The BGK equation of the Sod shock tube over one position and one velocity axis.

    The density relaxes at every position towards the maximum-entropy density with its density, momentum, and energy,
    and the gas starts from the two states of the shock tube of Sod on either side of the middle of a unit interval. As
    the Knudsen number falls the density moment tends to the Euler solution with the adiabatic index 3 of one velocity
    dimension. The system has the density `gas.x`.

    Args:
        positions:
            The number of positions.

        velocities:
            The number of points of the velocity axis.

        knudsen:
            The Knudsen number, the relaxation time.

        half_width:
            The half-width of the velocity axis.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.fluids import BGKEquationSystem
        >>> BGKEquationSystem(8, 16).prepare().system_.elements[0].prepare().size_
        128
    """

    field = 'physics/fluids'
    topics = ('field', 'kinetic', 'phase space', 'shocks', 'pde')
    references = ('Bhatnagar, Gross, and Krook 1954', 'Mieussens 2000', 'Sod 1978')
    until = 0.2
    dt = 0.05
    solver = StrangSplittingSolver(SSPRKSolver(), RK4Solver(0.005), 0.005)

    def __init__(
        self,
        positions: int = 100,
        velocities: int = 32,
        *,
        knudsen: float = 1e-2,
        half_width: float = 6.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.positions = positions
        self.velocities = velocities
        self.knudsen = knudsen
        self.half_width = half_width

    @staticmethod
    def _get_maxwellian(
        density: npt.NDArray[np.float64],
        mean: npt.NDArray[np.float64],
        temperature: npt.NDArray[np.float64],
        speeds: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.float64]:
        """Return the Maxwellian of densities, mean velocities, and temperatures.

        The Maxwellian is over a velocity axis, a row each.
        """
        rows = (speeds[None, :] - mean[:, None]) ** 2 / (2 * temperature[:, None])
        return density[:, None] * np.exp(-rows) / np.sqrt(2 * np.pi * temperature[:, None])

    @staticmethod
    def _get_streaming(_: SimpleNamespace) -> list[Expression]:
        """Return the drift of free streaming over one position and one velocity axis."""
        return [coordinate('v'), 0]

    def _build_system(self) -> System:
        """Return the BGK equation of the Sod shock tube.

        The equation is over one position and one velocity axis.
        """
        position = LatticeSpace((self.positions,), spacing=1.0 / self.positions, boundary='reflecting')
        space = PhaseSpace(position, [AxisSpec('v', self.velocities, -self.half_width, self.half_width)]).prepare()
        left = np.arange(self.positions) < self.positions // 2
        density = np.where(left, 1.0, 0.125)
        temperature = np.where(left, 1.0, 0.8)
        values = self._get_maxwellian(density, np.zeros(self.positions), temperature, space.coordinates_['v'])
        gas = FieldElement(
            'gas',
            element=PhaseSpaceDensityElement,
            space=space,
            values={'x': values},
            discretization='finite-volume',
            visual=VLASOV_VISUAL,
        )
        stream = AdvectionInteraction('stream', target='gas', state='x', drift=self._get_streaming)
        relax = BhatnagarGrossKrookInteraction('relax', target='gas', axes=('v',), time=self.knudsen)
        return System(self.label, elements=[gas], interactions=[stream, relax])


class VlasovEquationSystem(BaseSystem):
    """The Vlasov-Poisson equation of electrons over a neutralizing background.

    The position axis is periodic.

    The electric field is solved from the density moment at every step. The set-up `'landau'` starts from a Maxwellian
    with a density wave of wave number 0.5, whose electric energy decays at twice the Landau damping rate 0.1533. The
    set-up `'two-stream'` starts from two beams of temperature 0.25 at the velocities -1.5 and 1.5, which are unstable
    at the wave number 0.5. The system has the density `gas.x`.

    Args:
        positions:
            The number of positions.

        velocities:
            The number of points of the velocity axis.

        setup:
            `'landau'` or `'two-stream'`.

        amplitude:
            The relative amplitude of the initial density wave.

        half_width:
            The half-width of the velocity axis.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The set-up is unknown.

    Examples:
        >>> from skcomplex.systems.physics.fluids import VlasovEquationSystem
        >>> VlasovEquationSystem(8, 16).prepare().system_.elements[0].prepare().size_
        128
    """

    field = 'physics/fluids'
    topics = ('field', 'kinetic', 'phase space', 'plasma', 'pde')
    references = ('Bellomo et al. 2017', 'Cheng and Knorr 1976')
    until = 30.0
    dt = 0.25
    solver = SemiLagrangianSolver(0.25)

    def __init__(
        self,
        positions: int = 32,
        velocities: int = 64,
        *,
        setup: str = 'landau',
        amplitude: float = 0.01,
        half_width: float = 6.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.positions = positions
        self.velocities = velocities
        self.setup = setup
        self.amplitude = amplitude
        self.half_width = half_width

    @staticmethod
    def _get_poisson_drift(s: SimpleNamespace) -> list[Expression]:
        """Return the velocity along position and the potential force along velocity.

        The potential is that of the density.
        """
        density = moment(s.target.x, 0, axes=('v',))
        return [coordinate('v'), -gradient(inverse_laplacian(1 - density), 1)[0]]

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        if self.setup not in SETUPS:
            error_msg = f'Unknown set-up {self.setup!r} of the Vlasov equation. Use one of {", ".join(SETUPS)}.'
            raise ValidationError(error_msg)
        length = 4 * math.pi
        space = PhaseSpace(
            LatticeSpace((self.positions,), spacing=length / self.positions),
            [AxisSpec('v', self.velocities, -self.half_width, self.half_width)],
        )
        space.prepare()
        x = np.arange(self.positions) * length / self.positions
        v = space.coordinates_['v']
        profile = np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi)
        if self.setup == 'two-stream':
            profile = (np.exp(-2 * (v - 1.5) ** 2) + np.exp(-2 * (v + 1.5) ** 2)) * np.sqrt(2 / np.pi) / 2
        gas = FieldElement(
            'gas',
            element=PhaseSpaceDensityElement,
            space=space,
            values={'x': np.outer(1.0 + self.amplitude * np.cos(0.5 * x), profile)},
            discretization='finite-volume',
            visual=VLASOV_VISUAL,
        )
        return System(
            self.label,
            elements=[gas],
            interactions=[AdvectionInteraction('vlasov', target='gas', state='x', drift=self._get_poisson_drift)],
        )
