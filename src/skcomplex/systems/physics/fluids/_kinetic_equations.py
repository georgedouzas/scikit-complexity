"""Kinetic equations of gases and plasmas over position and velocity.

The equations live on the phase spaces of the core.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, vector
from sage.modules.free_module_element import FreeModuleElement

from ....core.components import Observable, Tags
from ....core.components.base import BaseComponent
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.components.laws.utils import coordinate, gradient, inverse_helmholtz, maximum_entropy, moment
from ....core.components.spaces import AxisSpec, BaseSpace, LatticeSpace, PhaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, RK4Solver, SemiLagrangianSolver, SSPRKSolver, StrangSplittingSolver
from ....core.errors import ValidationError
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.kinetics import EncounterInteraction
from ..common.fields import AdvectionInteraction, PhaseSpaceDensityElement

SETUPS = ('landau', 'two-stream')
VELOCITY_AXES = ('vx', 'vy')


class GasDensityElement(PhaseSpaceDensityElement):
    """Phase space density of a gas, drawn as its density along position.

    Args:
        label:
            The label of the gas.

        x:
            The initial density, one nonnegative value per point or one for all.

        space:
            The phase space of the points, or `None` for one point.

    Examples:
        >>> from skcomplex.systems.physics.fluids._kinetic_equations import GasDensityElement
        >>> GasDensityElement('gas', x=0.5).x
        0.5
    """

    visual = VisualSpec('image', {'moments': (0,)})


class RelaxationTimeInteraction(Interaction):
    """Relaxation of a state towards a goal over a relaxation time.

    The state follows `x' = (goal - x) / time`, with the noise `noise dW` when one is given. With a constant goal and
    time this is the Ornstein-Uhlenbeck process, whose stationary variance is `noise**2 time / 2`. It is Newton
    cooling, a belief adapting to news, or the BGK collision term of a gas. The goal, the time, and the noise are
    numbers or functions of the namespace of the interaction, and a goal that is a number applies to every component
    of a vector state.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the element or group whose state relaxes.

        state:
            The name of the state of the target that relaxes.

        goal:
            A function of the namespace that returns the goal, or a number.

        time:
            A function of the namespace that returns the relaxation time, or a positive number.

        noise:
            A function of the namespace that returns the amplitude of the noise, a number, or `None` for no noise.

    Raises:
        ValidationError:
            A goal, a time, or a noise is neither callable nor a number.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.fluids import RelaxationTimeInteraction
        >>> relaxation = RelaxationTimeInteraction('relaxation', target='stock', goal=10.0, time=0.5)
        >>> system = System(elements=[Element('stock', values={'x': 0.0})], interactions=[relaxation]).prepare()
        >>> round(float(system.simulate(until=5.0, dt=0.01).get_values('stock.x')[-1]), 3)
        10.0
    """

    law_classes = (RateLaw, NoiseLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the rate law, with a noise law for a noise."""
        return self._get_variant_tags((RateLaw, NoiseLaw) if self.noise is not None else (RateLaw,))

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'x',
        goal: Callable[[SimpleNamespace], object] | float = 0.0,
        time: Callable[[SimpleNamespace], object] | float = 1.0,
        noise: Callable[[SimpleNamespace], object] | float | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.goal = goal
        self.time = time
        self.noise = noise

    def _evaluate(self, name: str, context: SimpleNamespace) -> object:
        """Return the value of a function of the namespace, or the symbol of a number."""
        value = getattr(self, name)
        return value(context) if callable(value) else getattr(context, name)

    def _get_goal(self, context: SimpleNamespace) -> object:
        """Return the goal, the hook a subclass overrides to derive it from its parameters."""
        return self._evaluate('goal', context)

    def _get_time(self, context: SimpleNamespace) -> object:
        """Return the relaxation time."""
        return self._evaluate('time', context)

    def _get_noise(self, context: SimpleNamespace) -> object:
        """Return the amplitude of the noise, or `None` for none."""
        return None if self.noise is None else self._evaluate('noise', context)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the relaxation of the state of the target and its noise.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.

        Raises:
            ValidationError:
                A goal, a time, or a noise is neither callable nor a number.
        """
        super()._prepare(context)
        for name in ('goal', 'time', 'noise'):
            value = getattr(self, name)
            if not (callable(value) or isinstance(value, int | float) or (name == 'noise' and value is None)):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace or a number.'
                raise ValidationError(error_msg)
        if isinstance(context, SimpleNamespace):
            x = getattr(context.target, self.state)
            goal = self._get_goal(context)
            if isinstance(x, FreeModuleElement) and not isinstance(goal, FreeModuleElement):
                goal = vector(SR, [goal] * len(x))
            laws: list[BaseComponent] = [RateLaw(x.diff(context.t), (goal - x) / self._get_time(context))]
            noise = self._get_noise(context)
            if noise is not None:
                laws.append(NoiseLaw(x, noise))
            self.laws_ = laws


class BhatnagarGrossKrookInteraction(RelaxationTimeInteraction):
    """Relaxation of Bhatnagar, Gross, and Krook of a density.

    The density relaxes towards the maximum-entropy density with its moments.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

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
        target: Role,
        state: str = 'x',
        axes: Sequence[str] = ('v',),
        time: Callable[[SimpleNamespace], object] | float = 1.0,
    ) -> None:
        super().__init__(label, target=target, state=state, goal=0.0, time=time)
        self.axes = axes

    def _get_goal(self, namespace: SimpleNamespace) -> object:
        """Return the maximum-entropy density with the moments of the state of the target."""
        return maximum_entropy(getattr(namespace.target, self.state), axes=self.axes)


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

        solver:
            The solver of the problem, or `'auto'` for the steps the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.fluids import BoltzmannEquationSystem
        >>> BoltzmannEquationSystem(4, 8).prepare().system_.elements[0].space.prepare().size_
        256
    """

    field = 'physics/fluids'
    topics = ('field', 'kinetic', 'phase space', 'collisions', 'pde')
    references = ('Bellomo et al. 2017', 'Mouhot and Pareschi 2006')
    until = 0.5
    dt = 0.1

    @staticmethod
    def _get_plane_streaming(_: SimpleNamespace) -> list[Expression]:
        """Return the drift of free streaming over one position and two velocity axes."""
        return [coordinate('vx'), 0, 0]

    @staticmethod
    def _get_uniform_gas(
        positions: int,
        velocities: int,
        knudsen: float,
        half_width: float,
    ) -> tuple[PhaseSpaceDensityElement, float]:
        """Return a gas over one position and two velocity axes, and its collision rate.

        The gas has a density wave, and the rate is its collision frequency.
        """
        length = 2 * math.pi
        axes = [AxisSpec(name, velocities, -half_width, half_width) for name in VELOCITY_AXES]
        position = LatticeSpace((positions,), spacing=length / positions, discretization='finite-volume')
        space = PhaseSpace(position, axes).prepare()
        vx, vy = np.meshgrid(space.coordinates_['vx'], space.coordinates_['vy'], indexing='ij')
        x = np.arange(positions) * length / positions
        profile = np.exp(-(vx**2 + vy**2) / 2) / (2 * np.pi)
        values = np.multiply.outer(1.0 + 0.2 * np.sin(x), profile)
        return PhaseSpaceDensityElement('gas', x=values.ravel(), space=space), 1.0 / knudsen

    @staticmethod
    def _get_maxwell_molecules(_speed: object, _cosine: object) -> float:
        """Return the cross-section of Maxwell molecules in two velocity dimensions."""
        return 1 / (2 * math.pi)

    def __init__(
        self,
        positions: int = 8,
        velocities: int = 12,
        *,
        knudsen: float = 1.0,
        half_width: float = 6.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.positions = positions
        self.velocities = velocities
        self.knudsen = knudsen
        self.half_width = half_width

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a density wave of Maxwell molecules over one position and two velocity axes.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        gas, frequency = BoltzmannEquationSystem._get_uniform_gas(
            self.positions,
            self.velocities,
            self.knudsen,
            self.half_width,
        )
        stream = AdvectionInteraction(
            'stream',
            target='gas',
            state='x',
            drift=BoltzmannEquationSystem._get_plane_streaming,
        )
        collide = EncounterInteraction(
            'collide',
            candidate='gas',
            cross_section=BoltzmannEquationSystem._get_maxwell_molecules,
            factor=lambda _: frequency,
        )
        self.system_ = System(
            self.label,
            elements=[gas],
            interactions=[stream, collide],
            solver=StrangSplittingSolver(SemiLagrangianSolver(0.1), RK4Solver(0.05), 0.1),
        )


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

        solver:
            The solver of the problem, or `'auto'` for the steps the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.fluids import EnskogSystem
        >>> EnskogSystem(4, 8).prepare().system_.elements[0].space.prepare().size_
        256
    """

    field = 'physics/fluids'
    topics = ('field', 'kinetic', 'phase space', 'collisions', 'pde')
    references = ('Bellomo et al. 2017', 'Enskog 1922')
    until = 0.5
    dt = 0.1

    def __init__(
        self,
        positions: int = 8,
        velocities: int = 12,
        *,
        knudsen: float = 1.0,
        diameter: float = 0.2,
        half_width: float = 6.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.positions = positions
        self.velocities = velocities
        self.knudsen = knudsen
        self.diameter = diameter
        self.half_width = half_width

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a density wave of dense disks over one position and two velocity axes.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        gas, frequency = BoltzmannEquationSystem._get_uniform_gas(
            self.positions,
            self.velocities,
            self.knudsen,
            self.half_width,
        )
        area = math.pi * self.diameter**2 / 4

        def contact(s: SimpleNamespace) -> Expression:
            fraction = area * moment(s.candidate.x, 0, axes=VELOCITY_AXES)
            return frequency * (1 - 7 * fraction / 16) / (1 - fraction) ** 2

        stream = AdvectionInteraction(
            'stream',
            target='gas',
            state='x',
            drift=BoltzmannEquationSystem._get_plane_streaming,
        )
        collide = EncounterInteraction(
            'collide',
            candidate='gas',
            cross_section=BoltzmannEquationSystem._get_maxwell_molecules,
            factor=contact,
        )
        self.system_ = System(
            self.label,
            elements=[gas],
            interactions=[stream, collide],
            solver=StrangSplittingSolver(SemiLagrangianSolver(0.1), RK4Solver(0.05), 0.1),
        )


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

        solver:
            The solver of the problem, or `'auto'` for the steps the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.fluids import BGKEquationSystem
        >>> BGKEquationSystem(8, 16).prepare().system_.elements[0].space.prepare().size_
        128
    """

    field = 'physics/fluids'
    topics = ('field', 'kinetic', 'phase space', 'shocks', 'pde')
    references = ('Bhatnagar, Gross, and Krook 1954', 'Mieussens 2000', 'Sod 1978')
    until = 0.2
    dt = 0.05

    def __init__(
        self,
        positions: int = 100,
        velocities: int = 32,
        *,
        knudsen: float = 1e-2,
        half_width: float = 6.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Sod shock tube over one position and one velocity axis.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        position = LatticeSpace(
            (self.positions,),
            spacing=1.0 / self.positions,
            boundary='reflecting',
            discretization='finite-volume',
        )
        space = PhaseSpace(position, [AxisSpec('v', self.velocities, -self.half_width, self.half_width)]).prepare()
        left = np.arange(self.positions) < self.positions // 2
        density = np.where(left, 1.0, 0.125)
        temperature = np.where(left, 1.0, 0.8)
        values = self._get_maxwellian(density, np.zeros(self.positions), temperature, space.coordinates_['v'])
        gas = GasDensityElement('gas', x=values.ravel(), space=space)
        stream = AdvectionInteraction('stream', target='gas', state='x', drift=self._get_streaming)
        relax = BhatnagarGrossKrookInteraction('relax', target='gas', axes=('v',), time=self.knudsen)
        self.system_ = System(
            self.label,
            elements=[gas],
            interactions=[stream, relax],
            solver=StrangSplittingSolver(SSPRKSolver(), RK4Solver(0.005), 0.005),
        )


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

        solver:
            The solver of the problem, or `'auto'` for semi-Lagrangian steps of 0.25.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The set-up is unknown.

    Examples:
        >>> from skcomplex.systems.physics.fluids import VlasovEquationSystem
        >>> VlasovEquationSystem(8, 16).prepare().system_.elements[0].space.prepare().size_
        128
    """

    field = 'physics/fluids'
    topics = ('field', 'kinetic', 'phase space', 'plasma', 'pde')
    references = ('Bellomo et al. 2017', 'Cheng and Knorr 1976')
    until = 30.0
    dt = 0.25

    def __init__(
        self,
        positions: int = 32,
        velocities: int = 64,
        *,
        setup: str = 'landau',
        amplitude: float = 0.01,
        half_width: float = 6.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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
        return [coordinate('v'), -gradient(inverse_helmholtz(1 - density), 1)[0]]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of electrons over a neutralizing background from the set-up.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The set-up is unknown.
        """
        super()._prepare(context)
        if self.setup not in SETUPS:
            error_msg = f'Unknown set-up {self.setup!r} of the Vlasov equation. Use one of {", ".join(SETUPS)}.'
            raise ValidationError(error_msg)
        length = 4 * math.pi
        space = PhaseSpace(
            LatticeSpace((self.positions,), spacing=length / self.positions, discretization='finite-volume'),
            [AxisSpec('v', self.velocities, -self.half_width, self.half_width)],
        )
        space.prepare()
        x = np.arange(self.positions) * length / self.positions
        v = space.coordinates_['v']
        profile = np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi)
        if self.setup == 'two-stream':
            profile = (np.exp(-2 * (v - 1.5) ** 2) + np.exp(-2 * (v + 1.5) ** 2)) * np.sqrt(2 / np.pi) / 2
        gas = GasDensityElement('gas', x=np.outer(1.0 + self.amplitude * np.cos(0.5 * x), profile).ravel(), space=space)
        self.system_ = System(
            self.label,
            elements=[gas],
            interactions=[AdvectionInteraction('vlasov', target='gas', state='x', drift=self._get_poisson_drift)],
            solver=SemiLagrangianSolver(0.25),
        )
