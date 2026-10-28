"""Run particle systems in LAMMPS, translating particle groups, pair potentials, boxes, forces, and thermostats."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import ClassVar, cast

import numpy as np
import numpy.typing as npt
from lammps import lammps
from sage.all__sagemath_symbolics import SR, Expression, oo
from sage.sets.real_set import RealSet

from ...core.elements import ParticleElement
from ...core.errors import BackendError, SolverError
from ...core.interactions import BaseInteraction, PairPotentialInteraction
from ...core.laws import BoundaryLaw, EquationLaw, FluxLaw, NoiseLaw, RateLaw
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...core.system import System
from ...core.types import Vector
from .._utils import declarations_of, symbol, symbols_of
from ..backends import NumpyBackend
from ..compile import compile_system
from ..problems import (
    CompiledElement,
    CompiledInteraction,
    ExpressionPrinter,
    Form,
    NoiseKind,
    OrderKind,
    Problem,
    SupportSpec,
    VariableKind,
)
from ..problems._steps import step_count
from ._external import ExternalSolver
from ._lammps_fix import LammpsFix

_LAMMPS_ARGUMENTS = ['-screen', 'none', '-log', 'none', '-nocite']
_DIMENSIONS = ('x', 'y', 'z')
MAX_SEED = 900_000_000
_SUPPORTED_DIMENSIONS = (2, 3)
type PairArguments = Callable[[Callable[[str], Vector]], str]
type FixCommand = Callable[[LammpsFix], str]


@dataclass
class _Atoms:
    """Particle elements and groups as LAMMPS atom types, with the flat state indices of every atom."""

    elements: list[CompiledElement] = field(default_factory=list)
    types: dict[str, int] = field(default_factory=dict)
    positions: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros((0, 0), dtype=np.intp))
    velocities: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros((0, 0), dtype=np.intp))
    atom_types: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    n_dim: int = 0


class LammpsSolver(ExternalSolver):
    """Solver that runs particle systems in LAMMPS, with LAMMPS velocity Verlet integration.

    Particle groups become atom types, and boxes become LAMMPS boundaries and `fix wall/reflect`. A pair potential
    becomes the native pair style `pairs` gives for its class, or a Lepton expression potential otherwise. An
    interaction whose class `fixes` gives a native fix becomes that fix. Any other interaction that only changes
    particle velocities runs through `fix external`, evaluated by the framework.

    Args:
        dt:
            The largest LAMMPS time step.

        pairs:
            The native LAMMPS pair style of pair potential classes, with the function of the parameter values that
            returns the coefficients of `pair_coeff` without the cutoff, which the solver appends.

        fixes:
            The function that returns the native LAMMPS fix command of interaction classes from the values of the fix.

    Attributes:
        lammps_:
            The LAMMPS instance, which the final run `run(..., final=True)` closes.

        released_:
            Whether the LAMMPS instance is closed.

    Examples:
        >>> from skcomplex.core.elements import GroupElement, ParticleElement
        >>> from skcomplex.core.interactions import PairPotentialInteraction
        >>> from skcomplex.core.spaces import ContinuousSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.engine.solvers import LammpsSolver
        >>> class LennardJonesInteraction(PairPotentialInteraction):
        ...     def _get_pair_energy(self, namespace, r):
        ...         return 4 * (r**-12 - r**-6)
        >>> system = System(
        ...     elements=[
        ...         GroupElement('gas', element=ParticleElement, size=2, values={'x': [[0.0, 0.0], [1.5, 0.0]]}),
        ...     ],
        ...     interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=2.5)],
        ...     space=ContinuousSpace(n_dim=2),
        ... )
        >>> solver = LammpsSolver(dt=0.005)
        >>> trajectory = Simulation(system, solver=solver, dt=0.1).check().run(until=0.5).trajectory_
        >>> positions = trajectory.values('gas.x')
        >>> bool(positions[-1, 1, 0] - positions[-1, 0, 0] < 1.5)
        True
    """

    supports: ClassVar[SupportSpec] = SupportSpec(
        time={ContinuousTime},
        space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
        laws={RateLaw, FluxLaw, NoiseLaw, EquationLaw, BoundaryLaw},
        noise={NoiseKind.DIAGONAL},
        orders={OrderKind.INTEGER},
    )

    name: ClassVar[str] = 'LAMMPS'
    form: ClassVar[Form] = Form.ODE
    forms: ClassVar[frozenset[Form]] = frozenset({Form.ODE, Form.SDE})

    def __init__(
        self,
        dt: float = 0.005,
        pairs: Mapping[type[BaseInteraction], tuple[str, PairArguments]] | None = None,
        fixes: Mapping[type[BaseInteraction], FixCommand] | None = None,
    ) -> None:
        self.dt = dt
        self.pairs = pairs
        self.fixes = fixes

    def _unsupported(self, problem: Problem) -> list[str]:
        """Return the elements and interactions of a problem that LAMMPS cannot run.

        Args:
            problem:
                The compiled problem.

        Returns:
            unsupported:
                One description per unsupported element or interaction.
        """
        unsupported = self._check_elements(problem)
        if unsupported:
            return unsupported + [f'interaction {item.path}' for item in problem.interactions]
        for item in problem.interactions:
            if isinstance(item.interaction, PairPotentialInteraction):
                unsupported += self._check_pair(problem, item)
        external = self._external_paths(problem)
        if external:
            unsupported += self._check_external(problem, external)
        return unsupported

    def _check_pair(self, problem: Problem, item: CompiledInteraction) -> list[str]:
        """Return a description of a pair interaction whose energy has no Lepton form."""
        if self._native(self.pairs or {}, item.interaction) is not None:
            return []
        try:
            self._pair_expression(problem, problem.initial_parameters, item)
        except BackendError as error:
            return [f'pair interaction {item.path}, because {error}']
        return []

    def _external_paths(self, problem: Problem) -> set[str]:
        """Return the paths of the interactions that run through `fix external`."""
        return {
            item.path
            for item in problem.interactions
            if not isinstance(item.interaction, PairPotentialInteraction)
            and self._native(self.fixes or {}, item.interaction) is None
        }

    def _check_external(self, problem: Problem, external: set[str]) -> list[str]:
        """Return the external interactions whose laws do more than change particle velocities."""
        filtered = self._filter_system(problem.system, external)
        sub_problem = cast('Problem', compile_system(filtered, backend=NumpyBackend()))  # type: ignore[arg-type]
        reasons = []
        if sub_problem.form != Form.ODE:
            reasons.append(f'the {sub_problem.form.value} form')
        if sub_problem.boundary is not None:
            reasons.append('boundary laws')
        velocities = {element.states['v'] for element in sub_problem.elements}
        for name, equation in sub_problem.equations.items():
            base = name.split('[')[0]
            if base not in velocities and not equation.is_symbol():
                reasons.append(f'a law on {base}')
        if not reasons:
            return []
        return [f'interactions {", ".join(sorted(external))}, which have {", ".join(sorted(set(reasons)))}']

    def _pair_expression(self, problem: Problem, p: Vector, item: CompiledInteraction) -> str:
        """Return the Lepton expression of a pair interaction with its parameters at their current values."""
        interaction = item.interaction
        values = {
            name: float(self._parameter(problem, p, item.path, name)[0])
            for name in declarations_of(interaction, VariableKind.PARAMETER)
        }
        roles = {
            role: SimpleNamespace(m=float(self._parameter(problem, p, self._role_path(item, role), 'm')[0]))
            for role in item.roles
        }
        distance = symbol('r', RealSet.open(0, oo))
        energy = interaction._get_pair_energy(SimpleNamespace(**values, **roles), distance)
        return self._lepton_expression(energy, distance)

    def _box_commands(self, problem: Problem, x: Vector) -> list[str]:
        """Return the commands that create the simulation box."""
        atoms = self.atoms_
        box = self._box(problem)
        positions = x[atoms.positions]
        if box is not None:
            low, high = np.zeros(atoms.n_dim), np.asarray(box.extent_)
            boundary = ['p' if box.boundary == 'periodic' else 'f'] * atoms.n_dim
        else:
            cutoffs = [
                float(self._parameter(problem, problem.initial_parameters, item.path, 'cutoff')[0])
                for item in problem.interactions
                if isinstance(item.interaction, PairPotentialInteraction)
            ]
            margin = max(cutoffs, default=0.0) + 1.0
            low, high = positions.min(axis=0) - margin, positions.max(axis=0) + margin
            boundary = ['m'] * atoms.n_dim
        bounds = [f'{low[dimension]} {high[dimension]}' for dimension in range(atoms.n_dim)]
        if atoms.n_dim == _SUPPORTED_DIMENSIONS[0]:
            bounds.append('-0.5 0.5')
            boundary.append('p')
        return [
            'units lj',
            f'dimension {atoms.n_dim}',
            'atom_style atomic',
            'atom_modify map array sort 0 0.0',
            f'boundary {" ".join(boundary)}',
            f'region box block {" ".join(bounds)}',
            f'create_box {len(atoms.elements)} box',
        ]

    def _pair_commands(self, problem: Problem, p: Vector) -> list[str]:
        """Return the commands of the pair style and of the coefficients of every pair interaction."""
        pairs = [item for item in problem.interactions if isinstance(item.interaction, PairPotentialInteraction)]
        styles, coefficients = [], []
        counts: defaultdict[str, int] = defaultdict(int)
        largest = 1.0
        for item in pairs:
            cutoff = float(self._parameter(problem, p, item.path, 'cutoff')[0])
            largest = max(largest, cutoff)
            first, second = sorted(
                (self.atoms_.types[self._role_path(item, 'first')], self.atoms_.types[self._role_path(item, 'second')]),
            )
            native = self._native(self.pairs or {}, item.interaction)
            if native is not None:
                style, coefficients_of = native
                arguments = f'{coefficients_of(functools.partial(self._parameter, problem, p, item.path))} {cutoff}'
            else:
                style = 'lepton'
                arguments = f'"{self._pair_expression(problem, p, item)}" {cutoff}'
            counts[style] += 1
            styles.append(f'{style} {cutoff}')
            coefficients.append((first, second, style, counts[style], arguments))
        repeated = {style for style, count in counts.items() if count > 1}
        commands = [f'pair_style hybrid/overlay zero {largest} {" ".join(styles)}', 'pair_coeff * * zero']
        for first, second, style, instance, arguments in coefficients:
            label = f'{style} {instance}' if style in repeated else style
            commands.append(f'pair_coeff {first} {second} {label} {arguments}')
        return commands

    def _fix_commands(self, problem: Problem, p: Vector) -> list[str]:
        """Return the commands of the integrator and of every box, force, drag, and thermostat fix."""
        atoms = self.atoms_
        commands = ['fix nve all nve']
        if atoms.n_dim == _SUPPORTED_DIMENSIONS[0]:
            commands.append('fix plane all enforce2d')
        box = self._box(problem)
        if box is not None and box.boundary == 'reflecting':
            walls = ' '.join(
                f'{name}lo 0.0 {name}hi {side}' for name, side in zip(_DIMENSIONS, box.extent_, strict=False)
            )
            commands.append(f'fix walls all wall/reflect {walls}')
        for index, item in enumerate(problem.interactions):
            command = self._native(self.fixes or {}, item.interaction)
            if command is None:
                continue
            target = self._role_path(item, 'target') if 'target' in item.roles else None
            fix = LammpsFix(
                name=f'fix{index}',
                group=f'group{atoms.types[target]}' if target is not None else 'all',
                value=functools.partial(self._parameter, problem, p, item.path),
                mass=functools.partial(self._target_mass, problem, p, target),
                seed=self._seed,
            )
            commands.append(command(fix))
        if self.external_ is not None:
            commands.append('fix external all external pf/callback 1 1')
        return commands

    def _target_mass(self, problem: Problem, p: Vector, target: str | None) -> float:
        """Return the mass of the atoms of the element or group an interaction targets."""
        return self._mass(problem, p, next(element for element in self.atoms_.elements if element.path == target))

    def _seed(self) -> int:
        """Return a random seed for a LAMMPS fix."""
        return int(self.rng_.integers(1, MAX_SEED))

    def _external_callback(self) -> Callable[..., None]:
        """Return the `fix external` callback that adds the forces of the external interactions."""
        sub_problem = self.external_
        assert sub_problem is not None
        drift = sub_problem.drift
        assert drift is not None
        atoms = self.atoms_
        sub_positions = np.concatenate(
            [self._state_indices(sub_problem, element.states['x']) for element in sub_problem.elements],
        )
        sub_velocities = np.concatenate(
            [self._state_indices(sub_problem, element.states['v']) for element in sub_problem.elements],
        )
        masses = np.concatenate(
            [
                np.full(element.members or 1, self._mass(self.problem_, self.parameters_, element))
                for element in atoms.elements
            ],
        )

        def callback(  # noqa: PLR0917  # the signature of LAMMPS fix external callbacks
            caller: object,
            step: int,
            n_local: int,
            tags: npt.NDArray[np.int64],
            x: Vector,
            forces: Vector,
        ) -> None:
            """Write the forces of the external interactions on every local atom."""
            order = np.asarray(tags[:n_local], dtype=np.intp) - 1
            state = sub_problem.initial_state.copy()
            state[sub_positions[order]] = x[:n_local, : atoms.n_dim]
            state[sub_velocities[order]] = self.lammps_.numpy.extract_atom('v')[:n_local, : atoms.n_dim]
            parameters = self._sub_parameters()
            time = self.time_ + (step - self.start_step_) * self.timestep_
            accelerations = drift(time, state, parameters)
            forces[:n_local, :] = 0.0
            forces[:n_local, : atoms.n_dim] = accelerations[sub_velocities[order]] * masses[order, None]

        return callback

    def _sub_parameters(self) -> Vector:
        """Return the parameter vector of the external interactions at the current parameter values."""
        sub_problem = self.external_
        assert sub_problem is not None
        parameters = sub_problem.initial_parameters.copy()
        for entry in sub_problem.parameters.entries:
            parameters[entry.indices] = self.parameters_[self.problem_.parameters.entry(entry.name).indices]
        return parameters

    def _build(self, problem: Problem, rng: np.random.Generator) -> None:
        """Create the LAMMPS instance, the box, the atoms, and every pair style and fix.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator that seeds the thermostats.
        """
        self.atoms_ = self._atoms_of(problem)
        external = self._external_paths(problem)
        self.external_ = (
            cast('Problem', compile_system(self._filter_system(problem.system, external)))  # type: ignore[arg-type]
            if external
            else None
        )
        self.parameters_ = problem.initial_parameters.copy()
        self.time_, self.timestep_, self.start_step_ = 0.0, self.dt, 0
        self.lammps_ = lammps(cmdargs=_LAMMPS_ARGUMENTS)
        self.released_ = False
        self.lammps_.commands_list(self._box_commands(problem, problem.initial_state))
        for element in self.atoms_.elements:
            self.lammps_.command(
                f'mass {self.atoms_.types[element.path]} {self._mass(problem, self.parameters_, element)}',
            )
        self._create_atoms(problem.initial_state)
        self.lammps_.commands_list([f'group group{number} type {number}' for number in self.atoms_.types.values()])
        self.lammps_.commands_list(
            [
                *self._pair_commands(problem, self.parameters_),
                'neigh_modify delay 0 every 1 check yes',
                f'timestep {self.dt}',
            ],
        )
        self.lammps_.commands_list(self._fix_commands(problem, self.parameters_))
        if self.external_ is not None:
            self.lammps_.set_fix_external_callback('external', self._external_callback(), None)

    def _create_atoms(self, x: Vector) -> None:
        """Create one LAMMPS atom per member, with ids in state vector order."""
        atoms = self.atoms_
        count = len(atoms.atom_types)
        positions, velocities = np.zeros((count, 3)), np.zeros((count, 3))
        positions[:, : atoms.n_dim], velocities[:, : atoms.n_dim] = x[atoms.positions], x[atoms.velocities]
        ids = list(range(1, count + 1))
        self.lammps_.create_atoms(
            count,
            ids,
            atoms.atom_types.tolist(),
            positions.ravel().tolist(),
            velocities.ravel().tolist(),
        )

    def _update_parameters(self, p: Vector) -> None:
        """Reissue the masses, pair coefficients, and fixes after a parameter change."""
        problem, atoms = self.problem_, self.atoms_
        self.parameters_ = p.copy()
        for element in atoms.elements:
            self.lammps_.command(f'mass {atoms.types[element.path]} {self._mass(problem, p, element)}')
        self.lammps_.commands_list(self._pair_commands(problem, p))
        fixes = self._fix_commands(problem, p)
        self.lammps_.commands_list([f'unfix {command.split()[1]}' for command in fixes])
        self.lammps_.commands_list(fixes)
        if self.external_ is not None:
            self.lammps_.set_fix_external_callback('external', self._external_callback(), None)

    def _local_atoms(self) -> tuple[npt.NDArray[np.intp], Vector, Vector]:
        """Return the state order, positions, and velocities of the local atoms, without periodic ghost atoms."""
        numpy, count = self.lammps_.numpy, int(self.lammps_.extract_global('nlocal'))
        order = np.asarray(numpy.extract_atom('id')[:count], dtype=np.intp) - 1
        return order, numpy.extract_atom('x')[:count], numpy.extract_atom('v')[:count]

    def _advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Write the state into LAMMPS, run it to time `until`, and read the state back.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to.

        Returns:
            state:
                The state vector at time `until`.

        Raises:
            SolverError:
                The LAMMPS instance was closed by a final run.
        """
        if self.released_:
            error_msg = 'LammpsSolver closed its LAMMPS instance at the end of the run. Fit the solver again to rerun.'
            raise SolverError(error_msg)
        if not np.array_equal(p, self.parameters_):
            self._update_parameters(p)
        atoms = self.atoms_
        order, positions, velocities = self._local_atoms()
        positions[:, : atoms.n_dim] = x[atoms.positions[order]]
        velocities[:, : atoms.n_dim] = x[atoms.velocities[order]]
        n_steps = step_count(t, until, self.dt)
        timestep = (until - t) / n_steps
        if timestep != self.timestep_:
            self.lammps_.command(f'timestep {timestep}')
            self.timestep_ = timestep
        self.time_ = t
        self.start_step_ = int(self.lammps_.extract_global('ntimestep'))
        self.lammps_.command(f'run {n_steps} pre yes post no')
        state = x.copy()
        order, positions, velocities = self._local_atoms()
        state[atoms.positions[order]] = positions[:, : atoms.n_dim]
        state[atoms.velocities[order]] = velocities[:, : atoms.n_dim]
        return state

    def _release(self) -> None:
        """Close the LAMMPS instance."""
        if not self.released_:
            self.lammps_.close()
            self.released_ = True

    @staticmethod
    def _native[T](table: Mapping[type[BaseInteraction], T], interaction: BaseInteraction) -> T | None:
        """Return the native translation of an interaction from the nearest class in a table, or `None`."""
        return next((table[cls] for cls in type(interaction).__mro__ if cls in table), None)

    @staticmethod
    def _state_indices(problem: Problem, name: str) -> npt.NDArray[np.intp]:
        """Return the flat indices of a vector state as one row per member."""
        entry = problem.states.entry(name)
        return (entry.start + np.arange(entry.size)).reshape(entry.rows, entry.columns)

    @staticmethod
    def _atoms_of(problem: Problem) -> _Atoms:
        """Return the atom types and state indices of the particle elements and groups of a problem."""
        atoms = _Atoms(elements=list(problem.elements))
        positions, velocities, types = [], [], []
        for number, element in enumerate(problem.elements, start=1):
            atoms.types[element.path] = number
            member_positions = LammpsSolver._state_indices(problem, element.states['x'])
            positions.append(member_positions)
            velocities.append(LammpsSolver._state_indices(problem, element.states['v']))
            types.append(np.full(len(member_positions), number, dtype=np.intp))
            atoms.n_dim = member_positions.shape[1]
        atoms.positions = np.concatenate(positions)
        atoms.velocities = np.concatenate(velocities)
        atoms.atom_types = np.concatenate(types)
        return atoms

    @staticmethod
    def _role_path(item: CompiledInteraction, role: str) -> str:
        """Return the path of the group or element a role of an interaction binds to."""
        path = item.roles[role]
        if not isinstance(path, str):
            error_msg = f'Interaction {item.path} binds the role {role!r} to an element set, which LAMMPS does not run.'
            raise BackendError(error_msg)
        return path

    @staticmethod
    def _parameter(problem: Problem, p: Vector, path: str, name: str) -> Vector:
        """Return the value of a parameter of an element, group, or interaction."""
        return np.atleast_1d(np.asarray(problem.parameters.read(p, f'{path}.{name}'), dtype=float))

    @staticmethod
    def _mass(problem: Problem, p: Vector, element: CompiledElement) -> float:
        """Return the mass of the members of an element or group, which LAMMPS stores per atom type."""
        return float(LammpsSolver._parameter(problem, p, element.path, 'm')[0])

    @staticmethod
    def _filter_system(system: System, keep: set[str], scope: str = '') -> System:
        """Return a copy of a system tree that keeps only the interactions whose paths are in `keep`.

        A box becomes an unbounded continuous space of its dimension, since LAMMPS applies the walls and the periodic
        sides itself and the copy only evaluates forces.
        """
        space = system.space
        if isinstance(space, ContinuousSpace):
            space = ContinuousSpace(n_dim=space.n_dim or len(space.extent))
        return System(
            system.label,
            elements=system.elements,
            interactions=[item for item in system.interactions if f'{scope}{item.label}' in keep],
            systems=[LammpsSolver._filter_system(child, keep, f'{scope}{child.label}/') for child in system.systems],
            space=space,
            time=system.time,
        )

    @staticmethod
    def _check_elements(problem: Problem) -> list[str]:
        """Return the elements that are not particles in two or three dimensions with one mass per group."""
        unsupported = []
        for element in problem.elements:
            element_type = element.element.element if element.members is not None else type(element.element)
            if not issubclass(element_type, ParticleElement):
                unsupported.append(f'element {element.path} of type {element_type.__name__}, which is not a Particle')
                continue
            n_dim = problem.states.entry(element.states['x']).columns
            if n_dim not in _SUPPORTED_DIMENSIONS:
                unsupported.append(f'element {element.path} in {n_dim} dimensions, while LAMMPS runs 2 or 3')
            masses = LammpsSolver._parameter(problem, problem.initial_parameters, element.path, 'm')
            if not np.all(masses == masses[0]):
                unsupported.append(f'group {element.path}, whose members have different masses')
        return unsupported

    @staticmethod
    def _box(problem: Problem) -> ContinuousSpace | None:
        """Return the continuous space of a problem when it is a box with walls or joined sides, or `None`."""
        space = getattr(problem.system, 'space', None)
        return space if isinstance(space, ContinuousSpace) and space.boundary != 'open' else None

    @staticmethod
    def _lepton_expression(expression: Expression, distance: Expression) -> str:
        """Return a pair energy as a Lepton expression of the distance `r`.

        Args:
            expression:
                The pair energy, whose only symbol is the distance.

            distance:
                The distance symbol.

        Returns:
            expression:
                The Lepton expression.

        Raises:
            BackendError:
                The energy has symbols other than the distance or functions Lepton does not know.

        Examples:
            >>> from skcomplex.engine._utils import symbol
            >>> from skcomplex.engine.solvers import LammpsSolver
            >>> x = symbol('x')
            >>> LammpsSolver._lepton_expression(4 * (x**-12 - x**-6), x)
            '-4/r^6 + 4/r^12'
        """
        expression = SR(expression)
        others = symbols_of([expression]) - {distance}
        if others:
            names = ', '.join(sorted(map(str, others)))
            error_msg = f'The pair energy depends on {names}, which have no value for LAMMPS.'
            raise BackendError(error_msg)
        lepton = expression.subs({distance: symbol('r', RealSet.open(0, oo))}).expand()
        return ExpressionPrinter('lepton', shared=False).check().prepare().run([lepton])[1][0]
