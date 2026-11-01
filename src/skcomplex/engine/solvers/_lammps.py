"""Run systems of points in LAMMPS, translating groups of points, pair potentials, boxes, and native fixes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from collections import defaultdict
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import cast

import numpy as np
import numpy.typing as npt
from lammps import lammps
from sage.all__sagemath_symbolics import SR, Expression, oo
from sage.sets.real_set import RealSet

from ...core.interactions import BaseInteraction, PairCouplingInteraction
from ...core.laws import BoundaryLaw, EquationLaw, FluxLaw, NoiseLaw, RateLaw
from ...core.spaces import CayleySpace, ContinuousSpace, LatticeSpace, MeshSpace, NetworkSpace
from ...core.specs import CompiledElementSpec, CompiledInteractionSpec
from ...core.system import System
from ...core.times import ContinuousTime
from ...errors import BackendError, SolverError
from ...specs import LammpsFixSpec, SupportSpec
from ...types import Array, FormKind, NoiseKind, OrderKind, VariableKind
from .._utils import declarations_of, symbol, symbols_of
from ..backends import NumpyBackend
from ..compile import compile_system
from ..problems import ExpressionPrinter, ProblemSpec
from ..problems._steps import step_count
from ..types import FixCommand, PairArguments
from ._external import ExternalSolver

_LAMMPS_ARGUMENTS = ['-screen', 'none', '-log', 'none', '-nocite']
_DIMENSIONS = ('x', 'y', 'z')
MAX_SEED = 900_000_000
_SUPPORTED_DIMENSIONS = (2, 3)
POSITION_ORDER = 2


class LammpsSolver(ExternalSolver):
    """Solver that runs systems of points in LAMMPS, with LAMMPS velocity Verlet integration.

    An element or group whose members declare a vector position state of order 2, such as `PointElement`, becomes an
    atom type, whose atoms move with the first derivative of the position as their velocity. The mass of an atom type
    is the parameter that the `divisor` of the interactions on its members names, and 1 without a divisor. Boxes become
    LAMMPS boundaries and `fix wall/reflect`. A `PairCouplingInteraction` with a potential becomes the native pair
    style `pairs` gives for its class, or a Lepton expression potential otherwise. An interaction whose class `fixes`
    gives a native fix becomes that fix. Any other interaction that only changes the derivatives of the positions runs
    through `fix external`, evaluated by the framework.

    Args:
        dt:
            The largest LAMMPS time step.

        pairs:
            The native LAMMPS pair style of pair potential classes, with the function of the parameter values that
            returns the coefficients of `pair_coeff` without the cutoff, which the solver appends.

        fixes:
            The function that returns the native LAMMPS fix command of interaction classes from the values of the fix.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Attributes:
        lammps_:
            The LAMMPS instance, which the final run `run(..., final=True)` closes.

        atom_elements_:
            The elements and groups of points, each a LAMMPS atom type.

        atom_numbers_:
            The LAMMPS atom type of every element and group of points, by path.

        position_indices_:
            The flat state indices of the position of every atom, of shape `(atoms, dimensions)`.

        velocity_indices_:
            The flat state indices of the velocity of every atom, of shape `(atoms, dimensions)`.

        atom_types_:
            The LAMMPS atom type of every atom.

        n_dim_:
            The number of dimensions of the space of the atoms.

        released_:
            Whether the LAMMPS instance is closed.

    Examples:
        >>> from skcomplex.core.elements import GroupElement, PointElement
        >>> from skcomplex.core.interactions import PairCouplingInteraction
        >>> from skcomplex.core.spaces import ContinuousSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.engine.solvers import LammpsSolver
        >>> potential = lambda namespace, r: 4 * (r**-12 - r**-6)
        >>> system = System(
        ...     elements=[
        ...         GroupElement('gas', element=PointElement, size=2, values={'x': [[0.0, 0.0], [1.5, 0.0]]}),
        ...     ],
        ...     interactions=[PairCouplingInteraction('lj', 'gas', 'gas', cutoff=2.5, potential=potential)],
        ...     space=ContinuousSpace(n_dim=2),
        ... )
        >>> solver = LammpsSolver(dt=0.005)
        >>> trajectory = Simulation(system, solver=solver, dt=0.1).check().prepare().run(until=0.5).trajectory_
        >>> positions = trajectory.values('gas.x')
        >>> bool(positions[-1, 1, 0] - positions[-1, 0, 0] < 1.5)
        True
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={RateLaw, FluxLaw, NoiseLaw, EquationLaw, BoundaryLaw},
            noise=frozenset({NoiseKind.DIAGONAL}),
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the external library the solver wraps."""
        return 'LAMMPS'

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.ODE

    @classmethod
    def _get_forms(cls) -> frozenset[FormKind]:
        """Return the other forms the solver also solves."""
        return frozenset({FormKind.ODE, FormKind.SDE})

    def __init__(
        self,
        dt: float = 0.005,
        pairs: Mapping[type[BaseInteraction], tuple[str, PairArguments]] | None = None,
        fixes: Mapping[type[BaseInteraction], FixCommand] | None = None,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.dt = dt
        self.pairs = pairs
        self.fixes = fixes
        self.random_state = random_state
        self.until = until

    def _unsupported(self, problem: ProblemSpec) -> list[str]:
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
            if self._is_potential(item):
                unsupported += self._check_pair(problem, item)
        external = self._external_paths(problem)
        if external:
            unsupported += self._check_external(problem, external)
        return unsupported

    def _check_pair(self, problem: ProblemSpec, item: CompiledInteractionSpec) -> list[str]:
        """Return a description of a pair interaction whose energy has no Lepton form."""
        if self._native(self.pairs or {}, item.interaction) is not None:
            return []
        try:
            self._pair_expression(problem, problem.initial_parameters, item)
        except BackendError as error:
            return [f'pair interaction {item.path}, because {error}']
        return []

    def _external_paths(self, problem: ProblemSpec) -> set[str]:
        """Return the paths of the interactions that run through `fix external`."""
        return {
            item.path
            for item in problem.interactions
            if not self._is_potential(item) and self._native(self.fixes or {}, item.interaction) is None
        }

    def _check_external(self, problem: ProblemSpec, external: set[str]) -> list[str]:
        """Return the external interactions whose laws do more than change the derivatives of the positions."""
        sub_problem = self._compile_external(problem.system, external, NumpyBackend())
        reasons = []
        if sub_problem.form != FormKind.ODE:
            reasons.append(f'the {sub_problem.form} form')
        if sub_problem.boundary is not None:
            reasons.append('boundary laws')
        velocities = {element.states[f'{self._position_name(element)}_d1'] for element in sub_problem.elements}
        for name, equation in sub_problem.equations.items():
            base = name.split('[')[0]
            if base not in velocities and not equation.is_symbol():
                reasons.append(f'a law on {base}')
        if not reasons:
            return []
        return [f'interactions {", ".join(sorted(external))}, which have {", ".join(sorted(set(reasons)))}']

    def _pair_expression(self, problem: ProblemSpec, p: Array, item: CompiledInteractionSpec) -> str:
        """Return the Lepton expression of a pair potential with its parameters at their current values."""
        interaction = cast(PairCouplingInteraction, item.interaction)
        assert interaction.potential is not None
        values = {
            name: float(self._parameter(problem, p, item.path, name)[0])
            for name in declarations_of(interaction, VariableKind.PARAMETER)
        }
        roles = {role: self._role_parameters(problem, p, self._role_path(item, role)) for role in item.roles}
        distance = symbol('r', RealSet.open(0, oo))
        energy = interaction.potential(SimpleNamespace(**values, **roles), distance)
        return self._lepton_expression(energy, distance)

    def _role_parameters(self, problem: ProblemSpec, p: Array, path: str) -> SimpleNamespace:
        """Return the namespace of the values of the parameters of the members of an element or group."""
        element = next(element for element in problem.elements if element.path == path)
        return SimpleNamespace(
            **{name: float(self._parameter(problem, p, path, name)[0]) for name in element.parameters},
        )

    def _box_commands(self, problem: ProblemSpec, x: Array) -> list[str]:
        """Return the commands that create the simulation box."""
        box = self._box(problem)
        positions = x[self.position_indices_]
        if box is not None:
            low, high = np.zeros(self.n_dim_), np.asarray(box.extent_)
            boundary = ['p' if box.boundary == 'periodic' else 'f'] * self.n_dim_
        else:
            cutoffs = [
                float(self._parameter(problem, problem.initial_parameters, item.path, 'cutoff')[0])
                for item in problem.interactions
                if self._is_potential(item)
            ]
            margin = max(cutoffs, default=0.0) + 1.0
            low, high = positions.min(axis=0) - margin, positions.max(axis=0) + margin
            boundary = ['m'] * self.n_dim_
        bounds = [f'{low[dimension]} {high[dimension]}' for dimension in range(self.n_dim_)]
        if self.n_dim_ == _SUPPORTED_DIMENSIONS[0]:
            bounds.append('-0.5 0.5')
            boundary.append('p')
        return [
            'units lj',
            f'dimension {self.n_dim_}',
            'atom_style atomic',
            'atom_modify map array sort 0 0.0',
            f'boundary {" ".join(boundary)}',
            f'region box block {" ".join(bounds)}',
            f'create_box {len(self.atom_elements_)} box',
        ]

    def _pair_commands(self, problem: ProblemSpec, p: Array) -> list[str]:
        """Return the commands of the pair style and of the coefficients of every pair interaction."""
        pairs = [item for item in problem.interactions if self._is_potential(item)]
        styles, coefficients = [], []
        counts: defaultdict[str, int] = defaultdict(int)
        largest = 1.0
        for item in pairs:
            cutoff = float(self._parameter(problem, p, item.path, 'cutoff')[0])
            largest = max(largest, cutoff)
            first, second = sorted(
                (
                    self.atom_numbers_[self._role_path(item, 'first')],
                    self.atom_numbers_[self._role_path(item, 'second')],
                ),
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

    def _fix_commands(self, problem: ProblemSpec, p: Array) -> list[str]:
        """Return the commands of the integrator and of every box, force, drag, and thermostat fix."""
        commands = ['fix nve all nve']
        if self.n_dim_ == _SUPPORTED_DIMENSIONS[0]:
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
            fix = LammpsFixSpec(
                name=f'fix{index}',
                group=f'group{self.atom_numbers_[target]}' if target is not None else 'all',
                value=functools.partial(self._parameter, problem, p, item.path),
                mass=functools.partial(self._target_mass, problem, p, target),
                seed=self._seed,
            )
            commands.append(command(fix))
        if self.external_ is not None:
            commands.append('fix external all external pf/callback 1 1')
        return commands

    def _target_mass(self, problem: ProblemSpec, p: Array, target: str | None) -> float:
        """Return the mass of the atoms of the element or group an interaction targets."""
        element = next(element for element in self.atom_elements_ if element.path == target)
        return self._mass(problem, p, element)

    def _seed(self) -> int:
        """Return a random seed for a LAMMPS fix."""
        return int(self.rng_.integers(1, MAX_SEED))

    def _external_callback(self) -> Callable[..., None]:
        """Return the `fix external` callback that adds the forces of the external interactions."""
        sub_problem = self.external_
        assert sub_problem is not None
        drift = sub_problem.drift
        assert drift is not None
        names = [self._position_name(element) for element in sub_problem.elements]
        sub_positions = np.concatenate(
            [
                self._state_indices(sub_problem, element.states[name])
                for element, name in zip(sub_problem.elements, names, strict=True)
            ],
        )
        sub_velocities = np.concatenate(
            [
                self._state_indices(sub_problem, element.states[f'{name}_d1'])
                for element, name in zip(sub_problem.elements, names, strict=True)
            ],
        )
        masses = np.concatenate(
            [
                np.full(element.members or 1, self._mass(self.problem_, self.parameters_, element))
                for element in self.atom_elements_
            ],
        )

        def callback(*arguments: object) -> None:
            """Write the forces of the external interactions on every local atom.

            LAMMPS passes the caller, the step, the number of local atoms, and their tags, positions, and forces.
            """
            signature = tuple[None, int, int, npt.NDArray[np.int64], Array, Array]
            _, step, n_local, tags, x, forces = cast(signature, arguments)
            order = np.asarray(tags[:n_local], dtype=np.intp) - 1
            state = sub_problem.initial_state.copy()
            state[sub_positions[order]] = x[:n_local, : self.n_dim_]
            state[sub_velocities[order]] = self.lammps_.numpy.extract_atom('v')[:n_local, : self.n_dim_]
            parameters = self._sub_parameters()
            time = self.time_ + (step - self.start_step_) * self.timestep_
            accelerations = drift(time, state, parameters)
            forces[:n_local, :] = 0.0
            forces[:n_local, : self.n_dim_] = accelerations[sub_velocities[order]] * masses[order, None]

        return callback

    def _sub_parameters(self) -> Array:
        """Return the parameter vector of the external interactions at the current parameter values."""
        sub_problem = self.external_
        assert sub_problem is not None
        parameters = sub_problem.initial_parameters.copy()
        for entry in sub_problem.parameters.entries:
            parameters[entry.indices] = self.parameters_[self.problem_.parameters.entry(entry.name).indices]
        return parameters

    def _build(self, problem: ProblemSpec) -> None:
        """Create the LAMMPS instance, the box, the atoms, and every pair style and fix.

        Args:
            problem:
                The compiled problem.

        """
        self._set_atoms(problem)
        external = self._external_paths(problem)
        self.external_ = self._compile_external(problem.system, external) if external else None
        self.parameters_ = problem.initial_parameters.copy()
        self.time_, self.timestep_, self.start_step_ = 0.0, self.dt, 0
        self.lammps_ = lammps(cmdargs=_LAMMPS_ARGUMENTS)
        self.released_ = False
        self.lammps_.commands_list(self._box_commands(problem, problem.initial_state))
        for element in self.atom_elements_:
            self.lammps_.command(
                f'mass {self.atom_numbers_[element.path]} {self._mass(problem, self.parameters_, element)}',
            )
        self._create_atoms(problem.initial_state)
        self.lammps_.commands_list([f'group group{number} type {number}' for number in self.atom_numbers_.values()])
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

    def _create_atoms(self, x: Array) -> None:
        """Create one LAMMPS atom per member, with ids in state vector order."""
        count = len(self.atom_types_)
        positions, velocities = np.zeros((count, 3)), np.zeros((count, 3))
        positions[:, : self.n_dim_], velocities[:, : self.n_dim_] = x[self.position_indices_], x[self.velocity_indices_]
        ids = list(range(1, count + 1))
        self.lammps_.create_atoms(
            count,
            ids,
            self.atom_types_.tolist(),
            positions.ravel().tolist(),
            velocities.ravel().tolist(),
        )

    def _update_parameters(self, p: Array) -> None:
        """Reissue the masses, pair coefficients, and fixes after a parameter change."""
        problem = self.problem_
        self.parameters_ = p.copy()
        for element in self.atom_elements_:
            self.lammps_.command(f'mass {self.atom_numbers_[element.path]} {self._mass(problem, p, element)}')
        self.lammps_.commands_list(self._pair_commands(problem, p))
        fixes = self._fix_commands(problem, p)
        self.lammps_.commands_list([f'unfix {command.split()[1]}' for command in fixes])
        self.lammps_.commands_list(fixes)
        if self.external_ is not None:
            self.lammps_.set_fix_external_callback('external', self._external_callback(), None)

    def _local_atoms(self) -> tuple[npt.NDArray[np.intp], Array, Array]:
        """Return the state order, positions, and velocities of the local atoms, without periodic ghost atoms."""
        numpy, count = self.lammps_.numpy, int(self.lammps_.extract_global('nlocal'))
        order = np.asarray(numpy.extract_atom('id')[:count], dtype=np.intp) - 1
        return order, numpy.extract_atom('x')[:count], numpy.extract_atom('v')[:count]

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
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
        order, positions, velocities = self._local_atoms()
        positions[:, : self.n_dim_] = x[self.position_indices_[order]]
        velocities[:, : self.n_dim_] = x[self.velocity_indices_[order]]
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
        state[self.position_indices_[order]] = positions[:, : self.n_dim_]
        state[self.velocity_indices_[order]] = velocities[:, : self.n_dim_]
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
    def _state_indices(problem: ProblemSpec, name: str) -> npt.NDArray[np.intp]:
        """Return the flat indices of a vector state as one row per member."""
        entry = problem.states.entry(name)
        return (entry.start + np.arange(entry.size)).reshape(entry.rows, entry.columns)

    def _set_atoms(self, problem: ProblemSpec) -> None:
        """Set the atom types and state indices of the elements and groups of points of a problem."""
        self.atom_elements_ = list(problem.elements)
        self.atom_numbers_: dict[str, int] = {}
        positions, velocities, types = [], [], []
        self.n_dim_ = 0
        for number, element in enumerate(problem.elements, start=1):
            self.atom_numbers_[element.path] = number
            name = self._position_name(element)
            member_positions = LammpsSolver._state_indices(problem, element.states[name])
            positions.append(member_positions)
            velocities.append(LammpsSolver._state_indices(problem, element.states[f'{name}_d1']))
            types.append(np.full(len(member_positions), number, dtype=np.intp))
            self.n_dim_ = member_positions.shape[1]
        self.position_indices_ = np.concatenate(positions)
        self.velocity_indices_ = np.concatenate(velocities)
        self.atom_types_ = np.concatenate(types)

    @staticmethod
    def _role_path(item: CompiledInteractionSpec, role: str) -> str:
        """Return the path of the group or element a role of an interaction binds to."""
        path = item.roles[role]
        if not isinstance(path, str):
            error_msg = f'Interaction {item.path} binds the role {role!r} to an element set, which LAMMPS does not run.'
            raise BackendError(error_msg)
        return path

    @staticmethod
    def _parameter(problem: ProblemSpec, p: Array, path: str, name: str) -> Array:
        """Return the value of a parameter of an element, group, or interaction."""
        return np.atleast_1d(np.asarray(problem.parameters.read(p, f'{path}.{name}'), dtype=float))

    @staticmethod
    def _divisors(problem: ProblemSpec, element: CompiledElementSpec) -> set[str]:
        """Return the names that the divisors of the interactions on the members of an element or group give."""
        return {
            divisor
            for item in problem.interactions
            if isinstance(divisor := item.interaction.get_params(deep=False).get('divisor'), str)
            and element.path in item.roles.values()
        }

    @staticmethod
    def _mass(problem: ProblemSpec, p: Array, element: CompiledElementSpec) -> float:
        """Return the mass of the members of an element or group, the parameter their divisor names, or 1."""
        divisors = LammpsSolver._divisors(problem, element)
        if not divisors:
            return 1.0
        return float(LammpsSolver._parameter(problem, p, element.path, divisors.pop())[0])

    @staticmethod
    def _is_potential(item: CompiledInteractionSpec) -> bool:
        """Return whether an interaction is a pair coupling through a pair potential."""
        return isinstance(item.interaction, PairCouplingInteraction) and item.interaction.potential is not None

    @staticmethod
    def _find_position(element: CompiledElementSpec) -> str | None:
        """Return the name of the vector position state of order 2 of the members of an element, or `None`."""
        element_type = element.element.element if element.members is not None else type(element.element)
        declarations = declarations_of(element_type, VariableKind.STATE)
        return next(
            (
                name
                for name, declaration in declarations.items()
                if declaration.vector and declaration.position and declaration.order == POSITION_ORDER
            ),
            None,
        )

    @staticmethod
    def _position_name(element: CompiledElementSpec) -> str:
        """Return the name of the vector position state of order 2 of the members of an element.

        Args:
            element:
                The compiled element.

        Returns:
            name:
                The name of the position state.

        Raises:
            BackendError:
                The members declare no such state.
        """
        name = LammpsSolver._find_position(element)
        if name is None:
            error_msg = f'LAMMPS runs points, but the members of {element.path} declare no vector position of order 2.'
            raise BackendError(error_msg)
        return name

    @staticmethod
    def _compile_external(system: System | None, keep: set[str], backend: NumpyBackend | None = None) -> ProblemSpec:
        """Return the problem of the interactions of a system that LAMMPS calls back, which only evaluates forces.

        Args:
            system:
                The system of the problem.

            keep:
                The paths of the interactions to keep.

            backend:
                The backend of the problem, or `None` for the backend that suits the system.

        Returns:
            problem:
                The compiled problem of the kept interactions.

        Raises:
            SolverError:
                The problem has no system, or the system does not compile to a problem of the compiler.
        """
        if system is None:
            error_msg = 'LammpsSolver evaluates external interactions of a system, but the problem has none.'
            raise SolverError(error_msg)
        problem = compile_system(LammpsSolver._filter_system(system, keep), backend=backend)
        if not isinstance(problem, ProblemSpec):
            error_msg = f'The external interactions compile to a {type(problem).__name__}, not a ProblemSpec.'
            raise SolverError(error_msg)
        return problem

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
    def _check_masses(problem: ProblemSpec, element: CompiledElementSpec) -> list[str]:
        """Return the problems of the mass of an element or group: several divisors, or several values."""
        divisors = LammpsSolver._divisors(problem, element)
        if len(divisors) > 1:
            return [f'element {element.path}, whose interactions divide by {", ".join(sorted(divisors))}']
        if not divisors:
            return []
        name = next(iter(divisors))
        if name not in element.parameters:
            return [f'element {element.path}, whose divisor {name} is not a parameter']
        masses = LammpsSolver._parameter(problem, problem.initial_parameters, element.path, name)
        if not np.all(masses == masses[0]):
            return [f'group {element.path}, whose members have different values of {name}']
        return []

    @staticmethod
    def _check_elements(problem: ProblemSpec) -> list[str]:
        """Return the elements that are not points in two or three dimensions with one mass per group."""
        unsupported = []
        for element in problem.elements:
            name = LammpsSolver._find_position(element)
            if name is None:
                unsupported.append(f'element {element.path}, which declares no vector position state of order 2')
                continue
            n_dim = problem.states.entry(element.states[name]).columns
            if n_dim not in _SUPPORTED_DIMENSIONS:
                unsupported.append(f'element {element.path} in {n_dim} dimensions, while LAMMPS runs 2 or 3')
            unsupported += LammpsSolver._check_masses(problem, element)
        return unsupported

    @staticmethod
    def _box(problem: ProblemSpec) -> ContinuousSpace | None:
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
        printed = ExpressionPrinter('lepton', shared=False).check().prepare().run([lepton]).code_
        if not isinstance(printed, tuple):
            error_msg = f'The Lepton printer returned a {type(printed).__name__}, not printed code.'
            raise TypeError(error_msg)
        return printed[1][0]
