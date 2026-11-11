"""Advance the partitions of a co-simulation with their own solvers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from sage.all__sagemath_symbolics import SR, ZZ, Expression
from sklearn.base import clone

from ...components.laws import (
    BoundaryLaw,
    CrossingLaw,
    EquationLaw,
    FluxLaw,
    JumpLaw,
    LinearProgramLaw,
    LocalRuleLaw,
    MatchLaw,
    NoiseLaw,
    RateLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
    TransitionLaw,
    UpdateLaw,
)
from ...components.spaces import (
    CayleySpace,
    ContinuousSpace,
    ContinuousTime,
    DiscreteTime,
    LatticeSpace,
    MeshSpace,
    NetworkSpace,
)
from ...enums import FormKind, NoiseKind, OrderKind
from ...errors import ContextError, SolverError, ValidationError
from ...specs import AgentSnapshotSpec, AgentStateSpec, EntrySpec, StepSpec, SupportSpec
from ...specs._held_reduction_spec import _HeldReductionSpec
from ...types import Array
from ..backends import NumbaBackend, NumpyBackend
from ..compile._observables import ObservableEvaluator
from ..compile._pass_compiler import PassCompiler
from ..problems import ProblemSpec
from ..problems._expression_printer import ExpressionPrinter
from ..problems._partition_spec import PartitionSpec
from ..problems._reduction_kernel import ReductionKernel
from ._base import BaseSolver
from ._defaults import DEFAULT_SOLVERS
from ._solver import Solver

STEPWISE_FORMS = (FormKind.DISCRETE, FormKind.AGENT)
TOLERANCE = 1e-9
SCHEMES = ('gauss-seidel', 'jacobi')


class CoSimulationSolver(Solver):
    """A solver that advances every partition of a co-simulation with its own solver.

    Every partition is advanced to each synchronisation point. Between synchronisation points a partition holds the
    states of the other partitions at their last values, so a split adds an error of the order of `sync_dt`. A discrete
    partition that holds a linear program takes one step per synchronisation interval, whatever its length, so the
    program solves at every synchronisation point. A law of one partition that reads a reduction of the agents of
    another, such as `count(s.farmers)`, reads the value the scheduler computes at every synchronisation point and holds
    until the next. Gauss-Seidel advances the partitions in system order, each reading the values the partitions before
    it already advanced. Jacobi advances every partition from the same values.

    Choose it when components of a system need different solvers, such as agents beside a field. It appears in many
    fields, such as cars on a road network beside their emissions, and cells beside the nutrients they eat.

    Args:
        solvers:
            The solver of a partition by its name, with the default solver of its form for the others.

        sync_dt:
            The time between synchronisation points, which must divide the frame duration, or `None` for one
            synchronisation per frame.

        scheme:
            `'gauss-seidel'` or `'jacobi'`.

        split:
            Whether to split a problem of one form into one partition per subsystem.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Attributes:
        steppers_:
            The stepper of every partition by partition name.

    Examples:
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.engine.solvers import CoSimulationSolver
        >>> system = CompositeSystem(
        ...     systems=[
        ...         CompositeSystem(
        ...             'a',
        ...             elements=[PopulationElement('x', x=1)],
        ...             interactions=[ExponentialGrowthInteraction('g', target='x')],
        ...         ),
        ...         CompositeSystem(
        ...             'b',
        ...             elements=[PopulationElement('y', x=1)],
        ...             interactions=[ExponentialGrowthInteraction('g', target='y')],
        ...         ),
        ...     ],
        ... )
        >>> trajectory = simulate(system, until=1, solver=CoSimulationSolver(split=True), dt=0.5)
        >>> [round(float(trajectory.get_values(name)[-1]), 2) for name in ('a/x.x', 'b/y.x')]
        [2.72, 2.72]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime, DiscreteTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={
                RateLaw,
                FluxLaw,
                NoiseLaw,
                EquationLaw,
                JumpLaw,
                LinearProgramLaw,
                CrossingLaw,
                UpdateLaw,
                TransitionLaw,
                RuleLaw,
                LocalRuleLaw,
                SampleLaw,
                MatchLaw,
                TransferLaw,
                BoundaryLaw,
            },
            noise=frozenset(NoiseKind) - {NoiseKind.FRACTIONAL},
            orders=frozenset(OrderKind),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.COUPLED

    def __init__(
        self,
        solvers: Mapping[str, BaseSolver] | None = None,
        sync_dt: float | None = None,
        scheme: str = 'gauss-seidel',
        split: bool = False,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.solvers = solvers
        self.sync_dt = sync_dt
        self.scheme = scheme
        self.split = split
        self.random_state = random_state
        self.until = until

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Partition the problem when asked, and build and seed the partition steppers.

        It first raises an error for an unknown coupling scheme.

        There is one stepper per partition.

        Args:
            context:
                The co-simulation problem, or a problem of one form with `split=True`.

        Raises:
            ContextError:
                No problem is passed.

            SolverError:
                The problem is not a co-simulation and `split` is false. The scheme is unknown.

            TypeError:
                The compiler of the partitions does not return a problem of the compiler.
        """
        if self.scheme not in SCHEMES:
            error_msg = f'scheme = {self.scheme!r} must be one of {", ".join(SCHEMES)}.'
            raise SolverError(error_msg)
        if context is None:
            error_msg = 'A solver prepares on a compiled problem. Pass the problem to `prepare`.'
            raise ContextError(error_msg)
        problem = context
        initial_state = problem.initial_state.copy()
        if problem.form != FormKind.COUPLED:
            if not self.split or problem.system is None:
                error_msg = (
                    f'CoSimulationSolver runs co-simulations, but the problem has the {problem.form} form. Pass '
                    '`split=True` to split it into one partition per subsystem.'
                )
                raise SolverError(error_msg)
            backend = NumbaBackend() if problem.backend == NumbaBackend._get_name() else NumpyBackend()
            compiler = PassCompiler(backend=backend, partition='subsystems')
            partitioned = compiler.prepare(problem.system).run().problem_
            if not isinstance(partitioned, ProblemSpec):
                error_msg = f'The compiler returned a {type(partitioned).__name__}, not a ProblemSpec.'
                raise TypeError(error_msg)
            problem = partitioned
        self.problem_, self.rng_ = problem, self._get_generator()
        generators = self.rng_.spawn(len(problem.partitions))
        given = dict(self.solvers or {})
        self.steppers_: dict[str, Solver] = {}
        for partition, generator in zip(problem.partitions, generators, strict=True):
            solver = given.get(partition.name)
            form = partition.problem.form
            if solver is None and not isinstance(form, FormKind):
                error_msg = f'Partition {partition.name} has the {form} form, which has no default solver. Give one.'
                raise ValidationError(error_msg)
            stepper = clone(solver) if solver is not None else DEFAULT_SOLVERS[FormKind(form)]()
            stepper.set_params(random_state=generator, until=self.until).prepare(partition.problem)
            self.steppers_[partition.name] = stepper
        self._held = [
            (path, reduction, self._get_reducer(reduction, problem.time))
            for path, reduction in ObservableEvaluator._get_held_reductions_of(problem)
        ]
        self.initial_state_ = initial_state
        self._update_held(0.0, initial_state)
        self.events_, self.stopped_at_ = [], None
        self._collect()

    @staticmethod
    def _get_reducer(
        reduction: _HeldReductionSpec,
        time: Expression,
    ) -> Callable[[float, Sequence[Array], Sequence[object] | Array], float]:
        """Return the function that computes a held reduction.

        The function takes time, the values of the variables, and the members.

        Args:
            reduction:
                The held reduction.

            time:
                The time symbol.

        Returns:
            reducer:
                The function.
        """
        reduce = reduction.reduce
        if reduction.argument is None:

            def reduce_values_of(t: float, columns: Sequence[Array], members: Sequence[object] | Array) -> float:
                del t
                return float(
                    np.real(ReductionKernel._reduce_values(np.asarray(columns[0] if columns else members), reduce)),
                )

            return reduce_values_of
        condition = SR(1) if reduction.condition is None else reduction.condition
        printer = ExpressionPrinter(time=time, compiled=True).prepare(list(reduction.symbols))
        function = printer.run([reduction.argument, condition]).code_
        if not callable(function):
            error_msg = f'The printer returned a {type(function).__name__}, not a compiled function.'
            raise TypeError(error_msg)

        def reduce_expression(t: float, columns: Sequence[Array], members: Sequence[object] | Array) -> float:
            size = len(members)
            values = function(t, *columns)
            if not isinstance(values, list):
                error_msg = f'The compiled function returned a {type(values).__name__}, not a list of values.'
                raise TypeError(error_msg)
            argument, selected = (np.broadcast_to(np.asarray(item, dtype=float), (size,)) for item in values)
            return float(np.real(ReductionKernel._reduce_values(argument[selected.astype(bool)], reduce)))

        return reduce_expression

    def _get_intervals(self, t: float, until: float) -> list[tuple[float, float]]:
        """Return the synchronisation intervals of a frame.

        `sync_dt` is checked to divide the frame.
        """
        duration = until - t
        interval = duration if self.sync_dt is None else self.sync_dt
        count = round(duration / interval) if interval > 0 else 0
        stepwise = any(
            partition.problem.form in STEPWISE_FORMS
            and not (isinstance(partition.problem, ProblemSpec) and partition.problem.programs)
            for partition in self.problem_.partitions
        )
        if count < 1 or abs(count * interval - duration) > TOLERANCE * max(1.0, duration):
            error_msg = f'sync_dt = {self.sync_dt!r} must divide the frame duration {duration!r}.'
            raise ValidationError(error_msg)
        if stepwise and not EntrySpec._contains(ZZ, round(interval, 9)):
            error_msg = f'sync_dt = {interval!r} must be a whole number of steps for discrete and agent partitions.'
            raise ValidationError(error_msg)
        return [
            (t + index * interval, until if index == count - 1 else t + (index + 1) * interval)
            for index in range(count)
        ]

    def _get_size_of(self, group: str) -> int:
        """Return the number of members of an element group."""
        return next(item.members or 1 for item in self.problem_.elements if item.path == group)

    def _update_held(self, t: float, x: Array) -> None:
        """Write the current value of every held reduction into the state vector."""
        states = self.problem_.states
        for path, reduction, reducer in self._held:
            agents = self._get_agents_of(reduction.group)
            members: Sequence[object] | Array
            if agents is not None:
                members = agents
                columns = [
                    np.asarray([getattr(agent, name) for agent in agents], dtype=float) for name in reduction.variables
                ]
            else:
                columns = [np.ravel(states.read(x, f'{reduction.group}.{name}')) for name in reduction.variables]
                members = columns[0] if columns else np.zeros(self._get_size_of(reduction.group))
            x[states.get_entry(f'{path}.value').indices] = reducer(t, columns, members)

    def _get_agents_of(self, group: str) -> list[object] | None:
        """Return the agents of an agent group from the partition that holds it, or `None`."""
        for stepper in self.steppers_.values():
            agents = stepper.agents_.get(group)
            if agents is not None:
                return agents
        return None

    def _advance_partition(self, partition: PartitionSpec, source: Array, p: Array, start: float, end: float) -> Array:
        """Advance one partition and return its new state vector.

        It starts from the values of a source state vector.
        """
        stepper = self.steppers_[partition.name]
        problem = partition.problem
        boundary = problem.boundary if isinstance(problem, ProblemSpec) else None
        if (
            isinstance(problem, ProblemSpec)
            and problem.programs
            and problem.form in STEPWISE_FORMS
            and problem.update is not None
        ):
            drawn = (stepper.rng_,) if problem.stochastic else ()
            state = problem.update(start, source.copy(), p, *drawn)
            return state if boundary is None else boundary(end, state, p)
        if stepper._reads_inputs():
            stepper.inputs_ = self.problem_.states.get_values(source)
        state = stepper.run(StepSpec(start, source.copy(), p, end)).state_
        if boundary is not None:
            state = boundary(end, state, p)
        return state

    def _add_deposits(self, partition: PartitionSpec, state: Array) -> None:
        """Add the amounts that the agents of a partition deposited into fields.

        They are the amounts since the last synchronisation.
        """
        deposits = getattr(self.steppers_[partition.name], 'deposits_', None)
        if deposits is not None and deposits.any():
            state += deposits
            deposits[:] = 0.0

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Advance every partition to `until`.

        Values are exchanged at every synchronisation point.

        Args:
            t:
                The current time.

            x:
                The current state vector of the whole system.

            p:
                The parameter vector.

            until:
                The time to advance to.

        Returns:
            state:
                The state vector at time `until`.

        Raises:
            ValidationError:
                The synchronisation interval does not divide the frame, or is not a whole number of steps for a
                discrete or agent partition.
        """
        if until <= t:
            return x
        state = x.copy()
        self._update_held(t, state)
        for start, end in self._get_intervals(t, until):
            base = state.copy()
            for partition in self.problem_.partitions:
                source = state if self.scheme == 'gauss-seidel' else base
                advanced = self._advance_partition(partition, source, p, start, end)
                state[partition.owned] = advanced[partition.owned]
                self._add_deposits(partition, state)
                if self.scheme == 'gauss-seidel':
                    self._update_held(end, state)
            self._update_held(end, state)
        return state

    def _collect(self) -> None:
        """Set the agents and reporters of every agent partition, merged.

        Without agents, there is no snapshot.
        """
        groups: dict[str, AgentSnapshotSpec] = {}
        reporters: dict[str, float] = {}
        agents: dict[str, list[object]] = {}
        for stepper in self.steppers_.values():
            external = stepper.snapshot_
            if isinstance(external, AgentStateSpec):
                groups.update(external.groups)
                reporters.update(external.reporters)
            agents.update(stepper.agents_)
        self.snapshot_ = AgentStateSpec(groups, reporters) if groups or reporters else None
        self.agents_ = agents

    def _release(self) -> None:
        """Release what the stepper of every partition acquired."""
        for partition in self.problem_.partitions:
            problem = partition.problem
            step = StepSpec(0.0, problem.initial_state, problem.initial_parameters, 0.0)
            self.steppers_[partition.name].run(step, final=True)
