"""Advance the partitions of a co-simulation with their own solvers and exchange values at synchronisation points."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import numpy as np
from sklearn.base import clone

from ..base import (
    ComponentKind,
    Domain,
    DomainKind,
    Form,
    LawKind,
    SolverError,
    SpaceKind,
    Supports,
    TimeKind,
    ValidationError,
)
from ..compile import Partition, Problem, Vector, partition_system
from ..model import Aggregate, reduce_values
from ._agents import AgentState, MesaStepper
from ._base import Stepper, register_solver, resolve_solver

STEPWISE_FORMS = (Form.DISCRETE, Form.AGENT)
TOLERANCE = 1e-9
SCHEMES = ('gauss-seidel', 'jacobi')


class CoSimulation(Stepper):
    """Scheduler that advances every partition of a co-simulation to each synchronisation point with its own solver.

    Between synchronisation points a partition holds the states of the other partitions at their last values, so a
    split adds an error of the order of `sync_dt`. Gauss-Seidel advances the partitions in system order, each reading
    the values the partitions before it already advanced. Jacobi advances every partition from the same values.

    Args:
        solvers:
            A mapping from partition name to a solver name or instance, with the default solver of each form for the
            others.

        sync_dt:
            The time between synchronisation points, which must divide the frame duration, or `None` for one
            synchronisation per frame.

        scheme:
            `'gauss-seidel'` or `'jacobi'`.

        split:
            Whether to split a problem of one form into one partition per subsystem.

    Attributes:
        steppers_:
            The stepper of every partition by partition name.

    Examples:
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowth, Population
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.core.solvers import CoSimulation
        >>> system = System(
        ...     components=[
        ...         System('a', components=[Population('x', n=1)], interactions=[ExponentialGrowth('g', target='x')]),
        ...         System('b', components=[Population('y', n=1)], interactions=[ExponentialGrowth('g', target='y')]),
        ...     ],
        ... )
        >>> simulation = Simulation(system, solver=CoSimulation(split=True), dt=0.5).run(until=1)
        >>> sorted(simulation.stepper_.steppers_)
        ['a', 'b']
    """

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.CONTINUOUS, TimeKind.DISCRETE},
        space=set(SpaceKind),
        laws={
            LawKind.RATE,
            LawKind.NOISE,
            LawKind.EQUATION,
            LawKind.JUMP,
            LawKind.EVENT,
            LawKind.UPDATE,
            LawKind.TRANSITION,
            LawKind.RULE,
            LawKind.LOCAL_RULE,
            LawKind.REWRITE,
            LawKind.SAMPLE,
            LawKind.MATCH,
            LawKind.TRANSFER,
            LawKind.BOUNDARY,
        },
        domains=set(DomainKind),
        components={
            ComponentKind.ELEMENT,
            ComponentKind.GROUP,
            ComponentKind.VARIABLE_GROUP,
            ComponentKind.SITE_GROUP,
            ComponentKind.SYSTEM,
        },
    )

    form: ClassVar[Form] = Form.COUPLED

    def __init__(
        self,
        solvers: Mapping[str, str | Stepper] | None = None,
        sync_dt: float | None = None,
        scheme: str = 'gauss-seidel',
        split: bool = False,
    ) -> None:
        self.solvers = solvers
        self.sync_dt = sync_dt
        self.scheme = scheme
        self.split = split

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Partition the problem when asked, and build and seed the stepper of every partition.

        Args:
            problem:
                The co-simulation problem, or a problem of one form with `split=True`.

            rng:
                The random number generator from which every partition gets its own generator.

        Raises:
            SolverError:
                The problem is not a co-simulation and `split` is false, or the scheme is unknown.
        """
        if self.scheme not in SCHEMES:
            error_msg = f'scheme = {self.scheme!r} must be one of {", ".join(SCHEMES)}.'
            raise SolverError(error_msg)
        if problem.form is not Form.COUPLED:
            if not self.split or problem.system is None:
                error_msg = (
                    f'CoSimulation runs co-simulations, but the problem has the {problem.form.value} form. Pass '
                    '`split=True` to split it into one partition per subsystem.'
                )
                raise SolverError(error_msg)
            problem = partition_system(problem.system, problem.backend)
        self.problem_, self.rng_ = problem, rng
        generators = rng.spawn(len(problem.partitions))
        given = dict(self.solvers or {})
        self.steppers_: dict[str, Stepper] = {}
        for partition, generator in zip(problem.partitions, generators, strict=True):
            solver = given.get(partition.name, 'auto')
            stepper = resolve_solver(partition.problem, clone(solver) if isinstance(solver, Stepper) else solver)
            stepper.initialize(partition.problem, generator)
            self.steppers_[partition.name] = stepper
        self._aggregates = [element for element in problem.elements if isinstance(element.element, Aggregate)]

    def _intervals(self, t: float, until: float) -> list[tuple[float, float]]:
        """Return the synchronisation intervals of a frame after checking that `sync_dt` divides it."""
        duration = until - t
        interval = duration if self.sync_dt is None else self.sync_dt
        count = round(duration / interval) if interval > 0 else 0
        stepwise = any(partition.problem.form in STEPWISE_FORMS for partition in self.problem_.partitions)
        if count < 1 or abs(count * interval - duration) > TOLERANCE * max(1.0, duration):
            error_msg = f'sync_dt = {self.sync_dt!r} must divide the frame duration {duration!r}.'
            raise ValidationError(error_msg)
        if stepwise and not Domain.INTEGER.contains(round(interval, 9)):
            error_msg = f'sync_dt = {interval!r} must be a whole number of steps for discrete and agent partitions.'
            raise ValidationError(error_msg)
        return [
            (t + index * interval, until if index == count - 1 else t + (index + 1) * interval)
            for index in range(count)
        ]

    def _update_aggregates(self, x: Vector) -> None:
        """Write the current value of every aggregate into the state vector."""
        problem = self.problem_
        for element in self._aggregates:
            aggregate = cast('Aggregate', element.element)
            scope = element.path.rpartition('/')[0]
            group = f'{scope}/{aggregate.group}' if scope else aggregate.group
            agents = self.agents_of(group)
            values: Any
            if agents is not None:
                values = [getattr(agent, aggregate.variable) for agent in agents] if aggregate.variable else agents
            else:
                values = np.ravel(problem.states.read(x, f'{group}.{aggregate.variable or "value"}'))
            entry = problem.states.entry(f'{element.path}.value')
            x[entry.indices] = float(np.real(reduce_values(values, aggregate.reduce)))

    def agents_of(self, group: str) -> list[Any] | None:
        """Return the agents of an agent group from the partition that holds it, or `None`.

        Args:
            group:
                The path of the agent group.

        Returns:
            agents:
                The agents alive now, or `None`.
        """
        for stepper in self.steppers_.values():
            agents = stepper.agents_of(group)
            if agents is not None:
                return agents
        return None

    def _advance_partition(self, partition: Partition, source: Vector, p: Vector, start: float, end: float) -> Vector:
        """Advance one partition from the values of a source state vector and return its new state vector."""
        stepper = self.steppers_[partition.name]
        if isinstance(stepper, MesaStepper):
            stepper.inputs_ = self.problem_.states.values(source)
        state = stepper.advance(start, source.copy(), p, end)
        if partition.problem.boundary is not None:
            state = partition.problem.boundary(end, state, p)
        return state

    def _add_deposits(self, partition: Partition, state: Vector) -> None:
        """Add the amounts that the agents of a partition deposited into fields since the last synchronisation."""
        deposits = getattr(self.steppers_[partition.name], 'deposits_', None)
        if deposits is not None and deposits.any():
            state += deposits
            deposits[:] = 0.0

    def initial_state(self, x: Vector) -> Vector:
        """Return the initial state vector with every aggregate computed from its group.

        Args:
            x:
                The initial state vector of the problem.

        Returns:
            state:
                The state vector of the first frame.
        """
        state = x.copy()
        self._update_aggregates(state)
        return state

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Advance every partition to `until`, exchanging values at every synchronisation point.

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
        self._update_aggregates(state)
        for start, end in self._intervals(t, until):
            base = state.copy()
            for partition in self.problem_.partitions:
                source = state if self.scheme == 'gauss-seidel' else base
                advanced = self._advance_partition(partition, source, p, start, end)
                state[partition.owned] = advanced[partition.owned]
                self._add_deposits(partition, state)
                if self.scheme == 'gauss-seidel':
                    self._update_aggregates(state)
            self._update_aggregates(state)
        return state

    def snapshot(self) -> AgentState | None:
        """Return the agents and reporters of every agent partition.

        Returns:
            state:
                The merged agents and reporters, or `None` when no partition has agents.
        """
        groups: dict[str, Any] = {}
        reporters: dict[str, float] = {}
        for stepper in self.steppers_.values():
            external = stepper.snapshot()
            if isinstance(external, AgentState):
                groups.update(external.groups)
                reporters.update(external.reporters)
        return AgentState(groups, reporters) if groups or reporters else None


register_solver('co-simulation', CoSimulation, default=True)
