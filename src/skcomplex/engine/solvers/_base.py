"""Define the base of the solvers, which advance a compiled problem through time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping

import numpy as np

from ...base import BaseConfig
from ...errors import SolverError, ValidationError
from ...specs import StepSpec, SupportSpec
from ...specs._base_record_spec import BaseRecordSpec
from ...types import Array, FormKind
from ..problems import BaseProblemSpec, ProblemSpec
from ..types import BoundaryFunction, InterpolatingStepFunction
from ._event_locator import EventLocator
from ._history import _RESTART, History


class BaseSolver[P: BaseProblemSpec](BaseConfig[P, StepSpec]):
    """Base of the solvers, which advance a problem of one form through time.

    A solver stores its settings in its constructor, among them the random state of stochastic solvers and the end
    time of the run it prepares for. `prepare(problem)` prepares it for a problem and `run(step)` advances the state
    of a `StepSpec` from its time to its end time and keeps the state vector there in `state_`. The last call
    `run(step, final=True)` releases what preparing acquired, such as an external process.

    Subclasses implement the hooks `_get_form`, the form they solve, `_get_support`, what they solve, and `_advance`,
    and solvers that keep events, agents, or other state outside the state vector also implement `_collect`. Solvers
    that also solve other forms list them in `_get_forms`. Solvers that treat diffusion implicitly answer
    `_is_implicit`, so they do not warn about the stability limit of explicit steps. Solvers that fire renewal laws
    answer `_is_renewing`, solvers that integrate fractional noise `_is_fractional`, and solvers that read external
    inputs `_reads_inputs`. A solver is generic over the problem it advances, so one written for the compiler's problem
    reads its fields.

    Args:
        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped,
            which solvers that plan or draw a whole run read.

    Attributes:
        problem_:
            The problem the solver is prepared to.

        rng_:
            The random number generator of stochastic solvers.

        initial_state_:
            The state vector of the first frame, which solvers that derive states fill in.

        events_:
            The events of the last run, in the order they happened.

        stopped_at_:
            The time at which a terminal event ended the run, or `None` while it goes on.

        snapshot_:
            The state the solver keeps outside the state vector, such as agents, after the last run, or `None`.

        agents_:
            The views of the agents alive after the last run, as rules receive them, by path of their agent group.

        state_:
            The state vector at the end time of the last step.
    """

    problem_: P
    rng_: np.random.Generator
    initial_state_: Array
    events_: list[BaseRecordSpec]
    stopped_at_: float | None
    snapshot_: object | None
    agents_: Mapping[str, list[object]]
    state_: Array

    def _check(self) -> None:
        """Accept every setting, since a solver has none to validate; a solver with settings validates them."""

    def __init__(self, *, random_state: int | np.random.Generator | None = None, until: float = math.inf) -> None:
        self.random_state = random_state
        self.until = until

    @classmethod
    def _get_form(cls) -> FormKind | str:
        """Return the form the solver solves.

        Returns:
            form:
                The form.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @classmethod
    def _get_forms(cls) -> frozenset[FormKind | str]:
        """Return the other forms the solver also solves, none by default."""
        return frozenset()

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves.

        Returns:
            support:
                What the solver solves.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @classmethod
    def _is_stepwise(cls) -> bool:
        """Return whether the solver advances in whole steps, so frames are a positive integer number of steps apart.

        The solvers of discrete maps, agents, lattices, and Monte Carlo sampling are stepwise through their forms; a
        solver of a form defined outside the engine, such as rewriting, says so here.
        """
        return False

    @classmethod
    def _is_implicit(cls) -> bool:
        """Return whether the solver treats diffusion implicitly, so explicit stability limits do not apply."""
        return False

    @classmethod
    def _is_renewing(cls) -> bool:
        """Return whether the solver fires renewal laws."""
        return False

    @classmethod
    def _is_fractional(cls) -> bool:
        """Return whether the solver integrates fractional noise."""
        return False

    @classmethod
    def _reads_inputs(cls) -> bool:
        """Return whether the solver reads the inputs of other partitions."""
        return False

    def _prepare(self, context: P | None) -> None:
        """Prepare the solver for a problem, with the random number generator of its random state.

        Args:
            context:
                The problem.

        Raises:
            ValidationError:
                No problem is passed.

            SolverError:
                The problem has a form the solver does not solve.
        """
        if context is None:
            error_msg = 'A solver prepares on a compiled problem. Pass the problem to `prepare`.'
            raise ValidationError(error_msg)
        problem = context
        form = self._get_form()
        if problem.form != form and problem.form not in self._get_forms():
            error_msg = f'{type(self).__name__} solves the {form} form, but the problem has the {problem.form} form.'
            raise SolverError(error_msg)
        self.problem_ = problem
        self.rng_ = np.random.default_rng(self.random_state)
        self.initial_state_ = problem.initial_state.copy()
        self.events_ = []
        self.stopped_at_ = None
        self.snapshot_ = None
        self.agents_ = {}

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Advance the state from time `t` to time `until`.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to, later than `t`.

        Returns:
            state:
                The state vector at time `until`.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def _collect(self) -> None:
        """Set the events, the stop time, the snapshot, and the agents after a run; the base keeps none."""
        self.events_ = []

    def _run(self, inputs: StepSpec | None) -> None:
        """Advance the state of the step to its end time, keep it in `state_`, and collect what the run produced.

        A step that does not go forward in time keeps its state unchanged, so a final run of no length releases the
        solver.

        Args:
            inputs:
                The step.

        Raises:
            ValidationError:
                No step is passed.
        """
        if inputs is None:
            error_msg = 'A solver runs on a step. Pass the step to `run`.'
            raise ValidationError(error_msg)
        t, x, p, until = inputs.t, inputs.x, inputs.p, inputs.until
        self.state_ = self._advance(t, x, p, until) if until > t else x
        self._collect()

    @staticmethod
    def _is_hybrid(problem: ProblemSpec, t: float, until: float) -> bool:
        """Check whether a problem needs the hybrid path between two times: delays, events, or breakpoints.

        Args:
            problem:
                The compiled problem.

            t:
                The start time.

            until:
                The end time.

        Returns:
            hybrid:
                Whether the problem has delays, events, or breakpoints up to and at the end.
        """
        return bool(problem.delays or problem.events or problem.breakpoints.between(t, math.nextafter(until, math.inf)))

    @staticmethod
    def _reject_hybrid(problem: ProblemSpec, solver: str) -> None:
        """Raise an error when a solver without a hybrid path receives delays or events.

        Args:
            problem:
                The compiled problem.

            solver:
                The name of the solver.

        Raises:
            SolverError:
                The problem has delays or events.
        """
        if problem.delays or problem.events:
            error_msg = (
                f'{solver} does not integrate delays or events. Use ScipyODESolver or RK4Solver for ODEs and '
                'EulerMaruyamaSolver, MilsteinSolver, or HeunSolver for SDEs.'
            )
            raise SolverError(error_msg)

    @staticmethod
    def _advance_fixed_steps(
        t: float,
        x: Array,
        p: Array,
        until: float,
        *,
        dt: float,
        locator: EventLocator,
        history: History | None,
        bounded: BoundaryFunction,
        take: InterpolatingStepFunction,
    ) -> Array:
        """Integrate with fixed steps that stop at breakpoints, record every step for delays, and locate events in it.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to.

            dt:
                The largest step.

            locator:
                The event locator of the run.

            history:
                The history that delayed laws read, or `None` without delays.

            bounded:
                The boundary laws `bounded(t, x, p)`.

            take:
                The step `take(t, x, h, ceiling)` of a scheme, which returns the state after a step of length `h` and
                its interpolant over the step, and reads inputs no later than `ceiling`.

        Returns:
            state:
                The state vector at time `until`, or at a terminal event.

        Raises:
            SolverError:
                The step exceeds the smallest delay.
        """
        problem = locator.problem
        smallest = EventLocator._find_smallest_delay(problem)
        if dt > smallest:
            error_msg = f'dt = {dt} exceeds the smallest delay {smallest}. Take a smaller step.'
            raise SolverError(error_msg)
        state, time = x.copy(), t
        for bound in EventLocator._find_bounds(problem, t, until):
            ceiling = EventLocator._find_left_limit(bound)
            while time < bound - _RESTART * max(1.0, abs(bound)):
                step = min(dt, bound - time)
                following, interpolant = take(time, state, step, ceiling)
                if history is not None:
                    history.add(time, time + step, interpolant)
                hit = (
                    locator.crossing(
                        time,
                        time + step,
                        state,
                        following,
                        p=p,
                        interpolant=interpolant,
                        include_hidden=False,
                    )
                    if problem.events
                    else None
                )
                if hit is not None:
                    time, index = hit
                    if history is not None:
                        history.truncate(time)
                    state = bounded(time, locator.apply(time, index, interpolant(time), p), p)
                    if locator.stopped_at is not None:
                        return state
                    continue
                state, time = bounded(time + step, following, p), time + step
            time = bound
        return state
