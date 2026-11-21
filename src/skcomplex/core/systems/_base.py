"""Define the abstract base of systems, prepared once and simulated many times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import ClassVar, Self

import numpy as np
import numpy.typing as npt
from sklearn.base import clone

from ..components import Observable, Tags
from ..components.base import BaseComponent
from ..engine.backends import BaseBackend
from ..engine.compile import OperatorCompiler
from ..engine.compile._pass_compiler import PassCompiler
from ..engine.problems import BaseProblemSpec
from ..engine.problems._composite_system import _CompositeSystem
from ..engine.runtime import BaseClock, BaseObserver, Trajectory, run_problem
from ..engine.solvers import BaseSolver
from ..enums import FormKind
from ..errors import NotPreparedError

AUTO = 'auto'
SETTINGS = ('solver', 'backend')
LEARNED = ('states_', 'parameters_', 'equations_', 'form_', 'solver_', 'backend_')


class BaseSystem(BaseComponent[Sequence[Observable]]):
    """An abstract system, which is prepared once and simulated many times.

    `prepare` checks the parts of the system and turns them into a problem once, as `fit` does for a scikit-learn
    estimator. `simulate` then solves that problem from a starting point and returns its trajectory, as `predict` does,
    so one prepared system runs many times, such as `LotkaVolterraSystem().prepare().simulate(until=5.0)`. A ready-made
    system takes the parameters of its model as arguments of `__init__`, with `solver` and `backend`, and its `_prepare`
    builds `system_`, the `System` of its parts. Class attributes may set the end time, the time between frames, the
    observables, and the starting values that `simulate` uses when a call leaves them out. It is the base of the
    ready-made systems of predators and prey, of reactions in a cell, and of prices in a market.

    Read more in the user guide, under Systems.

    Attributes:
        system_:
            The prepared `System` of the parts built from the parameters of the model.

        observables_:
            The observables compiled with the problem, the class attribute `observables` unless `_prepare` sets them
            from the parameters of the model, such as the lowest energy found by an annealing.

        initial_state_:
            The starting values `simulate` uses when a call gives none, the class attribute `initial_state` unless
            `_prepare` sets them from the parameters of the model, such as the ground state of a chain of spins.

    See Also:
        System: The system of parts that every ready-made system builds.

    Examples:
        >>> from skcomplex.core.systems import BaseSystem
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> system = LotkaVolterraSystem().prepare()
        >>> isinstance(system, BaseSystem), float(system.simulate(until=5.0).times[-1])
        (True, 5.0)
    """

    until: ClassVar[float | None] = None
    dt: ClassVar[float | None] = None
    observables: ClassVar[Sequence[Observable]] = ()
    initial_state: ClassVar[Mapping[str, object] | None] = None
    invariants: ClassVar[Mapping[str, float] | None] = None
    schedule: ClassVar[Mapping[str, Callable[[float], float | npt.ArrayLike]] | None] = None

    system_: BaseSystem
    observables_: tuple[Observable, ...]
    initial_state_: Mapping[str, object] | None
    states_: tuple[str, ...]
    parameters_: tuple[str, ...]
    equations_: dict[str, str]
    form_: str
    solver_: BaseSolver
    backend_: BaseBackend
    _composite: _CompositeSystem
    _problem: BaseProblemSpec
    _backend: BaseBackend | None

    def __init__(
        self,
        label: str = 'system',
        *,
        solver: BaseSolver | str = AUTO,
        backend: BaseBackend | str = AUTO,
    ) -> None:
        """Store the label and the settings of the system.

        Args:
            label:
                The label of the system, which names it when it is nested in another one.

            solver:
                The solver of the problem, or `'auto'` for the one `system_` chooses.

            backend:
                The backend that evaluates the compiled functions, or `'auto'` for the one `system_` chooses.
        """
        super().__init__(label)
        self.solver = solver
        self.backend = backend

    def _get_composite(self) -> _CompositeSystem:
        """Return the composite of the components that the engine compiles.

        A copy of the system builds the components, so reading the tags before `prepare` changes nothing.
        """
        built = clone(self)
        built._prepare(None)
        return built.system_._get_composite()

    def _get_tags(self) -> Tags:
        """Return the tags derived from the components of the system.

        The domains of an element given only by its values, such as `Element('ball', values={'x': 10.0})`, follow from
        the laws that change them, so the states and parameters of such elements are inferred first. The kinds of laws
        that interactions learn only when they are prepared, such as those of `Interaction('flip', laws=...)`, follow
        from the census of the laws of the system, and so does their randomness when their kinds draw, such as noise.
        A system of operators takes no census, since `OperatorCompiler` reads its laws, so only its declarations give
        its tags.
        """
        compiler = PassCompiler()
        declared = compiler._get_declared_system(self._get_composite())
        tags = declared._get_tags()
        if tags.laws is not None or OperatorCompiler._contains_operator(declared):
            return tags
        census = compiler._get_census_of(declared, compiler.nodes_).census
        if census is None or census.errors:
            return tags
        learned = self._get_law_tags([type(record.law) for record in census.records])
        randomness = 'stochastic' if learned.randomness == 'stochastic' else tags.randomness
        return replace(tags, laws=learned.laws, randomness=randomness, time=tags.time or learned.time)

    def _get_values(self) -> dict[str, object]:
        """Return the parameters of the system and of its components by name."""
        return self.get_params(deep=True)

    def _check_prepared(self) -> None:
        """Raise an error for a system not prepared, or changed since it was prepared.

        Raises:
            NotPreparedError:
                The system was never prepared, or a parameter changed after `prepare`, named in the message.
        """
        if not getattr(self, '_prepared', False):
            error_msg = f'The system {self.label!r} must be prepared before it simulates. Call `prepare()` first.'
            raise NotPreparedError(error_msg)
        prepared: dict[str, object] = getattr(self, '_prepared_from', {})
        changed = [name for name, value in self._get_values().items() if not self._is_equal(prepared.get(name), value)]
        if changed:
            error_msg = (
                f'The parameters {", ".join(changed)} of the system {self.label!r} changed since it was prepared. '
                'Call `prepare()` again.'
            )
            raise NotPreparedError(error_msg)

    @abstractmethod
    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Check the model parameters and set `system_`, the `System` of the components.

        A subclass starts with `super()._prepare(context)`, which checks the label and sets `observables_` to the
        observables of the context and `initial_state_` to the tuned starting values. A subclass whose observables or
        starting values depend on its model parameters sets them after it.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        self._check_specs()
        self.observables_ = tuple(context or ())
        self.initial_state_ = self.initial_state

    def prepare(self, context: Sequence[Observable] | None = None) -> Self:
        """Check the system, build its components, and compile them into a problem.

        Args:
            context:
                The observables to compile with the problem, or `None` for the tuned ones.

        Returns:
            self:
                The prepared system.

        Raises:
            ValidationError:
                A model parameter or a component is invalid, named in the message.
        """
        super().prepare(self.observables if context is None else context)
        if self.system_ is not self:
            settings = {name: getattr(self, name) for name in SETTINGS if getattr(self, name) != AUTO}
            self.system_.set_params(**settings).prepare(self.observables_)
            for name in LEARNED:
                setattr(self, name, getattr(self.system_, name))
        self._prepared_from = self._get_values()
        return self

    def simulate(
        self,
        until: float | None = None,
        *,
        dt: float | None = None,
        initial_state: Mapping[str, object] | None = None,
        random_state: int | np.random.Generator | None = None,
        observables: Sequence[Observable] = (),
        observers: Sequence[BaseObserver] = (),
        invariants: Mapping[str, float] | None = None,
        schedule: Mapping[str, Callable[[float], float | npt.ArrayLike]] | None = None,
        clock: BaseClock | None = None,
        checked: bool = False,
        record_draws: bool = False,
        replay: Trajectory | None = None,
    ) -> Trajectory:
        """Simulate the prepared system until a time and return its trajectory.

        A run reuses the problem compiled by `prepare`, so a prepared system simulates many times with different end
        times, initial states, and seeds. Observables other than the tuned ones compile the system again for that run.

        Args:
            until:
                The end time, or `None` for the tuned one. A static system, which only solves its objectives and
                constraints, needs none.

            dt:
                The time between frames, or `None` for the tuned one.

            initial_state:
                Initial values by dotted state name, such as `{'ball.x': 20.0}`, or `None` for the tuned or the declared
                ones.

            random_state:
                The seed or the generator of the random numbers of the run, or `None` for fresh entropy.

            observables:
                The observables to record, or none for the tuned ones.

            observers:
                The observers that watch the run, such as a Rerun recording.

            invariants:
                The tolerance of each observable that must stay constant, or `None` for the tuned ones.

            schedule:
                The functions of time that drive parameters, by dotted name, or `None` for the tuned ones.

            clock:
                The clock that paces the run, or `None` to run as fast as possible.

            checked:
                Whether every frame checks that states stay in their domains.

            record_draws:
                Whether the run records its random draws.

            replay:
                A trajectory whose recorded draws the run replays, or `None`.

        Returns:
            trajectory:
                The states and observables of the run at every frame.

        Raises:
            NotPreparedError:
                The system was not prepared, or a parameter changed since it was prepared.

        Examples:
            >>> from skcomplex.core.systems import System
            >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
            >>> system = System(systems=[LotkaVolterraSystem(label='meadow')]).prepare()
            >>> short, long = system.simulate(until=5.0), system.simulate(until=10.0)
            >>> float(short.times[-1]), float(long.times[-1])
            (5.0, 10.0)
        """
        self._check_prepared()
        prepared = self.system_
        starting = self.initial_state_ if initial_state is None else initial_state
        named = any(isinstance(value, str) for value in (starting or {}).values())
        compiled = None if observables or named else (prepared._problem, prepared.solver_)
        end = self.until if until is None else until
        return run_problem(
            prepared._composite,
            compiled,
            until=0.0 if end is None and prepared.form_ == FormKind.STATIC else end,
            dt=self.dt if dt is None else dt,
            initial_state=starting,
            solver=prepared.solver_ if compiled is None else None,
            backend=prepared._backend,
            observables=observables if compiled is not None or observables else self.observables_,
            observers=observers,
            invariants=self.invariants if invariants is None else invariants,
            schedule=self.schedule if schedule is None else schedule,
            random_state=random_state,
            checked=checked,
            clock=clock,
            record_draws=record_draws,
            replay=replay,
        )
