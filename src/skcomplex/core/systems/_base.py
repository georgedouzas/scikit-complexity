"""Define the abstract base of systems, prepared once and simulated many times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..components.base import BaseComponent
from ..components.systems import CompositeSystem
from ..engine.backends import BaseBackend
from ..engine.problems import BaseProblemSpec
from ..engine.runtime import BaseClock, BaseObserver, Trajectory, run_problem
from ..engine.solvers import BaseSolver
from ..errors import NotPreparedError
from ..specs import ObservableSpec, Tags


class BaseSystem(BaseComponent[Sequence[ObservableSpec], SimpleNamespace]):
    """An abstract system, which is prepared once and simulated many times.

    `prepare` checks the components of the system and compiles them into a problem once, as `fit` does for a
    scikit-learn estimator. `simulate` then solves that problem from a starting point and returns its trajectory, as
    `predict` does, so one prepared system runs many times. A subclass takes the parameters of its model as arguments
    of its `__init__` and returns the `System` of its components from the public hook `build`. Class attributes may
    tune the end time, the time between frames, the observables, the initial state, the invariants, and the schedule
    that `simulate` uses for the arguments a call leaves out.

    Read more in the user guide, under Systems.

    Attributes:
        system_:
            The prepared `System` of the components built from the model parameters.

    See Also:
        System: The system of components that every ready-made system builds.

    Examples:
        >>> from skcomplex.core.systems import BaseSystem, System
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> class EcologySystem(BaseSystem):
        ...     def build(self):
        ...         return System(self.label, systems=[LotkaVolterraSystem(label='meadow')])
        >>> trajectory = EcologySystem().prepare().simulate(until=5.0)
        >>> float(trajectory.times[-1])
        5.0
    """

    until: ClassVar[float | None] = None
    dt: ClassVar[float | None] = None
    observables: ClassVar[Sequence[ObservableSpec]] = ()
    initial_state: ClassVar[Mapping[str, object] | None] = None
    invariants: ClassVar[Mapping[str, float] | None] = None
    schedule: ClassVar[Mapping[str, Callable[[float], float | npt.ArrayLike]] | None] = None

    system_: BaseSystem
    composite_: CompositeSystem
    problem_: BaseProblemSpec
    solver_: BaseSolver
    backend_: BaseBackend | None

    def __init__(self, label: str = 'system') -> None:
        """Store the label of the system.

        Args:
            label:
                The label of the system, which names it when it is nested in another one.
        """
        super().__init__(label)

    def _get_composite(self) -> CompositeSystem:
        """Return the composite of the components that the engine compiles."""
        return self.build()._get_composite()

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return the laws of the components for a namespace of symbols.

        Args:
            namespace:
                The namespace of the system.

        Returns:
            laws:
                The laws of the components.
        """
        return self._get_composite()._get_laws(namespace)

    def _get_tags(self) -> Tags:
        """Return the tags derived from the components of the system."""
        return self._get_composite()._get_tags()

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

    def _prepare(self, context: Sequence[ObservableSpec] | None = None) -> None:
        """Build the components and prepare their system as `system_`.

        The system of the components compiles the tuned observables, so a run that records only them compiles nothing.

        Args:
            context:
                Unused, since a system serves nothing outside itself.
        """
        super()._prepare(context)
        self.system_ = self.build().prepare(self.observables)
        self._prepared_from = self._get_values()

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Set `laws_`, the laws of the system for a namespace of symbols.

        Args:
            inputs:
                The namespace of the symbols the laws can use.
        """
        self.laws_ = self._get_laws_of(inputs)

    @abstractmethod
    def build(self) -> BaseSystem:
        """Return the `System` of the components, built from the model parameters.

        Returns:
            system:
                The system of the elements, interactions, nested systems, and space.
        """

    def simulate(
        self,
        until: float | None = None,
        *,
        dt: float | None = None,
        initial_state: Mapping[str, object] | None = None,
        random_state: int | np.random.Generator | None = None,
        observables: Sequence[ObservableSpec] = (),
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
                The end time, or `None` for the tuned one.

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
        compiled = None if observables else (prepared.problem_, prepared.solver_)
        return run_problem(
            prepared.composite_,
            compiled,
            until=self.until if until is None else until,
            dt=self.dt if dt is None else dt,
            initial_state=self.initial_state if initial_state is None else initial_state,
            solver=prepared.solver_ if observables else None,
            backend=prepared.backend_,
            observables=observables,
            observers=observers,
            invariants=self.invariants if invariants is None else invariants,
            schedule=self.schedule if schedule is None else schedule,
            random_state=random_state,
            checked=checked,
            clock=clock,
            record_draws=record_draws,
            replay=replay,
        )
