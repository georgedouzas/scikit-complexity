"""Define the abstract base of systems and the run settings they tune."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar, TypedDict

import numpy.typing as npt

from ...base import BaseConfig
from ...errors import ValidationError
from .._observable_spec import Observable
from .._tags import Tags
from ..base import BaseComponent
from ..spaces import BaseSpace


class RunSettings(TypedDict):
    """A mapping of the tuned run settings of a system, which `simulate` reads.

    `simulate` uses each setting for the argument a call leaves out, such as `until`, `dt`, or `solver`.

    Unlike the arguments of `simulate`, which hold for one call, it holds for every run of a system class. Choose it
    when a system class knows the end time, the step, and the solver that suit it.

    It appears in many fields. An epidemic runs for a season, a pendulum for a few periods, and a lattice model for a
    number of sweeps that its system class fixes.
    """

    until: float | None
    dt: float | None
    solver: BaseConfig | None
    observables: Sequence[Observable]
    initial_state: Mapping[str, object] | None
    invariants: Mapping[str, float] | None
    schedule: Mapping[str, Callable[[float], float | npt.ArrayLike]] | None
    backend: BaseConfig | None


class BaseSystem[CompositeT: BaseSystem](BaseComponent[BaseSpace]):
    """An abstract system that builds its components from the parameters of its model.

    A subclass takes the parameters of its model as arguments of its `__init__`, such as the rates of an epidemic, and
    stores each under the same name, as a scikit-learn estimator does, so `get_params`, `set_params`, and `clone` work.
    It builds its components in the hook `_build_system`, which returns the `CompositeSystem` of its elements,
    interactions, nested systems, space, and time. `prepare` builds the components again when a parameter has changed
    since the last build, and prepares them as `system_`. The tags of a system come from its components and are never
    declared. The type parameter `CompositeT` names the class of the composite, so `BaseSystem[CompositeSystem]` types
    `system_` as `CompositeSystem`.

    A subclass may give tuned run settings as class variables, which `simulate` reads for the arguments a call leaves
    out:

    - `until` and `dt`: the end time and the time between frames.
    - `solver` and `backend`: how the system runs.
    - `observables` and `invariants`: what to record, and the tolerance of each observable that must stay constant.
    - `initial_state` and `schedule`: the initial values by dotted state name, and the functions of time that drive
      parameters.

    Unlike `CompositeSystem`, which takes its components as data, it writes them in terms of named model parameters.
    Choose it when a model has parameters that users set, sweep, or calibrate, as every system of the catalog does.

    It appears in many fields. The SIR model of an epidemic, the Lotka-Volterra model of predators and prey, the Ising
    model of a magnet, and the Kuramoto model of oscillators are system classes.

    Attributes:
        system_:
            The composite of the components built from the model parameters at the last `prepare`.
    """

    until: ClassVar[float | None] = None
    dt: ClassVar[float | None] = None
    solver: BaseConfig | None = None
    observables: Sequence[Observable] = ()
    initial_state: ClassVar[Mapping[str, object] | None] = None
    invariants: ClassVar[Mapping[str, float] | None] = None
    schedule: ClassVar[Mapping[str, Callable[[float], float | npt.ArrayLike]] | None] = None
    backend: ClassVar[BaseConfig | None] = None

    system_: CompositeT

    def __init__(self, label: str = 'system') -> None:
        """Store the label of the system.

        Args:
            label:
                The label of the system, which names it when it is nested in another one.
        """
        super().__init__(label)

    @abstractmethod
    def _build_system(self) -> CompositeT:
        """Return the composite of the components of the system.

        The components are built from its model parameters.
        """

    def _get_pass_parameters(self) -> CompositeT:
        """Return the components of the system built from its current model parameters.

        The components are built again only when a model parameter differs by value from those of the last build, so a
        system that fits or generates data when it builds its components does so once per set of parameters, however
        often `get_tags` and `prepare` read them.

        Returns:
            system:
                The composite of the components.

        Raises:
            ValidationError:
                The hook `_build_system` returns no system that is its own composite.
        """
        params = self.get_params(deep=False)
        built: tuple[dict[str, object], CompositeT] | None = getattr(self, '_built_from', None)
        if built is not None and self._is_equal(built[0], params):
            return built[1]
        system = self._build_system()
        if not isinstance(system, BaseSystem) or system._build_system() is not system:
            error_msg = (
                f'{type(self).__name__} builds its components as a system of its own, not as {type(system).__name__}.'
            )
            raise ValidationError(error_msg)
        self._built_from = (params, system)
        return system

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Pass the model parameters down to the components, and prepare them as `system_`.

        Preparing the composite of the components checks every component.

        Args:
            context:
                The space a nested system inherits from the system that holds it, or `None`.
        """
        super()._prepare(context)
        self.system_ = self._get_pass_parameters().prepare(context)

    def _get_tags(self) -> Tags:
        """Return the tags derived from the components built from the current parameters."""
        return self._get_pass_parameters()._get_tags()

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings of the system.

        `simulate` uses them for every argument a call leaves out. The settings are the class attributes `until`, `dt`,
        `solver`, `observables`, `initial_state`, `invariants`, `schedule`, and `backend`. A system whose
        settings depend on its model parameters or on its built components overrides this method.

        Returns:
            settings:
                The tuned run settings by name.
        """
        return RunSettings(
            until=self.until,
            dt=self.dt,
            solver=self.solver,
            observables=self.observables,
            initial_state=self.initial_state,
            invariants=self.invariants,
            schedule=self.schedule,
            backend=self.backend,
        )
