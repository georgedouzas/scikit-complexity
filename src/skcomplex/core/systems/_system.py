"""Define the system of components, the estimator of the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence

from ..components import Observable
from ..components.base import BaseComponent
from ..components.elements import BaseElement
from ..components.interactions import BaseInteraction
from ..components.spaces import BaseSpace
from ..components.systems import CompositeSystem
from ..engine.backends import BaseBackend
from ..engine.runtime import compile_problem
from ..engine.solvers import BaseSolver
from ..errors import ValidationError
from ._base import BaseSystem

AUTO = 'auto'
SETTINGS = ('space', 'solver', 'backend')


class System(BaseSystem):
    """A system of elements, interactions, nested systems, and a space.

    It is the estimator of the library. `prepare` checks every component, compiles the laws of the interactions into
    one problem, and chooses the solver and the backend, and `simulate` solves the problem and returns the trajectory.
    The parameters of the components are reachable by nested names, `<label>__<parameter>`, as the steps of a
    scikit-learn `Pipeline` are, so `set_params(gravity__g=1.62)` sets the parameter `g` of the component labelled
    `gravity`. A solver or backend left as `'auto'` follows from the problem. It models an ecosystem of interacting
    populations, a reaction network, a circuit, or a market, and a system nests in another one as a component.

    Read more in the user guide, under Systems.

    Attributes:
        composite_:
            The composite of the components that the engine compiles.

        problem_:
            The compiled problem.

        solver_:
            The solver that every run copies.

        backend_:
            The backend of the runs, or `None` for the default one.

    See Also:
        BaseSystem: The base of ready-made systems, which prepare a system from the parameters of their model.

    Examples:
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> system = System(systems=[LotkaVolterraSystem(label='meadow')])
        >>> system.set_params(meadow__r=1.5).get_params()['meadow__r']
        1.5
        >>> float(system.prepare().simulate(until=5.0).times[-1])
        5.0
    """

    def __init__(
        self,
        label: str = 'system',
        *,
        elements: Sequence[BaseElement] = (),
        interactions: Sequence[BaseInteraction] = (),
        systems: Sequence[BaseComponent] = (),
        space: BaseSpace | None = None,
        solver: BaseSolver | str = AUTO,
        backend: BaseBackend | str = AUTO,
    ) -> None:
        """Store the label, the components, and the settings of the system.

        Args:
            label:
                The label of the system, which names it when it is nested in another one.

            elements:
                The elements, which hold the states.

            interactions:
                The interactions, which change the states through their laws.

            systems:
                The nested systems, each with a label of its own.

            space:
                The space the elements live on, or `None` for none.

            solver:
                The solver of the problem, or `'auto'` for the default solver of the problem.

            backend:
                The backend that evaluates the compiled functions, or `'auto'` for the default one.
        """
        super().__init__(label)
        self.elements = elements
        self.interactions = interactions
        self.systems = systems
        self.space = space
        self.solver = solver
        self.backend = backend

    def _get_components(self) -> list[BaseComponent]:
        """Return the elements, interactions, and nested systems, addressed by label."""
        return [*self.elements, *self.interactions, *self.systems]

    def _get_composite(self) -> CompositeSystem:
        """Return the composite that the engine compiles.

        A nested system of this kind is replaced by its own composite, and any other nested system is kept as it is.
        """
        systems = [system._get_composite() if isinstance(system, BaseSystem) else system for system in self.systems]
        return CompositeSystem(
            self.label,
            elements=list(self.elements),
            interactions=list(self.interactions),
            systems=systems,
            space=self.space,
        )

    def _check_runners(self) -> None:
        """Raise an error for a solver or a backend of the wrong kind.

        Raises:
            ValidationError:
                The solver is not `'auto'` or a solver, or the backend is not `'auto'` or a backend.
        """
        for name, kind in (('solver', BaseSolver), ('backend', BaseBackend)):
            value = getattr(self, name)
            if value != AUTO and not isinstance(value, kind):
                error_msg = (
                    f'The {name} of the system {self.label!r} must be {AUTO!r} or a {kind.__name__}, not {value!r}.'
                )
                raise ValidationError(error_msg)

    def _get_target(self, name: str) -> BaseComponent:
        """Return the setting or the component that a nested name starts with.

        Args:
            name:
                The first segment of a nested name: `space`, `solver`, `backend`, or the label of a component.

        Returns:
            target:
                The setting or the component.

        Raises:
            ValidationError:
                No setting and no component has the name, with the labels of the components in the message.
        """
        target = getattr(self, name) if name in SETTINGS else None
        target = target or next((component for component in self._get_components() if component.label == name), None)
        if not isinstance(target, BaseComponent | BaseSolver | BaseBackend):
            labels = ', '.join(repr(component.label) for component in self._get_components()) or 'none'
            error_msg = f'The system {self.label!r} has no component labelled {name!r}. Its components are {labels}.'
            raise ValidationError(error_msg)
        return target

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Check the components, compile the problem, and choose the solver and backend.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self._check_runners()
        self.composite_ = self._get_composite()
        solver = self.solver if isinstance(self.solver, BaseSolver) else None
        self.problem_, self.solver_ = compile_problem(self.composite_, solver=solver, observables=context or ())
        self.backend_ = self.backend if isinstance(self.backend, BaseBackend) else None
        self.system_ = self

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Return the parameters of the system and of its components by nested names.

        Args:
            deep:
                Whether to add the parameters of the space, the solver, the backend, and every component, each under
                `<name>__<parameter>`.

        Returns:
            params:
                The parameters by name.

        Examples:
            >>> from skcomplex.core.systems import System
            >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
            >>> System(systems=[LotkaVolterraSystem(label='meadow')]).get_params()['meadow__prey']
            40.0
        """
        params = super().get_params(deep=False)
        if not deep:
            return params
        for name in SETTINGS:
            value = params[name]
            if isinstance(value, BaseComponent | BaseSolver | BaseBackend):
                params.update({f'{name}__{key}': item for key, item in value.get_params(deep=True).items()})
        for component in self._get_components():
            params.update({f'{component.label}__{key}': item for key, item in component.get_params(deep=True).items()})
        return params

    def set_params(self, **params: object) -> System:
        """Set parameters of the system and of its components by nested names.

        Args:
            **params:
                The new values, by the name of a parameter of the system, or by `<name>__<parameter>` for a setting
                or a component.

        Returns:
            self:
                The system.

        Raises:
            ValidationError:
                A name matches no parameter, no setting, and no component, named in the message.

        Examples:
            >>> from skcomplex.core.systems import System
            >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
            >>> system = System(systems=[LotkaVolterraSystem(label='meadow')]).set_params(meadow__a=0.5)
            >>> system.get_params()['meadow__a']
            0.5
        """
        own = super().get_params(deep=False)
        nested: dict[str, dict[str, object]] = {}
        for key, value in params.items():
            head, separator, rest = key.partition('__')
            if separator:
                nested.setdefault(head, {})[rest] = value
                continue
            if key not in own:
                error_msg = f'The system {self.label!r} has no parameter {key!r}. Its parameters are {", ".join(own)}.'
                raise ValidationError(error_msg)
            setattr(self, key, value)
        for head, values in nested.items():
            target = self._get_target(head)
            accepted = target.get_params(deep=isinstance(target, System))
            for name in values:
                first = name.partition('__')[0]
                if first not in accepted and name not in accepted:
                    error_msg = (
                        f'The component {head!r} of the system {self.label!r} has no parameter {first!r}. '
                        f'Its parameters are {", ".join(target.get_params(deep=False))}.'
                    )
                    raise ValidationError(error_msg)
            target.set_params(**values)
        return self
