"""Define the system of components, the estimator of the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from ..components import Observable
from ..components.base import BaseComponent
from ..components.elements import BaseElement
from ..components.interactions import BaseInteraction
from ..components.spaces import BaseSpace
from ..components.systems import CompositeSystem
from ..engine.backends import BaseBackend, JaxBackend, NumbaBackend, NumpyBackend
from ..engine.problems import BaseProblemSpec, ProblemSpec
from ..engine.problems._neighborhood_kernel import NeighborhoodKernel
from ..engine.problems._reduction_kernel import ReductionKernel
from ..engine.problems._rule_kernel import RuleKernel
from ..engine.runtime import Trajectory, compile_problem
from ..engine.solvers import BaseSolver
from ..enums import FormKind
from ..errors import ValidationError
from ..specs import EntrySpec
from ._base import AUTO, BaseSystem

SETTINGS = ('space', 'solver', 'backend')
BACKENDS = (NumpyBackend, NumbaBackend, JaxBackend)


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
        states_:
            The names of the values the laws change, with the derivatives and lags they need.

        parameters_:
            The names of the values no law changes.

        equations_:
            One readable equation per state, summed over the laws and reduced to first order.

        form_:
            The kind of problem the laws make, such as the ordinary differential equations.

        solver_:
            The solver that every run copies.

        backend_:
            The backend that evaluates the compiled functions.

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
        super().__init__(label, solver=solver, backend=backend)
        self.elements = elements
        self.interactions = interactions
        self.systems = systems
        self.space = space

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

    @staticmethod
    def _get_backend_of(problem: BaseProblemSpec) -> BaseBackend:
        """Return the backend whose name the compiled problem records, NumPy by default."""
        name = getattr(problem, 'backend', 'numpy')
        kind = next((item for item in BACKENDS if item._get_name() == name), NumpyBackend)
        return kind()

    @staticmethod
    def _get_readable(expression: object, names: Mapping[str, str]) -> str:
        """Return an expression as text, with every symbol named by its dotted path."""
        text = str(expression)
        for symbol in sorted(names, key=len, reverse=True):
            text = re.sub(rf'(?<![\w.]){re.escape(symbol)}(?![\w])', names[symbol], text)
        return text

    def _get_readable_reduction(self, reduction: ReductionKernel, readable: Mapping[str, str]) -> str:
        """Return a reduction over the members of a group as text, such as `mean(crowd.x)`."""
        argument = self._get_readable(reduction.argument, readable).replace('[i]', '')
        weighted = '' if reduction.weights is None else ', weighted'
        condition = (
            '' if reduction.condition is None else f', where={self._get_readable(reduction.condition, readable)}'
        )
        return f'{reduction.kind}({argument}{condition}{weighted})'

    def _get_readable_neighbours(self, kernel: NeighborhoodKernel, readable: Mapping[str, str]) -> str:
        """Return a reduction over neighbours as text, such as `neighbors(x[j] - x[i])`.

        `[i]` marks the member that reads the reduction and `[j]` a neighbour.
        """
        indices = {'reader': '[i]', 'other': '[j]', 'edge': '[ij]'}
        names = dict(readable)
        for symbol, read in kernel.reads.items():
            entry = read.read.entry
            if entry is not None:
                names[str(symbol)] = f'{Trajectory._get_display_name(entry.name)}{indices.get(read.side, "")}'
        argument = self._get_readable(kernel.argument, names)
        weight = '' if str(kernel.weight) == '1' else f', weight={self._get_readable(kernel.weight, names)}'
        return f"neighbors({argument}{weight}, reduce='{kernel.kind}')"

    @staticmethod
    def _get_readable_rules(problem: BaseProblemSpec, entries: Sequence[EntrySpec]) -> dict[str, str]:
        """Return the local rules of a problem as text, by the states they set.

        A rule reads the ordered values of the neighbours of a cell, such as
        `life(neighbors(cells.alive, reduce='each'))`, and the rules of one state run one after the other.
        """
        names = {entry.name: entry.generic_names for entry in entries}
        display = Trajectory._get_display_name
        rules: dict[str, list[str]] = {}
        for step in getattr(problem, 'lattice', None) or ():
            if not isinstance(step, RuleKernel):
                continue
            reads = ', '.join(f"neighbors({display(state.name)}, reduce='each')" for state in step.reads)
            for state in step.targets:
                for name in names.get(state.name, ()):
                    rules.setdefault(display(name), []).append(f'{step.name}({reads})')
        return {name: ', then '.join(items) for name, items in rules.items()}

    def _set_learned(self, problem: BaseProblemSpec) -> None:
        """Set the names, the readable equations, and the form of a compiled problem."""
        states = getattr(problem, 'states', None)
        parameters = getattr(problem, 'parameters', None)
        state_entries = list(states.entries) if states is not None else []
        parameter_entries = list(parameters.entries) if parameters is not None else []
        display = Trajectory._get_display_name
        agents = [f'{group.path}.{name}' for group in getattr(problem, 'agents', ()) for name in group.states]
        names = [display(name) for entry in state_entries for name in entry.generic_names]
        self.states_ = tuple(dict.fromkeys([*names, *agents]))
        self.parameters_ = tuple(display(name) for entry in parameter_entries for name in entry.generic_names)
        symbols: Mapping[str, object] = getattr(problem, 'symbols', {})
        readable = {
            str(item): display(name)
            for entry in [*state_entries, *parameter_entries]
            if entry.name in symbols
            for item, name in zip(
                ProblemSpec._get_real_components_of(symbols[entry.name]),
                entry.generic_names,
                strict=False,
            )
        }
        for delay in getattr(problem, 'delays', ()):
            state = self._get_readable(delay.state, readable)
            readable[str(delay.symbol)] = f'{state}(t - {self._get_readable(delay.delay, readable)})'
        for reduction in getattr(problem, 'reductions', ()):
            readable[str(reduction.symbol)] = self._get_readable_reduction(reduction, readable)
        for kernel in getattr(problem, 'neighbourhoods', ()):
            readable[str(kernel.symbol)] = self._get_readable_neighbours(kernel, readable)
        equations: Mapping[str, object] = getattr(problem, 'equations', {})
        self.equations_ = {display(name): self._get_readable(value, readable) for name, value in equations.items()}
        self.equations_.update(self._get_readable_rules(problem, state_entries))
        form = str(problem.form)
        if problem.form == FormKind.ODE and getattr(problem, 'algebraic', None) is not None:
            form = 'differential-algebraic equations'
        self.form_ = f'{form} with events' if getattr(problem, 'events', ()) else form

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Check the components, compile the problem, and choose the solver and backend.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self._check_runners()
        self._composite = self._get_composite()
        solver = self.solver if isinstance(self.solver, BaseSolver) else None
        self._backend = self.backend if isinstance(self.backend, BaseBackend) else None
        self._problem, self.solver_ = compile_problem(
            self._composite, solver=solver, backend=self._backend, observables=context or (),
        )
        self.backend_ = self._backend or self._get_backend_of(self._problem)
        self._set_learned(self._problem)
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
