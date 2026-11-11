"""Define the abstract base of every interaction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ...enums import DomainKind
from ...errors import SpaceError, ValidationError
from ...specs import BaseSpec, InteractionRequirementSpec, ParameterSpec, PartnerSpec, RoleSpec, StateSpec, Tags
from ..base import BaseComponent
from ..laws import BaseLaw
from ..spaces import BaseSpace, ContinuousSpace
from ..types import AgentViewProtocol, SystemViewProtocol

GRID_CANDIDATES = ('neighbors', 'cell', 'local')
RATE_DOMAINS = (DomainKind.REAL, DomainKind.INTEGER, DomainKind.COMPLEX, DomainKind.VECTOR)


class BaseInteraction(BaseComponent[BaseSpace, SimpleNamespace]):
    """An abstract interaction, the component that changes the states of elements.

    An interaction binds elements through roles, such as `'source'` and `'target'`, reads their states, and returns laws
    that change them. It holds no state of its own. A state on a coupling, such as the weight of an edge, lives in a
    group on the edges or hyperedges of a network. A subclass takes each role and parameter as an argument of its
    `__init__`, annotated with `RoleSpec` or `ParameterSpec` inside `Annotated`, and stores it under the same name, as a
    scikit-learn estimator does. It returns its laws from `_get_laws`, where `namespace.<role>` reads the states of the
    element a role binds and `namespace.<parameter>` is a symbol.

    A role states the element kinds and states it needs in its `RoleSpec`, and the class lists the laws it may return in
    `law_classes`, so that `get_requirements` and `get_tags` can name them before a run. A subclass whose arguments
    choose among its laws overrides `_get_tags`. A subclass that names its roles or parameters otherwise returns their
    names from `_get_role` and `_get_parameter`. An interaction that draws a random order of agents, breaks ties at
    random, or picks members at random is stochastic whatever its laws.

    Unlike `BaseComponent`, it changes the states of other components and holds none of its own. Choose it when a new
    kind of interaction runs its laws in a way of its own, as `NeighborInteraction` runs them once per edge. Choose
    `Interaction` for laws that run once for the elements the roles bind.

    It appears in every field through its subclasses, such as infection in an epidemic, a chemical reaction, a payment
    between accounts, and the coupling of oscillators.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'growth'`.

    Attributes:
        laws_:
            The laws the interaction contributes for the namespace of its last run.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.interactions import BaseInteraction
        >>> from skcomplex.core.specs import ParameterSpec, RoleSpec
        >>> class GrowthInteraction(BaseInteraction):
        ...     def __init__(
        ...         self, label, target: Annotated[str, RoleSpec()], r: Annotated[float, ParameterSpec()] = 1.0
        ...     ):
        ...         super().__init__(label)
        ...         self.target = target
        ...         self.r = r
        ...
        ...     def _get_laws(self, namespace):
        ...         return []
        >>> GrowthInteraction('growth', target='rabbits', r=0.5).get_params()
        {'label': 'growth', 'r': 0.5, 'target': 'rabbits'}
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = ()
    laws_: Sequence[BaseComponent]

    def _get_argument(self, name: str) -> object:
        """Return the value of an argument of the general interaction.

        It reads the argument under its subclass name, and returns `None` without one.

        Args:
            name:
                The name of the argument in the general interaction, such as `'noise'`.

        Returns:
            value:
                The value of the argument, or `None` when the interaction has no such attribute.
        """
        return getattr(self, self._get_parameter(name), None)

    def _get_variant_tags(
        self,
        classes: Sequence[type[BaseLaw]],
        *,
        stochastic: bool = False,
        structure: frozenset[str] = frozenset(),
    ) -> Tags:
        """Return the tags of the laws an interaction returns for its arguments.

        The tags are unknown below its own `_get_laws`.

        Args:
            classes:
                The law classes the arguments choose.

            stochastic:
                Whether the arguments make laws of deterministic kinds draw, such as an update with a probability.

            structure:
                How the laws change the structure of the system, such as `'changing topology'`.

        Returns:
            tags:
                The tags of the chosen laws, or unknown law kinds when a subclass writes laws it does not list.
        """
        if self._get_declared_laws() is None:
            return replace(self._get_law_tags(None), structure=structure)
        tags = self._get_law_tags(classes)
        return replace(tags, randomness='stochastic' if stochastic else tags.randomness, structure=structure)

    def _get_tags(self) -> Tags:
        """Return the tags of the laws the class lists.

        The tags are unknown law kinds when the class writes laws it does not list.
        """
        return self._get_law_tags(self._get_declared_laws())

    def _get_draws(self) -> bool:
        """Return whether the interaction draws.

        It draws, by `draws`, a random order of its agents, or random ties or picks.
        """
        return (
            self.draws
            or self._get_argument('schedule') in ('random-sequential', 'member')
            or 'random' in (self._get_argument('ties'), self._get_argument('pick'))
        )

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Set `laws_`, the laws of the interaction for a namespace of symbols.

        Args:
            inputs:
                The namespace of the symbols the laws can use.
        """
        self.laws_ = self._get_laws_of(inputs)

    def _get_role(self, role: str) -> str:
        """Return the role a subclass binds in place of a role of its general interaction.

        It returns the name of the role.

        Args:
            role:
                The name of the role in the general interaction, such as `'source'` or `'target'`.

        Returns:
            name:
                The name of the role of the subclass, the role itself by default.
        """
        return role

    def _get_member(self, namespace: SimpleNamespace, role: str) -> SimpleNamespace:
        """Return the namespace of the element bound to a role of the general interaction.

        The role is read under its subclass name.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

            role:
                The name of the role in the general interaction, such as `'source'` or `'target'`.

        Returns:
            member:
                The namespace bound to the name `_get_role` returns for the role.
        """
        member: SimpleNamespace = getattr(namespace, self._get_role(role))
        return member

    def _get_parameter(self, name: str) -> str:
        """Return the parameter a subclass reads in place of a general one.

        It returns the name of the parameter that the subclass reads in place of a parameter of its general interaction.

        Args:
            name:
                The name of the parameter in the general interaction, such as `'probability'`.

        Returns:
            parameter:
                The name of the parameter of the subclass, the name itself by default.
        """
        return name

    def _declares(self, name: str) -> bool:
        """Return whether the class declares a name as a parameter."""
        return isinstance(type(self)._get_specs().get(name), ParameterSpec)

    def _get_parameter_value(self, name: str, system: SystemViewProtocol) -> float:
        """Return the value of a parameter in an agent form.

        It is the plain attribute when the class does not declare the parameter.

        Args:
            name:
                The name of the parameter in the general interaction.

            system:
                The system view of the interaction.

        Returns:
            value:
                The value of the parameter of the system, or of the attribute of the interaction.
        """
        parameter = self._get_parameter(name)
        if self._declares(parameter):
            return system.get_parameter(parameter)
        return float(getattr(self, parameter))

    def _get_parameter_symbol(self, name: str, namespace: SimpleNamespace) -> object:
        """Return the symbol of a parameter in a symbolic form.

        It is the plain attribute when the class does not declare the parameter.

        Args:
            name:
                The name of the parameter in the general interaction.

            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            symbol:
                The symbol of the parameter in the namespace, read also when a component other than the interaction
                declares it, or else the value of the attribute of the interaction.
        """
        parameter = self._get_parameter(name)
        if self._declares(parameter) or hasattr(namespace, parameter):
            return getattr(namespace, parameter)
        return getattr(self, parameter)

    def _check_variant(
        self,
        name: str,
        value: object,
        values: Collection[object],
        kinds: tuple[type, ...] = (),
        *,
        functions: bool = False,
    ) -> None:
        """Raise an error for a variant of no named value and of no kind it also takes.

        The variant is none of its named values and of none of the kinds it also takes.

        Args:
            name:
                The name of the variant argument, such as `'pairing'`.

            value:
                The value of the variant argument.

            values:
                The named values of the variant.

            kinds:
                The types the variant takes besides its named values, such as spec classes.

            functions:
                Whether the variant takes a function.

        Raises:
            ValidationError:
                The value is none of the named values and of none of the kinds.
        """
        if isinstance(value, kinds) or (functions and callable(value)) or any(value == item for item in values):
            return
        named = [kind.__name__ for kind in kinds if kind is not BaseSpec]
        accepted = [*named, *(repr(str(item)) for item in values), *(['a function'] if functions else [])]
        spec = ', or a spec that the component handles' if BaseSpec in kinds else ''
        error_msg = f'Unknown {name} {value!r} of {self.label!r}. Use one of {", ".join(accepted)}{spec}.'
        raise ValidationError(error_msg)

    def _get_variant_error(self, value: object, name: str, method: str) -> ValidationError:
        """Return the error for a spec of a variant that no subclass runs.

        Args:
            value:
                The value of the variant argument.

            name:
                The name of the variant argument, such as `'rule'`.

            method:
                The private method a subclass overrides to run the spec.

        Returns:
            error:
                The error that names the variant and the method.
        """
        error_msg = (
            f'{type(self).__name__} {self.label!r} has no way to run the {name} {value!r}. Override `{method}` in a '
            'subclass to run it.'
        )
        return ValidationError(error_msg)

    def _update_agents[T](
        self,
        system: SystemViewProtocol,
        schedule: str,
        get_next: Callable[[AgentViewProtocol, SystemViewProtocol], T],
        set_next: Callable[[AgentViewProtocol, T], None],
    ) -> None:
        """Set the next value of every agent of the target by the schedule.

        The agents update all at once or one after the other.

        Args:
            system:
                The system view of the interaction.

            schedule:
                `'synchronous'` for every agent from the values at the start of the rule, or `'sequential'` or
                `'random-sequential'` for one agent after the other, in their order or in a new random order, each
                reading the values the agents before it set.

            get_next:
                The function of an agent and the system view that returns its next value.

            set_next:
                The function that sets the next value of an agent.
        """
        agents = list(system.get_agents(self._get_role('target')))
        if schedule == 'synchronous':
            values = [get_next(agent, system) for agent in agents]
            for agent, value in zip(agents, values, strict=True):
                set_next(agent, value)
            return
        order = system.rng.permutation(len(agents)) if schedule == 'random-sequential' else range(len(agents))
        for index in order:
            agent = agents[int(index)]
            if agent.alive:
                set_next(agent, get_next(agent, system))

    def _pick_best(self, scores: npt.ArrayLike, ties: str, system: SystemViewProtocol) -> int:
        """Return the position of the largest score.

        It is the first one or one drawn uniformly among ties.

        Args:
            scores:
                The scores of the options, in their order.

            ties:
                `'first'` for the first of the largest scores, or `'random'` for one of them drawn uniformly.

            system:
                The system view of the interaction, whose random stream draws among ties.

        Returns:
            position:
                The position of the chosen option.
        """
        values = np.asarray(scores, dtype=float)
        best = np.flatnonzero(values == values.max())
        if ties == 'first':
            return int(best[0])
        return int(best[int(system.rng.integers(len(best)))])

    def _check_states(self) -> None:
        """Raise an error for a type with states or adaptive parameters without a shape.

        The type declares the states or the adaptive parameters.
        """
        specs = self._get_specs()
        states = [name for name, spec in specs.items() if isinstance(spec, StateSpec)]
        adaptive = [
            name
            for name, spec in specs.items()
            if isinstance(spec, ParameterSpec) and spec.adaptive and spec.shape is None
        ]
        if adaptive:
            error_msg = (
                f'{type(self).__name__} declares the adaptive parameters {", ".join(adaptive)}, which laws change as '
                'states, but interactions hold no state. Declare them on the element whose parameters they are, or '
                'give them the shape of the groups the interaction couples, `ParameterSpec(shape=...)`.'
            )
            raise ValidationError(error_msg)
        if states:
            error_msg = (
                f'{type(self).__name__} declares the states {", ".join(states)}, but interactions hold no state. Put '
                'state on couplings in a group on the edges of a network space, `GroupElement(..., on="edges")`.'
            )
            raise ValidationError(error_msg)

    def _check_grid(self, name: str, value: object, space: BaseSpace | None, alternative: str) -> None:
        """Raise an error for a variant that needs cells or nodes in a continuous space.

        The variant needs the cells of a grid or the nodes of a network.

        Args:
            name:
                The name of the variant argument, such as `'pairing'`.

            value:
                The value of the variant argument.

            space:
                The space of the system.

            alternative:
                The variant or argument that runs in a continuous space.

        Raises:
            SpaceError:
                The space is continuous.
        """
        if not isinstance(space, ContinuousSpace):
            return
        error_msg = (
            f'The {name} {value!r} of {self.label!r} needs the cells of a grid or the nodes of a network, but the '
            f'system has a continuous space. Use {alternative}.'
        )
        raise SpaceError(error_msg)

    def _check_partners(self, partners: object, space: BaseSpace | None) -> None:
        """Raise an error for a selection of partners among cells or nodes.

        The error is for a continuous space.
        """
        if isinstance(partners, PartnerSpec) and partners.candidates in GRID_CANDIDATES:
            self._check_grid('candidates', partners.candidates, space, 'a radius, `PartnerSpec(candidates=<radius>)`')

    @classmethod
    def _get_law_classes(cls) -> tuple[type[BaseLaw], ...]:
        """Return the law classes the class lists.

        There are none when a subclass writes its own laws below the list.
        """
        for klass in cls.__mro__:
            declared = vars(klass)
            if 'law_classes' in declared:
                classes: tuple[type[BaseLaw], ...] = declared['law_classes']
                return classes
            if '_get_laws' in declared:
                return ()
        return ()

    @staticmethod
    def _get_accepted(kinds: Sequence[tuple[type, ...]]) -> tuple[type, ...]:
        """Return the classes that some law accepts, none when a law accepts any class."""
        if not all(kinds):
            return ()
        return tuple(dict.fromkeys(kind for classes in kinds for kind in classes))

    def _get_state_names(self, names: Sequence[str]) -> tuple[str, ...]:
        """Return the states that declared names mean.

        They are the names an argument holds, or else the names themselves.
        """
        found: list[str] = []
        for name in names:
            value = getattr(self, name, name)
            if isinstance(value, str):
                found.append(value)
            elif isinstance(value, Mapping | list | tuple):
                found += [item for item in value if isinstance(item, str)]
        return tuple(dict.fromkeys(found))

    def _check_space(self, space: BaseSpace | None) -> None:
        """Raise an error for a variant the space of the system cannot run.

        There is none by default.

        Args:
            space:
                The space of the system, or `None`.
        """

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for a variant the space of the system cannot run.

        It first raises an error for declarations the interaction cannot take.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            SpaceError:
                A variant needs a kind of space the system does not have.
            ValidationError:
                The type declares states.
        """
        super()._prepare(context)
        self._check_states()
        self._check_space(context)

    def get_requirements(self) -> InteractionRequirementSpec:
        """Return what the interaction needs.

        It needs the element kinds and states of its roles, and its laws' time and space.

        Returns:
            requirements:
                The roles by name, with the states each reads for the values of this interaction, the law classes
                the class lists, and the classes of time and of space these laws run in.

        Examples:
            >>> from skcomplex.core.components.interactions import RateInteraction
            >>> interaction = RateInteraction('growth', target='rabbits', state='n', rate=lambda s: s.target.n)
            >>> interaction.get_requirements().roles['target'].states
            ('n',)
        """
        laws = self._get_law_classes()
        needs = [law.get_requirements() for law in laws]
        return InteractionRequirementSpec(
            time=self._get_accepted([need.time for need in needs]),
            space=self._get_accepted([need.space for need in needs]),
            roles={
                name: replace(spec, states=self._get_state_names(spec.states))
                for name, spec in self._get_specs().items()
                if isinstance(spec, RoleSpec)
            },
            laws=laws,
        )
