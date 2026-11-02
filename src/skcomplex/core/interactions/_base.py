"""Define the base of interactions, the parts of a system that act between elements."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt

from ...errors import SpaceError, ValidationError
from ...specs import ParameterSpec, PartnerSpec, StateSpec
from ..base import BasePart
from ..spaces import BaseSpace, ContinuousSpace
from ..types import AgentViewProtocol, ModelViewProtocol

SCHEDULES = ('synchronous', 'sequential', 'random-sequential', 'member')
GRID_CANDIDATES = ('neighbors', 'cell', 'local')


class BaseInteraction(BasePart[BaseSpace, SimpleNamespace]):
    """Base of every interaction, a part of a system whose laws couple elements to each other or to the environment.

    A subclass takes its roles and parameters as arguments of its own `__init__`, annotated with `Annotated` and the
    spec `RoleSpec` or `ParameterSpec`, and stores each under its own name, as in scikit-learn. It returns its
    laws from `_get_laws`, where `namespace.<role>` is the namespace of the element bound to the role and
    `namespace.<parameter>` is a symbol. Interactions hold no state: state on couplings lives in groups on the edges or
    hyperedges of a network space. A general interaction reads the members of its roles, such as `'source'` or
    `'target'`, by the name `_get_role` returns for them, the role itself by default, so a subclass that names its
    roles otherwise returns its own names, in the agent forms and, through `_get_member`, in the symbolic forms.
    It reads its parameters by the name `_get_parameter` returns for them, and a parameter the class does not declare
    reads the plain attribute, so a subclass renames a parameter or fixes its value by setting the attribute.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

    Attributes:
        laws_:
            The laws the interaction contributes for the namespace of its last run.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.interactions._base import BaseInteraction
        >>> from skcomplex.specs import ParameterSpec, RoleSpec
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

    laws_: Sequence[BasePart]

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Set `laws_`, the laws of the interaction for a namespace of symbols.

        Args:
            inputs:
                The namespace of the symbols the laws can use.
        """
        self.laws_ = self._laws_of(inputs)

    def _get_role(self, role: str) -> str:
        """Return the name of the role a subclass binds in place of a role of its general interaction.

        Args:
            role:
                The name of the role in the general interaction, such as `'source'` or `'target'`.

        Returns:
            name:
                The name of the role of the subclass, the role itself by default.
        """
        return role

    def _get_member(self, namespace: SimpleNamespace, role: str) -> SimpleNamespace:
        """Return the namespace of the element bound to a role of the general interaction, under its subclass name.

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
        """Return the name of the parameter a subclass reads in place of a parameter of its general interaction.

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

    def _get_parameter_value(self, name: str, model: ModelViewProtocol) -> float:
        """Return the value of a parameter in an agent form, the plain attribute when the class does not declare it.

        Args:
            name:
                The name of the parameter in the general interaction.

            model:
                The model view of the interaction.

        Returns:
            value:
                The value of the parameter of the model, or of the attribute of the interaction.
        """
        parameter = self._get_parameter(name)
        if self._declares(parameter):
            return model.parameter(parameter)
        return float(getattr(self, parameter))

    def _get_parameter_symbol(self, name: str, namespace: SimpleNamespace) -> object:
        """Return the symbol of a parameter in a symbolic form, the plain attribute when the class does not declare it.

        Args:
            name:
                The name of the parameter in the general interaction.

            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            symbol:
                The symbol of the parameter in the namespace, read also when a part other than the interaction
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
        values: Sequence[object],
        kinds: tuple[type, ...] = (),
        *,
        functions: bool = False,
    ) -> None:
        """Raise an error for a variant that is none of its named values and of none of the kinds it also takes.

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
        accepted = [*(kind.__name__ for kind in kinds), *map(repr, values), *(['a function'] if functions else [])]
        error_msg = f'Unknown {name} {value!r} of {self.label!r}. Use one of {", ".join(accepted)}.'
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
        model: ModelViewProtocol,
        schedule: str,
        get_next: Callable[[AgentViewProtocol, ModelViewProtocol], T],
        set_next: Callable[[AgentViewProtocol, T], None],
    ) -> None:
        """Set the next value of every agent of the target, all at once or one after the other, by the schedule.

        Args:
            model:
                The model view of the interaction.

            schedule:
                `'synchronous'` for every agent from the values at the start of the rule, or `'sequential'` or
                `'random-sequential'` for one agent after the other, in their order or in a new random order, each
                reading the values the agents before it set.

            get_next:
                The function of an agent and the model view that returns its next value.

            set_next:
                The function that sets the next value of an agent.
        """
        agents = list(model.agents(self._get_role('target')))
        if schedule == 'synchronous':
            values = [get_next(agent, model) for agent in agents]
            for agent, value in zip(agents, values, strict=True):
                set_next(agent, value)
            return
        order = model.rng.permutation(len(agents)) if schedule == 'random-sequential' else range(len(agents))
        for index in order:
            agent = agents[int(index)]
            if agent.alive:
                set_next(agent, get_next(agent, model))

    def _pick_best(self, scores: npt.ArrayLike, ties: str, model: ModelViewProtocol) -> int:
        """Return the position of the largest score, the first one or one drawn uniformly among ties.

        Args:
            scores:
                The scores of the options, in their order.

            ties:
                `'first'` for the first of the largest scores, or `'random'` for one of them drawn uniformly.

            model:
                The model view of the interaction, whose random stream draws among ties.

        Returns:
            position:
                The position of the chosen option.
        """
        values = np.asarray(scores, dtype=float)
        best = np.flatnonzero(values == values.max())
        if ties == 'first':
            return int(best[0])
        return int(best[int(model.rng.integers(len(best)))])

    def _check_states(self) -> None:
        """Raise an error for a type that declares states or adaptive parameters without a shape."""
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

    def _check(self) -> None:
        """Raise an error for declarations the interaction cannot take.

        Raises:
            ValidationError:
                The type declares states.
        """
        self._check_states()

    def _check_grid(self, name: str, value: object, space: BaseSpace | None, alternative: str) -> None:
        """Raise an error for a variant that needs the cells of a grid or the nodes of a network in a continuous space.

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
        """Raise an error for a selection of partners among cells or nodes in a continuous space."""
        if isinstance(partners, PartnerSpec) and partners.candidates in GRID_CANDIDATES:
            self._check_grid('candidates', partners.candidates, space, 'a radius, `PartnerSpec(candidates=<radius>)`')

    def _check_space(self, space: BaseSpace | None) -> None:
        """Raise an error for a variant the space of the system cannot run, none by default.

        Args:
            space:
                The space of the system, or `None`.
        """

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for a variant the space of the system cannot run.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            SpaceError:
                A variant needs a kind of space the system does not have.
        """
        self._check_space(context)
