"""Define the base of interactions, the parts of a system that act between elements."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from ...errors import ValidationError
from ...specs import ParameterSpec, StateSpec
from ..base import BasePart
from ..spaces import BaseSpace
from ..types import ModelViewProtocol


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

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Leave the interaction as it is, since it has nothing to prepare.

        Args:
            context:
                The space the interaction inherits from its system, which an interaction does not read.
        """
