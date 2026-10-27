"""Define the base of interactions, the parts of a system that act between elements."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

from ..base import BasePart
from ..errors import ValidationError
from ..specs import ParameterSpec, StateSpec


class BaseInteraction(BasePart):
    """Base of every interaction, a part of a system whose laws couple elements to each other or to the environment.

    A subclass takes its roles and parameters as arguments of its own `__init__`, annotated with `Annotated` and the
    spec `RoleSpec` or `ParameterSpec`, and stores each under its own name, as in scikit-learn. It returns its
    laws from `_get_laws`, where `namespace.<role>` is the namespace of the element bound to the role and
    `namespace.<parameter>` is a symbol. Interactions hold no state: state on couplings lives in groups on the edges or
    hyperedges of a network space.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.interactions._base import BaseInteraction
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

    def _check_states(self) -> None:
        """Raise an error for a type that declares states or adaptive parameters, which interactions do not hold."""
        specs = self._get_specs()
        states = [name for name, spec in specs.items() if isinstance(spec, StateSpec)]
        adaptive = [name for name, spec in specs.items() if isinstance(spec, ParameterSpec) and spec.adaptive]
        if adaptive:
            error_msg = (
                f'{type(self).__name__} declares the adaptive parameters {", ".join(adaptive)}, which laws change as '
                'states, but interactions hold no state. Declare them on the element whose parameters they are.'
            )
            raise ValidationError(error_msg)
        if states:
            error_msg = (
                f'{type(self).__name__} declares the states {", ".join(states)}, but interactions hold no state. Put '
                'state on couplings in a group on the edges of a network space, `GroupElement(..., on="edges")`.'
            )
            raise ValidationError(error_msg)

    def _check(self) -> Self:
        """Raise an error for declarations the interaction cannot take.

        Raises:
            ValidationError:
                The type declares states.

        Returns:
            interaction:
                The interaction itself.
        """
        self._check_states()
        return self

    def _prepare(self, context: object = None) -> Self:
        """Return the interaction, which has nothing to prepare.

        Args:
            context:
                The space the interaction inherits from its system, which an interaction does not read.

        Returns:
            interaction:
                The interaction itself.
        """
        return self
