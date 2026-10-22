"""Define interactions, which couple components bound to their roles through laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from ..base import (
    ValidationError,
    declaration_of,
    parameter,
    role,
)
from ..components._components import BINDINGS, Component, _caller_module, _define, _is_declaration, _is_state
from ..laws._laws import Law


class Interaction(Component):
    """Base class of laws that couple elements to each other or to the environment.

    Subclasses declare their roles with `role` and their parameters with `parameter`, and override `laws`. Inside
    `laws`, `s.<role>` is the namespace of the element bound to the role and `s.<parameter>` is a symbol.

    Args:
        label:
            The label of the interaction, unique among the children of its system.

    Examples:
        >>> from skcomplex.core.base import Domain, parameter, role
        >>> from skcomplex.core.interactions import Interaction
        >>> from skcomplex.core.laws import Rate
        >>> class Growth(Interaction):
        ...     target: str = role()
        ...     r: float = parameter(1.0, domain=Domain.POSITIVE)
        ...
        ...     def laws(self, s):
        ...         return [Rate(s.target.n, s.r * s.target.n)]
        >>> Growth('growth', target='rabbits', r=0.5)
        Growth(label='growth', r=0.5, target='rabbits')

    The class variable `over` is the binding of the laws: `'members'` runs them for the bound components or members,
    `'pairs'` once per pair of members of the roles `first` and `second` closer than the parameter `cutoff`,
    `'neighbours'` once per pair of adjacent members of the role `target`, and `'hyperedges'` once per hyperedge.
    Interactions hold no state: state on couplings lives in groups on the edges or hyperedges of a network space.

    Raises:
        ValidationError:
            The subclass declares a state, or an unknown binding.
    """

    over: ClassVar[str] = 'members'

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Reject states and unknown bindings when the subclass is created."""
        super().__init_subclass__(**kwargs)
        if cls.over not in BINDINGS:
            error_msg = f'{cls.__name__} binds its laws over {cls.over!r}. Use one of {", ".join(BINDINGS)}.'
            raise ValidationError(error_msg)
        states = [field.name for field in dataclasses.fields(cls) if _is_state(field)]
        adaptive = [name for name in states if declaration_of(cls.__dataclass_fields__[name]).adaptive]
        if adaptive:
            error_msg = (
                f'{cls.__name__} declares the adaptive parameters {", ".join(adaptive)}, which laws change as states, '
                'but interactions hold no state. Declare them on the element whose parameters they are.'
            )
            raise ValidationError(error_msg)
        if states:
            error_msg = (
                f'{cls.__name__} declares the states {", ".join(states)}, but interactions hold no state. Put state on '
                'couplings in a group on the edges of a network space, `Group(..., on="edges")`.'
            )
            raise ValidationError(error_msg)

    @classmethod
    def define(
        cls,
        name: str,
        *,
        roles: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
        laws: Callable[..., list[Law]] | None = None,
    ) -> type['Interaction']:
        """Return a new interaction type built from declarations, without writing a class.

        The type behaves like a subclass in cloning and parameter introspection. Assign it to a module-level name equal
        to `name` for it to pickle.

        Args:
            name:
                The name of the type.

            roles:
                The names of the roles, each bound to an element path in the constructor.

            parameters:
                A mapping from parameter name to a `parameter` declaration or to a default value.

            laws:
                A function `laws(s)` or `laws(self, s)` that returns the laws of the interaction.

        Returns:
            interaction_type:
                The new interaction type, a subclass of the type `define` is called on.

        Examples:
            >>> from skcomplex.core.interactions import Interaction
            >>> from skcomplex.core.laws import Rate
            >>> Growth = Interaction.define(
            ...     'Growth',
            ...     roles=['target'],
            ...     parameters={'r': 1.0},
            ...     laws=lambda s: [Rate(s.target.n, s.r * s.target.n)],
            ... )
            >>> Growth('growth', target='rabbits', r=0.5)
            Growth(label='growth', r=0.5, target='rabbits')
        """
        fields = {
            **{key: role() for key in roles},
            **{key: value if _is_declaration(value) else parameter(value) for key, value in (parameters or {}).items()},
        }
        return _define(cls, name=name, fields=fields, laws=laws, attributes={}, module=_caller_module())
