"""Define interactions, which couple components bound to their roles through laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import inspect
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
    `'neighbours'` once per edge of the space of the group of nodes its role binds, and `'hyperedges'` once per
    hyperedge. Interactions hold no state: state on couplings lives in groups on the edges or hyperedges of a network
    space.

    Over neighbours, the interaction takes the keyword `layer`, the layer of the network whose edges it runs on, or
    `None` for the layer of the edge group it binds, or else the graph of the network. It binds one role to a group
    with one member per node or site and may bind a second role to a group on the edges of the same network and
    layer. Its laws then run once per edge and read `s.source` and `s.target`, the members at both ends, `s.edge`, the
    member of the edge group, and its parameters. On a directed graph the ends follow the direction of the edge, and on
    an undirected graph the source comes first in node order. Rate, noise, and jump laws may target any of them, and
    update laws the edge member.

    Raises:
        ValidationError:
            The subclass declares a state, or an unknown binding.
    """

    over: ClassVar[str] = 'members'

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Add the layer of a binding over neighbours, and reject states and unknown bindings."""
        if (
            cls.over == 'neighbours'
            and 'layer' not in vars(cls)
            and 'layer' not in getattr(cls, '__dataclass_fields__', {})
        ):
            cls.layer = None  # type: ignore[attr-defined]
            cls.__annotations__ = {**inspect.get_annotations(cls), 'layer': str | None}
        super().__init_subclass__(**kwargs)
        if cls.over not in BINDINGS:
            error_msg = f'{cls.__name__} binds its laws over {cls.over!r}. Use one of {", ".join(BINDINGS)}.'
            raise ValidationError(error_msg)
        states = [field.name for field in dataclasses.fields(cls) if _is_state(field)]
        adaptive = [
            name for name in states if getattr(declaration_of(cls.__dataclass_fields__[name]), 'adaptive', False)
        ]
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
        over: str | None = None,
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

            over:
                The binding of the laws, such as `'neighbours'`, or `None` to inherit it.

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
        attributes = {} if over is None else {'over': over}
        return _define(cls, name=name, fields=fields, laws=laws, attributes=attributes, module=_caller_module())
