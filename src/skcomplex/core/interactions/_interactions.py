"""Define interactions, which couple components bound to their roles through laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from ..base import ValidationError, Variable, VariableKind
from ..components import BaseComponent
from ..laws import BaseLaw

BINDINGS = ('members', 'pairs', 'neighbors', 'hyperedges')


class BaseInteraction(BaseComponent):
    """Base class of laws that couple elements to each other or to the environment.

    Subclasses declare their roles with `role` and their parameters with `parameter`, and override `laws`. Inside
    `laws`, `s.<role>` is the namespace of the element bound to the role and `s.<parameter>` is a symbol.

    Args:
        label:
            The label of the interaction, unique among the children of its system.

        layer:
            Over neighbours, the layer of the network whose edges the interaction runs on, or `None` for the layer of
            the edge group it binds, or else the graph of the network.

        **values:
            The values of the declared parameters and roles, by name.

    Examples:
        >>> from skcomplex.core.base import NumericDomain, parameter, role
        >>> from skcomplex.core.interactions import BaseInteraction
        >>> from skcomplex.core.laws import RateLaw
        >>> class GrowthInteraction(BaseInteraction):
        ...     target: str = role()
        ...     r: float = parameter(1.0, domain=NumericDomain.POSITIVE)
        ...
        ...     def laws(self, s):
        ...         return [RateLaw(s.target.n, s.r * s.target.n)]
        >>> GrowthInteraction('growth', target='rabbits', r=0.5)
        GrowthInteraction(label='growth', r=0.5, target='rabbits')

    The class variable `over` is the binding of the laws: `'members'` runs them for the bound components or members,
    `'pairs'` once per pair of members of the roles `first` and `second` closer than the parameter `cutoff`,
    `'neighbors'` once per edge of the space of the group of nodes its role binds, and `'hyperedges'` once per
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
            The type declares a state, or an unknown binding, or an interaction not over neighbours takes a layer.
    """

    over: ClassVar[str] = 'members'

    def __init__(self, label: str, *, layer: str | None = None, **values: Any) -> None:  # noqa: ANN401
        super().__init__(label, **values)
        self.layer = layer
        if layer is not None and self.over != 'neighbors':
            error_msg = (
                f'{type(self).__name__} {label!r} takes the layer {layer!r}, but only interactions over neighbours run '
                'on the edges of a layer.'
            )
            raise ValidationError(error_msg)

    @classmethod
    def declarations(cls) -> dict[str, Variable]:
        """Return the declarations of the parameters and roles of the type, after checking its binding.

        Returns:
            declarations:
                A mapping from name to declaration, in declaration order.

        Raises:
            ValidationError:
                The type binds its laws over an unknown binding, or declares states or adaptive parameters.

        Examples:
            >>> from skcomplex.models.biology.ecology import PredationInteraction
            >>> list(PredationInteraction.declarations())[:2]
            ['predator', 'prey']
        """
        declarations = super().declarations()
        if cls.over not in BINDINGS:
            error_msg = f'{cls.__name__} binds its laws over {cls.over!r}. Use one of {", ".join(BINDINGS)}.'
            raise ValidationError(error_msg)
        states = [name for name, item in declarations.items() if item.kind is VariableKind.STATE]
        adaptive = [name for name in states if declarations[name].adaptive]
        if adaptive:
            error_msg = (
                f'{cls.__name__} declares the adaptive parameters {", ".join(adaptive)}, which laws change as states, '
                'but interactions hold no state. Declare them on the element whose parameters they are.'
            )
            raise ValidationError(error_msg)
        if states:
            error_msg = (
                f'{cls.__name__} declares the states {", ".join(states)}, but interactions hold no state. Put state on '
                'couplings in a group on the edges of a network space, `GroupComponent(..., on="edges")`.'
            )
            raise ValidationError(error_msg)
        return declarations

    @classmethod
    def define(
        cls,
        name: str,
        *,
        roles: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
        laws: Callable[..., list[BaseLaw]] | None = None,
        over: str | None = None,
    ) -> type['BaseInteraction']:
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
                The binding of the laws, such as `'neighbors'`, or `None` to inherit it.

        Returns:
            interaction_type:
                The new interaction type, a subclass of the type `define` is called on.

        Examples:
            >>> from skcomplex.core.interactions import BaseInteraction
            >>> from skcomplex.core.laws import RateLaw
            >>> GrowthInteraction = BaseInteraction.define(
            ...     'GrowthInteraction',
            ...     roles=['target'],
            ...     parameters={'r': 1.0},
            ...     laws=lambda s: [RateLaw(s.target.n, s.r * s.target.n)],
            ... )
            >>> GrowthInteraction('growth', target='rabbits', r=0.5)
            GrowthInteraction(label='growth', r=0.5, target='rabbits')
        """
        attributes = {} if over is None else {'over': over}
        return cls._define(name, roles=roles, parameters=parameters, laws=laws, attributes=attributes)
