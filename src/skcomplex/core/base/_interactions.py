"""Define the base of interactions, the parts of a system that act between elements."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Self, cast

from ._errors import ValidationError
from ._laws import BaseLaw, BaseQuantity
from ._types import (
    Variable,
    VariableKind,
    _argument_names,
    _caller_module,
    _check_declared,
    _declarations,
    _declare,
    _define_type,
    _get_default,
    _get_params,
)
from ._utils import BaseEstimator

BINDINGS = ('members', 'pairs', 'neighbors', 'hyperedges')


def _interaction_declarations(kind: type['BaseInteraction']) -> dict[str, Variable]:
    """Return the declarations of an interaction type, after checking its binding and that it declares no state."""
    declarations = _declarations(kind)
    if kind.over not in BINDINGS:
        error_msg = f'{kind.__name__} binds its laws over {kind.over!r}. Use one of {", ".join(BINDINGS)}.'
        raise ValidationError(error_msg)
    states = [name for name, item in declarations.items() if item.kind == VariableKind.STATE]
    adaptive = [name for name in states if declarations[name].adaptive]
    if adaptive:
        error_msg = (
            f'{kind.__name__} declares the adaptive parameters {", ".join(adaptive)}, which laws change as states, '
            'but interactions hold no state. Declare them on the element whose parameters they are.'
        )
        raise ValidationError(error_msg)
    if states:
        error_msg = (
            f'{kind.__name__} declares the states {", ".join(states)}, but interactions hold no state. Put state on '
            'couplings in a group on the edges of a network space, `GroupElement(..., on="edges")`.'
        )
        raise ValidationError(error_msg)
    return declarations


class BaseInteraction(BaseEstimator):
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
        >>> from skcomplex.core.base import BaseInteraction
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

    `check()` rejects a value the type does not declare, a declared variable without a default that is not given, a type
    that declares a state or an unknown binding, and a layer of an interaction not over neighbours. `prepare()` sets
    `declarations_`, and `run(s)` returns the laws of the interaction for a namespace of symbols, from the hook `laws`.
    The hook `quantities` returns the auxiliaries, invariants, and objectives of the interaction for the same namespace.

    Attributes:
        declarations_:
            The declarations of the parameters and roles of the type, by name in declaration order.
    """

    reserved_names: ClassVar[frozenset[str]] = frozenset({'label'})
    over: ClassVar[str] = 'members'

    def __init__(self, label: str, *, layer: str | None = None, **values: Any) -> None:  # noqa: ANN401
        _declare(self, label, values)
        self.layer = layer

    @classmethod
    def _define(cls, name: str, **declarations: Any) -> type[Self]:  # noqa: ANN401
        """Return a subclass of the type built from declarations, in the module of the code that called `define`."""
        return cast('type[Self]', _define_type(cls, name, _caller_module(), **declarations))

    @classmethod
    def _declarations(cls) -> dict[str, Variable]:
        """Return the declarations of the type, after checking its binding and that it declares no state."""
        return _interaction_declarations(cls)

    def _check(self) -> None:
        """Raise an error for a value or a layer the interaction cannot take, or a type that declares a state.

        Raises:
            ArgumentError:
                A value names a variable the type does not declare, or a declared variable without a default is not
                given.

            ValidationError:
                The type binds its laws over an unknown binding or declares states, or an interaction not over
                neighbours takes a layer.
        """
        _interaction_declarations(type(self))
        _check_declared(self, BaseInteraction)
        if self.layer is not None and self.over != 'neighbors':
            error_msg = (
                f'{type(self).__name__} {self.label!r} takes the layer {self.layer!r}, but only interactions over '
                'neighbours run on the edges of a layer.'
            )
            raise ValidationError(error_msg)

    def _prepare(self) -> None:
        """Set `declarations_`, the declarations of the type."""
        self.declarations_ = type(self)._declarations()

    def _run(self, s: Any) -> list[Any]:  # noqa: ANN401
        """Return the laws of the interaction for a namespace of symbols, from the hook `laws`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return self.laws(s)

    @classmethod
    def get_default(cls, name: str) -> Any:  # noqa: ANN401
        """Return the default value of a declared name or a setting, a fresh copy for a mutable declared default.

        Args:
            name:
                The declared name or the setting.

        Returns:
            default:
                The default value, `dataclasses.MISSING` for a declared name without one.

        Raises:
            KeyError:
                The type neither declares the name nor takes it as a setting with a default.

        Examples:
            >>> from skcomplex.models.biology.ecology import PopulationElement
            >>> PopulationElement.get_default('n')
            1.0
        """
        return _get_default(cls, BaseInteraction, name)

    @classmethod
    def get_argument_names(cls) -> list[str]:
        """Return the names the constructor takes besides the label: its settings and the declared names.

        Returns:
            names:
                The names, settings first.

        Examples:
            >>> from skcomplex.models.economics.finance import CorrelatedGeometricBrownianMotionInteraction
            >>> CorrelatedGeometricBrownianMotionInteraction.get_argument_names()
            ['index', 'layer', 'target', 'market', 'mu', 'sigma']
        """
        return _argument_names(cls, BaseInteraction)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the label, the settings, and the values of every declared name.

        Args:
            deep:
                Whether to add the parameters of nested estimators as `<name>__<parameter>`.

        Returns:
            params:
                A mapping from argument name to value, in sorted order.

        Examples:
            >>> from skcomplex.models.biology.ecology import PopulationElement
            >>> PopulationElement('rabbits', n=40).get_params()['n']
            40
        """
        return _get_params(self, BaseInteraction, deep=deep)

    @classmethod
    def define(
        cls,
        name: str,
        *,
        roles: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
        laws: Callable[..., list[BaseLaw]] | None = None,
        quantities: Callable[..., Sequence[BaseQuantity]] | None = None,
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

            quantities:
                A function `quantities(s)` or `quantities(self, s)` that returns the auxiliaries, invariants, and
                objectives of the interaction.

            over:
                The binding of the laws, such as `'neighbors'`, or `None` to inherit it.

        Returns:
            interaction_type:
                The new interaction type, a subclass of the type `define` is called on.

        Examples:
            >>> from skcomplex.core.base import BaseInteraction
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
        return cls._define(
            name,
            roles=roles,
            parameters=parameters,
            laws=laws,
            quantities=quantities,
            attributes=attributes,
        )

    def laws(self, s: Any) -> list[Any]:  # noqa: ANN401
        """Return the laws of the interaction, none by default.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return []

    def quantities(self, s: Any) -> list[Any]:  # noqa: ANN401
        """Return the auxiliaries, invariants, and objectives of the interaction, none by default.

        Args:
            s:
                The namespace of the symbols the quantities can use.

        Returns:
            quantities:
                The quantities of the interaction.
        """
        return []
