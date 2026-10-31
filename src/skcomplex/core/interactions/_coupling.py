"""Define coupling of a state of the members to their neighbours or to their group, through a function."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, RateLaw
from ..laws.utils import delayed, mean
from ._neighbor import NeighborInteraction

MODES = ('edges', 'mean-field')


class CouplingInteraction(NeighborInteraction):
    """Coupling of a state of every member to the states of its neighbours or of its whole group.

    The rate of the state `x_i` of a member gains `K f(x_j(t - tau), x_i + alpha)` for every neighbour `j` in the mode
    `'edges'`, along both directions of every edge, or the mean of it over every member `j` of the group, the member
    itself included, in the mode `'mean-field'`. The function `f` is the difference `x_j - x_i` by default, `alpha` is
    a phase lag, and `tau` a delay, with no delay when it is 0.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group on the nodes or sites.

        state:
            The name of the coupled state of the members.

        function:
            A function of the state of the other member and of the state of the member, which returns a symbolic
            expression, or `None` for their difference.

        K:
            The coupling strength.

        alpha:
            The phase lag added to the state of the member.

        tau:
            The delay of the state of the other member, 0 for none.

        mode:
            `'edges'` for the sum over the neighbours, or `'mean-field'` for the mean over every member of the group.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the graph of the network.

    Raises:
        ValidationError:
            The mode is unknown, or the function is not callable.

    Examples:
        >>> from sage.all__sagemath_symbolics import sin
        >>> from skcomplex.core.interactions import CouplingInteraction
        >>> coupling = CouplingInteraction('coupling', nodes='cells', state='theta', function=lambda a, b: sin(a - b))
        >>> coupling.check().mode
        'edges'
    """

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        state: str = 'x',
        function: Callable[[Expression, Expression], Expression] | None = None,
        K: Annotated[float, ParameterSpec()] = 1.0,
        alpha: Annotated[float, ParameterSpec()] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        mode: str = 'edges',
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.nodes = nodes
        self.state = state
        self.function = function
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.mode = mode

    def _check(self) -> None:
        """Raise an error for an unknown mode or a function that cannot be called.

        Raises:
            ValidationError:
                The mode is unknown, or the function is not callable.
        """
        super()._check()
        if self.mode not in MODES:
            error_msg = f'The mode of coupling is {self.mode!r}. Use one of {", ".join(MODES)}.'
            raise ValidationError(error_msg)
        if self.function is not None and not callable(self.function):
            error_msg = f'The function of coupling must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)

    def _pull(self, namespace: SimpleNamespace, other: SimpleNamespace, member: SimpleNamespace) -> Expression:
        """Return the coupling of the state of a member to the state of another member."""
        read, own = getattr(other, self.state), getattr(member, self.state)
        if self.tau != 0:
            read = delayed(read, namespace.tau)
        own = own + namespace.alpha
        value = read - own if self.function is None else self.function(read, own)
        return namespace.K * value

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the coupling of both ends of an edge, or the mean-field coupling of the target.

        Args:
            namespace:
                The namespace of an edge.

        Returns:
            laws:
                The rates of the coupled state.
        """
        source, target = namespace.source, namespace.target
        if self.mode == 'mean-field':
            pull = mean(self._pull(namespace, source, target), over=source)
            return [RateLaw(getattr(target, self.state), pull)]
        return [
            RateLaw(getattr(target, self.state), self._pull(namespace, source, target)),
            RateLaw(getattr(source, self.state), self._pull(namespace, target, source)),
        ]
