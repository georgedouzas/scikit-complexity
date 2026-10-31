"""Define spreading of a discrete state to neighbours, with recovery and a return to the inactive value."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, JumpLaw
from ..laws.utils import where
from ._neighbor import NeighborInteraction

FORMS = ('number', 'fraction')


class SpreadingInteraction(NeighborInteraction):
    """Spreading of the active value of a discrete state along the edges of a network or the bonds of a lattice.

    An inactive member becomes active at the rate `beta` times the number of its active neighbours, or times their
    fraction of its neighbours. An active member recovers at the rate `gamma`, to the inactive value, or to the removed
    value when there is one, and a removed member returns to the inactive value at the rate `omega`. Spreading runs
    along both directions of every edge, and recovery and return run once per member. The laws are jumps, so the
    state changes by random events in continuous time.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group on the nodes or sites.

        state:
            The name of the discrete state of the members.

        beta:
            The rate of spreading from one active neighbour, or from all neighbours when they are all active.

        gamma:
            The rate of recovery of an active member.

        omega:
            The rate at which a removed member returns to the inactive value.

        inactive:
            The value of the state of an inactive member.

        active:
            The value of the state of an active member.

        removed:
            The value of the state of a recovered member, or `None` for members that recover to the inactive value.

        form:
            `'number'` for a rate proportional to the number of active neighbours, or `'fraction'` for one
            proportional to their fraction of the neighbours.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the graph of the network.

    Raises:
        ValidationError:
            The form is unknown, or the values of the state are not distinct.

    Examples:
        >>> from skcomplex.core.interactions import SpreadingInteraction
        >>> spreading = SpreadingInteraction('spreading', nodes='sites', beta=2.0, gamma=1.0, form='fraction')
        >>> spreading.check().form
        'fraction'
    """

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        state: str = 'state',
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        omega: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        inactive: float = 0.0,
        active: float = 1.0,
        removed: float | None = None,
        form: str = 'number',
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.nodes = nodes
        self.state = state
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.inactive = inactive
        self.active = active
        self.removed = removed
        self.form = form

    def _check(self) -> None:
        """Raise an error for an unknown form or values of the state that are not distinct.

        Raises:
            ValidationError:
                The form is unknown, or the values of the state are not distinct.
        """
        super()._check()
        if self.form not in FORMS:
            error_msg = f'The form of spreading is {self.form!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        values = [self.inactive, self.active, *([] if self.removed is None else [self.removed])]
        if len(set(values)) < len(values):
            error_msg = (
                f'The inactive, active, and removed values of spreading are {values}. Give each a distinct value.'
            )
            raise ValidationError(error_msg)

    def _spread(self, spreader: SimpleNamespace, receiver: SimpleNamespace, beta: Expression) -> JumpLaw:
        """Return the activation of the receiving end of an edge by the spreading end."""
        share = 1 / receiver.degree if self.form == 'fraction' else 1
        held, receiving = getattr(spreader, self.state), getattr(receiver, self.state)
        rate = beta * share * where(held == self.active, 1, 0) * where(receiving == self.inactive, 1, 0)
        return JumpLaw(rate, {receiving: self.active - self.inactive})

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the spreading along both directions of an edge, the recovery, and the return of the target.

        Args:
            namespace:
                The namespace of an edge.

        Returns:
            laws:
                The jumps of the state.
        """
        source, target = namespace.source, namespace.target
        value = getattr(target, self.state)
        recovered = self.inactive if self.removed is None else self.removed
        laws: list[BaseLaw] = [
            self._spread(source, target, namespace.beta),
            self._spread(target, source, namespace.beta),
            JumpLaw(namespace.gamma * where(value == self.active, 1, 0), {value: recovered - self.active}),
        ]
        if self.removed is not None:
            rate = namespace.omega * where(value == self.removed, 1, 0)
            laws.append(JumpLaw(rate, {value: self.inactive - self.removed}))
        return laws
