"""Define spreading of the active value of a discrete state to neighbours.

The spreading comes with recovery and a return.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import TransitionInteraction
from .....core.components.laws import BaseLaw, JumpLaw
from .....core.components.laws.utils import neighbors, where
from .....core.components.spaces import AdjacencyNeighborhood
from .....core.errors import ValidationError
from .....core.specs import ParameterSpec, RoleSpec, Tags

FORMS = ('number', 'fraction')


class SpreadingInteraction(TransitionInteraction):
    """Spreading of the active value of a discrete state to neighbours.

    The neighbours are along the edges of a network or the bonds of a lattice.

    An inactive member becomes active at the rate `beta` times the number of its active neighbours, or times their
    fraction of its neighbours. An active member recovers at the rate `gamma`, to the inactive value, or to the removed
    value when there is one, and a removed member returns to the inactive value at the rate `omega`. Every move is a
    transition at a rate, so the state changes by random events in continuous time.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
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
            The layer of the network whose edges the spreading runs on, or `None` for the graph of the network or the
            neighbourhood of the lattice.

    Raises:
        ValidationError:
            The form is unknown, or the values of the state are not distinct.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import SpreadingInteraction
        >>> spreading = SpreadingInteraction('spreading', target='sites', beta=2.0, gamma=1.0, form='fraction')
        >>> spreading.form
        'fraction'
    """

    law_classes = (JumpLaw,)

    def _get_tags(self) -> Tags:
        """Return the tags of the jumps at a rate, the only laws of the interaction."""
        return self._get_variant_tags(self.law_classes)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
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
        super().__init__(label, target=target, state=state)
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.inactive = inactive
        self.active = active
        self.removed = removed
        self.form = form
        self.layer = layer

    def _check_transition(self) -> None:
        """Raise an error for an unknown form or values of the state that are not distinct.

        Raises:
            ValidationError:
                The form is unknown, or the values of the state are not distinct.
        """
        if self.form not in FORMS:
            error_msg = f'The form of spreading is {self.form!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        values = [self.inactive, self.active, *([] if self.removed is None else [self.removed])]
        if len(set(values)) < len(values):
            error_msg = (
                f'The inactive, active, and removed values of spreading are {values}. Give each a distinct value.'
            )
            raise ValidationError(error_msg)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the spreading, the recovery, and the return of every member.

        The spreading comes from the active neighbours.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The jumps of the state.
        """
        member = self._get_member(namespace, 'target')
        value = getattr(member, self.state)
        over = None if self.layer is None else AdjacencyNeighborhood(layer=self.layer)
        reduce = 'mean' if self.form == 'fraction' else 'sum'
        held = neighbors(where(getattr(member.other, self.state) == self.active, 1, 0), over=over, reduce=reduce)
        recovered = self.inactive if self.removed is None else self.removed
        laws: list[BaseLaw] = [
            self._get_jump(value, self.inactive, self.active, self._get_parameter_symbol('beta', namespace) * held),
            self._get_jump(value, self.active, recovered, self._get_parameter_symbol('gamma', namespace)),
        ]
        if self.removed is not None:
            laws.append(
                self._get_jump(value, self.removed, self.inactive, self._get_parameter_symbol('omega', namespace)),
            )
        return laws
