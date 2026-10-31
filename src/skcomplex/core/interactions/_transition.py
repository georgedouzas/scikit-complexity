"""Define transitions, moves of a discrete state between its values."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, exp, log

from ...errors import ValidationError
from ...specs import RoleSpec
from ..laws import BaseLaw, EmissionLaw, JumpLaw, TransitionLaw, UpdateLaw
from ..laws.utils import neighbors, uniform, where
from ..spaces import BaseNeighborhood
from ._interaction import Interaction

FORMS = ('rate', 'probability', 'condition', 'matrix')
OVERS = ('member', 'neighbors')


class TransitionInteraction(Interaction):
    """Move of a discrete state of the target from one value to another, or between all its values by a matrix.

    A member whose state holds `source` moves to `destination` at a `rate`, as a jump in continuous time, with a
    `probability` in every step of discrete time, or in the step in which a `condition` holds. With `over='neighbors'`
    the rate and the probability are those of one neighbour, written with `namespace.target.other` for the neighbour:
    the rates of the neighbours add up, and their probabilities combine as `1 - prod(1 - p)`, the probability that at
    least one of them moves the member. With a `matrix` instead, the state takes the values `0` to `n - 1` and steps by
    the rows of a transition matrix in discrete time, the row of its current value. An `emission` matrix then draws the
    `observed` state of the target from the row that the next value of the state selects, as a hidden Markov source.
    Every function reads `namespace.target`, the states and parameters of the target, and the time `namespace.t`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose state moves.

        state:
            The name of the discrete state of the target.

        source:
            The value the state moves from.

        destination:
            The value the state moves to.

        rate:
            The function `rate(namespace)` of the rate of the move, or `None`.

        probability:
            The function `probability(namespace)` of the probability of the move in a step, or `None`.

        condition:
            The function `condition(namespace)` of the condition under which the state moves in a step, or `None`.

        matrix:
            The function `matrix(namespace)` of the transition matrix, one row and one column per value of the state,
            or `None`.

        emission:
            With a matrix, the function `emission(namespace)` of the emission matrix, one row per value of the state
            and one column per value of the observed state, or `None`.

        observed:
            The name of the state of the target that the emission matrix draws, or `None`.

        over:
            `'member'` for a rate or a probability of the member, or `'neighbors'` for that of one neighbour.

        neighborhood:
            The neighbourhood of `over='neighbors'`, or `None` for the neighbours of the network or the lattice.

    Raises:
        ValidationError:
            The transition has not exactly one form, the source and the destination are equal, an emission has no
            matrix or no observed state, `over` is unknown or goes with a condition or a matrix, or a function is not
            callable.

    Examples:
        >>> from skcomplex.core.interactions import TransitionInteraction
        >>> recovery = TransitionInteraction(
        ...     'recovery', target='hosts', state='state', source=1, destination=0, rate=lambda s: 0.5
        ... )
        >>> recovery.check().over
        'member'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'state',
        source: float = 0.0,
        destination: float = 1.0,
        rate: Callable[[SimpleNamespace], object] | None = None,
        probability: Callable[[SimpleNamespace], object] | None = None,
        condition: Callable[[SimpleNamespace], object] | None = None,
        matrix: Callable[[SimpleNamespace], Sequence[Sequence[object]]] | None = None,
        emission: Callable[[SimpleNamespace], Sequence[Sequence[object]]] | None = None,
        observed: str | None = None,
        over: str = 'member',
        neighborhood: BaseNeighborhood | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.source = source
        self.destination = destination
        self.rate = rate
        self.probability = probability
        self.condition = condition
        self.matrix = matrix
        self.emission = emission
        self.observed = observed
        self.over = over
        self.neighborhood = neighborhood

    def _check_form(self) -> None:
        """Raise an error for a transition without exactly one callable form, or with arguments its form ignores."""
        forms = [form for form in FORMS if getattr(self, form) is not None]
        if len(forms) != 1 or not callable(getattr(self, forms[0])):
            error_msg = (
                f'The transition {self.label!r} has the forms {forms}. Give exactly one of {", ".join(FORMS)}, as a '
                'function of the namespace.'
            )
            raise ValidationError(error_msg)
        unpaired = (self.emission is None) != (self.observed is None)
        if unpaired or (self.emission is not None and self.matrix is None):
            error_msg = (
                f'The transition {self.label!r} has the emission {self.emission!r} of the state {self.observed!r}. '
                'Give both with a matrix, or neither.'
            )
            raise ValidationError(error_msg)
        if self.over not in OVERS:
            error_msg = f'The transition {self.label!r} runs over {self.over!r}. Use one of {", ".join(OVERS)}.'
            raise ValidationError(error_msg)
        if self.over == 'neighbors' and forms[0] not in {'rate', 'probability'}:
            error_msg = f'The transition {self.label!r} runs over neighbours, which only a rate or a probability does.'
            raise ValidationError(error_msg)
        if self.matrix is None and self.source == self.destination:
            error_msg = f'The transition {self.label!r} moves from {self.source} to itself. Give distinct values.'
            raise ValidationError(error_msg)

    def _check_transition(self) -> None:
        """Raise an error for arguments the transition cannot take."""
        self._check_form()

    def _check(self) -> None:
        """Raise an error for a transition without exactly one form, or with arguments its form ignores.

        Raises:
            ValidationError:
                The transition has not exactly one form, the source and the destination are equal, an emission has
                no matrix or no observed state, `over` is unknown or goes with a condition or a matrix, or a function
                is not callable.
        """
        super()._check()
        self._check_transition()

    def _get_moving(self, value: Expression, source: float) -> Expression:
        """Return 1 for a member whose state holds the source value, and 0 otherwise."""
        return where(value == source, 1, 0)

    def _get_jump(self, value: Expression, source: float, destination: float, rate: Expression) -> JumpLaw:
        """Return the jump of the state from the source value to the destination at a rate."""
        return JumpLaw(rate * self._get_moving(value, source), {value: destination - source})

    def _get_step(self, value: Expression, source: float, destination: float, moves: Expression) -> UpdateLaw:
        """Return the update of the state that moves a member from the source value where a condition holds."""
        return UpdateLaw(value, where(value == source, where(moves, destination, source), value))

    def _combine_rates(self, rate: Expression) -> Expression:
        """Return the rate of the member, or the sum of the rates of its neighbours."""
        return neighbors(rate, over=self.neighborhood) if self.over == 'neighbors' else rate

    def _combine_probabilities(self, probability: Expression) -> Expression:
        """Return the probability of the member, or the probability that at least one neighbour moves it."""
        if self.over == 'member':
            return probability
        return 1 - exp(neighbors(log(1 - probability), over=self.neighborhood))

    def _get_draw(self) -> Expression:
        """Return the uniform draw of the transition in a step."""
        return uniform(re.sub(r'\W', '_', f'transition_{self.label}'))

    def _get_chain(self, namespace: SimpleNamespace, value: Expression) -> list[BaseLaw]:
        """Return the transition law of the state by its matrix, with the emission law of the observed state."""
        laws: list[BaseLaw] = []
        if self.matrix is not None:
            laws.append(TransitionLaw(value, self.matrix(namespace), label=self.label))
        if self.emission is not None and self.observed is not None:
            observed = getattr(namespace.target, self.observed)
            laws.append(EmissionLaw(observed, value, self.emission(namespace), label=self.label))
        return laws

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the jump, the update, or the transition or emission law of the state.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The law of the move.
        """
        value = getattr(namespace.target, self.state)
        if self.matrix is not None:
            return self._get_chain(namespace, value)
        if self.rate is not None:
            rate = self._combine_rates(SR(self.rate(namespace)))
            return [self._get_jump(value, self.source, self.destination, rate)]
        if self.probability is not None:
            probability = self._combine_probabilities(SR(self.probability(namespace)))
            moves = self._get_draw() < probability
        else:
            moves = SR(self.condition(namespace)) if self.condition is not None else SR(0)
        return [self._get_step(value, self.source, self.destination, moves)]
