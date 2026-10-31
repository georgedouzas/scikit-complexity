"""Define choice interactions, which set a discrete state to an option drawn from the utilities of the options."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, exp, oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, UpdateLaw
from ..laws.utils import categorical, uniform, where
from ._interaction import Interaction

RULES = ('logit', 'metropolis', 'fermi')
MIN_OPTIONS = 2


class ChoiceInteraction(Interaction):
    """Choice of every member of a group among options, drawn every step from the utilities of the options by a rule.

    The utility `u_k` of option `k` is a symbolic expression that `utility(namespace, option)` returns, where
    `namespace.members` holds the states and parameters of the group and `namespace.intensity` is the intensity of
    choice `beta`. The rule `'logit'`, which is the heat-bath rule, draws option `k` with probability
    `exp(beta u_k) / sum_j exp(beta u_j)`. The rules `'metropolis'` and `'fermi'` propose one of the other options
    uniformly and accept it with probability `min(1, exp(beta (u_new - u_old)))` or
    `1 / (1 + exp(-beta (u_new - u_old)))`. Every member draws at once in every step of a discrete-time system.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        members:
            The path of the element or group whose members choose.

        state:
            The name of the discrete state of the members that holds the chosen option.

        options:
            The values the state can take, at least two.

        utility:
            The function `utility(namespace, option)` that returns the utility of an option as a symbolic expression.

        rule:
            The rule of the draw, `'logit'`, `'metropolis'`, or `'fermi'`.

        intensity:
            The intensity of choice `beta`, zero for uniform choice.

    Raises:
        ValidationError:
            The rule is unknown, there are fewer than two options, or the utility is not callable.

    Examples:
        >>> from skcomplex.core.interactions import ChoiceInteraction
        >>> from skcomplex.core.laws.utils import mean
        >>> choice = ChoiceInteraction(
        ...     'choice', members='spins', state='spin', options=(-1, 1), utility=lambda s, o: o * mean(s.members.spin)
        ... )
        >>> choice.check().rule
        'logit'
    """

    def __init__(
        self,
        label: str,
        *,
        members: Annotated[str, RoleSpec()],
        state: str,
        options: Sequence[float],
        utility: Callable[[SimpleNamespace, float], object],
        rule: str = 'logit',
        intensity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.members = members
        self.state = state
        self.options = options
        self.utility = utility
        self.rule = rule
        self.intensity = intensity

    def _check(self) -> None:
        """Raise an error for an unknown rule, fewer than two options, or a utility that cannot be called.

        Raises:
            ValidationError:
                The type declares states, the rule is unknown, there are fewer than two options, or the utility is
                not callable.
        """
        super()._check()
        if self.rule not in RULES:
            error_msg = f'Unknown rule {self.rule!r} of {self.label!r}. Use one of {RULES}.'
            raise ValidationError(error_msg)
        if len(self.options) < MIN_OPTIONS:
            error_msg = f'{self.label!r} chooses among {list(self.options)}. Give at least {MIN_OPTIONS} options.'
            raise ValidationError(error_msg)
        if not callable(self.utility):
            error_msg = f'The utility of {self.label!r} must be callable, but it is {self.utility!r}.'
            raise ValidationError(error_msg)

    def _select(self, index: Expression, values: Sequence[Expression]) -> Expression:
        """Return the value of the option at a symbolic index."""
        return sum((where(index == position, value, 0) for position, value in enumerate(values)), SR(0))

    def _draw_softmax(self, state: Expression, utilities: Sequence[Expression], intensity: Expression) -> Expression:
        """Return the option drawn with probabilities proportional to `exp(beta u_k)`."""
        index = categorical(f'{state}_option', [exp(intensity * utility) for utility in utilities])
        return self._select(index, [SR(option) for option in self.options])

    def _draw_proposal(self, state: Expression, utilities: Sequence[Expression], intensity: Expression) -> Expression:
        """Return a uniform proposal of another option, accepted by the Metropolis or the Fermi probability."""
        index = categorical(f'{state}_proposal', [where(state == option, 0, 1) for option in self.options])
        proposal = self._select(index, [SR(option) for option in self.options])
        gain = self._select(index, utilities) - sum(
            (where(state == option, utility, 0) for option, utility in zip(self.options, utilities, strict=True)),
            SR(0),
        )
        if self.rule == 'metropolis':
            acceptance = where(gain >= 0, 1, exp(intensity * gain))
        else:
            acceptance = 1 / (1 + exp(-intensity * gain))
        return where(uniform(f'{state}_acceptance') < acceptance, proposal, state)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the update of the state of every member to the option it draws.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.members` and `namespace.intensity`.

        Returns:
            laws:
                The update law of the state.
        """
        state = getattr(namespace.members, self.state)
        utilities = [SR(self.utility(namespace, option)) for option in self.options]
        if self.rule == 'logit':
            return [UpdateLaw(state, self._draw_softmax(state, utilities, namespace.intensity))]
        return [UpdateLaw(state, self._draw_proposal(state, utilities, namespace.intensity))]
