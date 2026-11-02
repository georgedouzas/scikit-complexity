"""Define the spring and damper between two coordinates of one element, each divided by its own mass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from .....core.base import BasePart
from .....core.interactions import GradientInteraction
from .....core.laws import RateLaw
from .....core.laws.utils import derivative
from .....errors import ValidationError
from .....specs import RoleSpec

N_STATES = 2


class SpringDamperInteraction(GradientInteraction):
    """Linear spring and damper between two coordinates `x1` and `x2` of order 2 of one element.

    The spring is the gradient of the energy `k (x1 - x2)^2 / 2`, and the damper adds `-b (x1' - x2')` to the first
    coordinate and its opposite to the second, both divided by the divisor of every coordinate, such as its own mass.
    The stiffness and the damping are named parameters of the element.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element with the two coordinates.

        states:
            The names of the two coordinates.

        stiffness:
            The name of the parameter of the stiffness `k`.

        damping:
            The name of the parameter of the damping `b`.

        divisor:
            The divisor of both coordinates, or a mapping from every coordinate to its own divisor.

    Raises:
        ValidationError:
            The states are not two names, the stiffness or the damping is not a name, or a divisor is invalid.

    Examples:
        >>> from skcomplex.models.engineering.common.mechanics import SpringDamperInteraction
        >>> SpringDamperInteraction('suspension', target='car', states=('y1', 'y2')).stiffness
        'k'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        states: Sequence[str] = ('x1', 'x2'),
        stiffness: str = 'k',
        damping: str = 'b',
        divisor: str | Callable[[SimpleNamespace], object] | Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(label, target=target, states=states, divisor=divisor)
        self.stiffness = stiffness
        self.damping = damping

    def _check(self) -> None:
        """Raise an error for states that are not two names, or a stiffness or damping that is not a name.

        Raises:
            ValidationError:
                The states are not two names, the stiffness or the damping is not a name, or a divisor is invalid.
        """
        super()._check()
        if len(self._get_names()) != N_STATES:
            error_msg = f'{self.label!r} joins the states {self.states!r}. Give two states.'
            raise ValidationError(error_msg)
        for name in (self.stiffness, self.damping):
            if not (isinstance(name, str) and name.isidentifier()):
                error_msg = f'{self.label!r} reads the parameter {name!r}. Give the name of a parameter.'
                raise ValidationError(error_msg)

    def _get_potential(self, namespace: SimpleNamespace) -> Expression:
        """Return the energy `k (x1 - x2)^2 / 2` of the spring.

        Args:
            namespace:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The energy of the spring.
        """
        first, second = (getattr(namespace.target, name) for name in self._get_names())
        return getattr(namespace.target, self.stiffness) * (first - second) ** 2 / 2

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the rates of the spring and of the damper of both coordinates.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the two coordinates.
        """
        laws = super()._get_laws(namespace)
        member = namespace.target
        names = self._get_names()
        first, second = (getattr(member, name) for name in names)
        force = -getattr(member, self.damping) * (derivative(first, 1) - derivative(second, 1))
        for name, state, sign in ((names[0], first, 1), (names[1], second, -1)):
            divisor = self._get_divisor(member, name)
            laws.append(RateLaw(state, sign * force if divisor is None else sign * force / divisor))
        return laws
