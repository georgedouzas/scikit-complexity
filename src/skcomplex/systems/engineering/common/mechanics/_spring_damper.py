"""Define the spring and damper between two coordinates of one element."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.base import BaseComponent
from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, RateLaw
from .....core.errors import ValidationError
from .....core.types import Role

N_STATES = 2


class SpringDamperInteraction(Interaction):
    """Linear spring and damper between two coordinates `x1` and `x2` of one element.

    The spring pulls the coordinates together with the force `-k (x1 - x2)` and the damper with `-b (x1' - x2')`. Both
    forces accelerate the first coordinate divided by its mass and the second, with the opposite sign, divided by its
    own, as the suspension between the body and the wheel of a car or the pad between a rail and its sleeper.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element with the two coordinates and their masses.

        states:
            The names of the two coordinates.

        masses:
            The names of the masses of the two coordinates.

        k:
            The stiffness of the spring.

        b:
            The damping of the damper.

    Raises:
        ValidationError:
            The states or the masses are not two names.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.engineering.common.mechanics import SpringDamperInteraction
        >>> car = Element('car', values={'x1': 0.1, 'x2': 0.0, 'm1': 250.0, 'm2': 30.0})
        >>> suspension = SpringDamperInteraction('suspension', target='car', k=16000.0, b=1000.0)
        >>> System(elements=[car], interactions=[suspension]).prepare().states_
        ('car.x1', "car.x1'", 'car.x2', "car.x2'")
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        states: Sequence[str] = ('x1', 'x2'),
        masses: Sequence[str] = ('m1', 'm2'),
        k: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        b: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.states = states
        self.masses = masses
        self.k = k
        self.b = b

    def _check_names(self) -> None:
        """Raise an error for states or masses that are not two names.

        Raises:
            ValidationError:
                The states or the masses are not two names.
        """
        for name, value in (('states', self.states), ('masses', self.masses)):
            names = [value] if isinstance(value, str) else list(value)
            if len(names) != N_STATES or not all(isinstance(item, str) and item.isidentifier() for item in names):
                error_msg = f'{self.label!r} has the {name} {value!r}. Give two names, such as ("y1", "y2").'
                raise ValidationError(error_msg)

    def _check_arguments(self) -> None:
        """Raise an error for an argument the interaction cannot take.

        Raises:
            ValidationError:
                The states or the masses are not two names.
        """
        super()._check_arguments()
        self._check_names()

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the accelerations of both coordinates by the spring and by the damper.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        member, t = context.target, context.t
        first, second = (getattr(member, name) for name in self.states)
        spring = -context.k * (first - second)
        damper = -context.b * (first.diff(t) - second.diff(t))
        laws: list[BaseComponent] = []
        for state, mass, sign in ((first, self.masses[0], 1), (second, self.masses[1], -1)):
            laws += [
                RateLaw(state.diff(t, 2), sign * spring / getattr(member, mass)),
                RateLaw(state.diff(t, 2), sign * damper / getattr(member, mass)),
            ]
        self.laws_ = laws
