"""Define mass exchange.

A mass exchange brings a concentration towards a feed or a surface value over a residence or transfer time.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

from sage.all__sagemath_symbolics import SR, vector
from sage.modules.free_module_element import FreeModuleElement

from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.errors import ValidationError
from ....core.types import Role


class MassExchangeInteraction(Interaction):
    """Mass exchange of a concentration towards a feed over a residence or transfer time.

    Chemical engineering names it the exchange of a reactor with its feed, or a mass transfer to a surface. The rate is
    `(goal(ns) - x) / time(ns)` of a state `x` of a target, with optional noise. With a constant goal and time and an
    additive noise of amplitude `sigma`, the state follows the Ornstein-Uhlenbeck process `dx = (goal - x) / time dt +
    sigma dW`, whose stationary variance is `sigma**2 time / 2`. A goal, a time, or a noise function takes the namespace
    of the interaction, with the target as `target` and the time as `t`, and a number stands for a constant. A goal
    that is a number applies to every component of a vector state.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the element or group whose state relaxes.

        state:
            The name of the state of the target that relaxes.

        goal:
            A function of the namespace that returns the goal, or a number.

        time:
            A function of the namespace that returns the relaxation time, or a positive number.

        noise:
            A function of the namespace that returns the amplitude of the noise, a number, or `None` for no noise.

    Raises:
        ValidationError:
            The state is not a name, or a function is neither callable nor a number.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.chemistry.kinetics import MassExchangeInteraction
        >>> relaxation = MassExchangeInteraction('relaxation', target='stock', goal=10.0, time=0.5)
        >>> system = System(elements=[Element('stock', values={'x': 0.0})], interactions=[relaxation]).prepare()
        >>> round(float(system.simulate(until=5.0, dt=0.01).get_values('stock.x')[-1]), 3)
        10.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'x',
        goal: Callable[[SimpleNamespace], object] | float = 0.0,
        time: Callable[[SimpleNamespace], object] | float = 1.0,
        noise: Callable[[SimpleNamespace], object] | float | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.goal = goal
        self.time = time
        self.noise = noise

    def _check_arguments(self) -> None:
        """Raise an error for an argument the interaction cannot take.

        Raises:
            ValidationError:
                The state is not a name, or a function is neither callable nor a number.
        """
        super()._check_arguments()
        if not isinstance(self.state, str) or not self.state.isidentifier():
            error_msg = f'{type(self).__name__} {self.label!r} relaxes the state {self.state!r}, which is no name.'
            raise ValidationError(error_msg)
        for name in ('goal', 'time', 'noise'):
            value = getattr(self, name)
            if not (callable(value) or isinstance(value, int | float) or (name == 'noise' and value is None)):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace or a number.'
                raise ValidationError(error_msg)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the relaxation `(goal - x) / time` of the state and its noise.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        x = getattr(context.target, self.state)
        goal = self._evaluate(self.goal, context)
        if isinstance(goal, list | tuple):
            goal = vector(SR, goal)
        elif isinstance(x, FreeModuleElement) and not isinstance(goal, FreeModuleElement):
            goal = vector(SR, [goal] * len(x))
        self.laws_ = [RateLaw(x, (goal - x) / self._evaluate(self.time, context))]
        if self.noise is not None:
            self.laws_.append(NoiseLaw(x, self._evaluate(self.noise, context)))

    @staticmethod
    def _evaluate(value: Callable[[SimpleNamespace], object] | float, context: SimpleNamespace) -> object:
        """Return a number, or the value of a function of the namespace."""
        return value(context) if callable(value) else value
