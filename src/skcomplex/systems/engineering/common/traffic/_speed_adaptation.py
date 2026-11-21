"""Define speed adaptation.

A speed adaptation brings a speed or a flow towards a desired one over an adaptation time, with optional noise.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

from .....core.components import Tags
from .....core.components.base import BaseComponent
from .....core.components.interactions import Interaction
from .....core.components.laws import NoiseLaw, RateLaw
from .....core.errors import ValidationError
from .....core.types import Role

type Setting = Callable[[SimpleNamespace], object] | float


class SpeedAdaptationInteraction(Interaction):
    """Adaptation of a speed or a flow towards a desired one over an adaptation time.

    Traffic engineering names it the speed adaptation of a driver, as in the optimal velocity and Payne-Whitham models.
    The rate is `(goal(s) - x) / time(s)` of a value `x` of the target, with optional noise. With a constant goal and
    time and an additive noise of amplitude `sigma`, the value follows the Ornstein-Uhlenbeck process
    `dx = (goal - x) / time dt + sigma dW`, whose stationary variance is `sigma**2 time / 2`. The value is a state of
    the target, such as the flow `Q` of a road, or the speed `"x'"` of a car, the derivative of its position. A goal, a
    time, or a noise is a number or a function of the namespace of the interaction, whose target is `s.target`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the element or group whose value relaxes.

        state:
            The name of the value of the target that relaxes, with a prime for the derivative of a value.

        goal:
            A function of the namespace that returns the goal, or a number.

        time:
            A function of the namespace that returns the relaxation time, or a positive number.

        noise:
            A function of the namespace that returns the amplitude of the noise, a number, or `None` for no noise.

    Raises:
        ValidationError:
            A goal, a time, or a noise is neither callable nor a number, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.engineering.common.traffic import SpeedAdaptationInteraction
        >>> relaxation = SpeedAdaptationInteraction('relaxation', target='stock', goal=10.0, time=0.5)
        >>> system = System(elements=[Element('stock', values={'x': 0.0})], interactions=[relaxation]).prepare()
        >>> round(float(system.simulate(until=5.0, dt=0.01).get_values('stock.x')[-1]), 3)
        10.0
    """

    law_classes = (RateLaw, NoiseLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'x',
        goal: Setting = 0.0,
        time: Setting = 1.0,
        noise: Setting | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.goal = goal
        self.time = time
        self.noise = noise

    def _get_tags(self) -> Tags:
        """Return the tags of the rate, with a noise when the interaction has one.

        It has one when the noise is given or a subclass writes `_get_noise`.
        """
        noisy = self.noise is not None or type(self)._get_noise is not SpeedAdaptationInteraction._get_noise
        return self._get_variant_tags((RateLaw, NoiseLaw) if noisy else (RateLaw,))

    def _check_arguments(self) -> None:
        """Raise an error for a goal, a time, or a noise that is no function or number.

        Raises:
            ValidationError:
                A goal, a time, or a noise is neither callable nor a number.
        """
        super()._check_arguments()
        for name in ('goal', 'time', 'noise'):
            value = getattr(self, name)
            if not (callable(value) or isinstance(value, int | float) or (name == 'noise' and value is None)):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace or a number.'
                raise ValidationError(error_msg)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the relaxation of the value of the target and its noise.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s, t = context, context.t
        position = getattr(s.target, self.state.rstrip("'"))
        order = len(self.state) - len(self.state.rstrip("'"))
        value = position.diff(t, order) if order else position
        laws: list[BaseComponent] = [RateLaw(value.diff(t), (self._get_goal(s) - value) / self._get_time(s))]
        noise = self._get_noise(s)
        if noise is not None:
            laws.append(NoiseLaw(position.diff(t, order + 1), noise))
        self.laws_ = laws

    @staticmethod
    def _evaluate(value: Setting, namespace: SimpleNamespace) -> object:
        """Return a number, or the value of a function of the namespace."""
        return value(namespace) if callable(value) else value

    def _get_goal(self, namespace: SimpleNamespace) -> object:
        """Return the goal, the hook a subclass overrides to derive it from its parameters."""
        return self._evaluate(self.goal, namespace)

    def _get_time(self, namespace: SimpleNamespace) -> object:
        """Return the relaxation time, the hook a subclass overrides to derive it from its parameters."""
        return self._evaluate(self.time, namespace)

    def _get_noise(self, namespace: SimpleNamespace) -> object:
        """Return the amplitude of the noise, or `None`, the hook a subclass overrides."""
        return None if self.noise is None else self._evaluate(self.noise, namespace)
