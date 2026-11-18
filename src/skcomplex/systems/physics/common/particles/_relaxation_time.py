"""Define relaxation over a relaxation time.

A relaxation brings a state towards a goal over a relaxation time, with optional noise.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

from sage.all__sagemath_symbolics import SR, vector
from sage.modules.free_module_element import FreeModuleElement

from .....core.components import Tags
from .....core.components.base import BaseComponent
from .....core.components.interactions import Interaction
from .....core.components.laws import NoiseLaw, RateLaw
from .....core.components.spaces import BaseSpace
from .....core.errors import ValidationError
from .....core.specs import Role


class RelaxationTimeInteraction(Interaction):
    """Relaxation of a state towards a goal over a relaxation time.

    The state follows `x' = (goal - x) / time`, with the noise `noise dW` when one is given. With a constant goal and
    time this is the Ornstein-Uhlenbeck process, whose stationary variance is `noise**2 time / 2`. It is Newton
    cooling, a belief adapting to news, or the BGK collision term of a gas. The goal, the time, and the noise are
    numbers or functions of the namespace of the interaction, and a goal that is a number applies to every component
    of a vector state.

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
            A goal, a time, or a noise is neither callable nor a number.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.particles import RelaxationTimeInteraction
        >>> relaxation = RelaxationTimeInteraction('relaxation', target='stock', goal=10.0, time=0.5)
        >>> system = System(elements=[Element('stock', values={'x': 0.0})], interactions=[relaxation]).prepare()
        >>> round(float(system.simulate(until=5.0, dt=0.01).get_values('stock.x')[-1]), 3)
        10.0
    """

    law_classes = (RateLaw, NoiseLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the rate law, with a noise law for a noise."""
        return self._get_variant_tags((RateLaw, NoiseLaw) if self.noise is not None else (RateLaw,))

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

    def _evaluate(self, name: str, context: SimpleNamespace) -> object:
        """Return the value of a function of the namespace, or the symbol of a number."""
        value = getattr(self, name)
        return value(context) if callable(value) else getattr(context, name)

    def _get_goal(self, context: SimpleNamespace) -> object:
        """Return the goal, the hook a subclass overrides to derive it from its parameters."""
        return self._evaluate('goal', context)

    def _get_time(self, context: SimpleNamespace) -> object:
        """Return the relaxation time."""
        return self._evaluate('time', context)

    def _get_noise(self, context: SimpleNamespace) -> object:
        """Return the amplitude of the noise, or `None` for none."""
        return None if self.noise is None else self._evaluate('noise', context)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the relaxation of the state of the target and its noise.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.

        Raises:
            ValidationError:
                A goal, a time, or a noise is neither callable nor a number.
        """
        super()._prepare(context)
        for name in ('goal', 'time', 'noise'):
            value = getattr(self, name)
            if not (callable(value) or isinstance(value, int | float) or (name == 'noise' and value is None)):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace or a number.'
                raise ValidationError(error_msg)
        if isinstance(context, SimpleNamespace):
            x = getattr(context.target, self.state)
            goal = self._get_goal(context)
            if isinstance(x, FreeModuleElement) and not isinstance(goal, FreeModuleElement):
                goal = vector(SR, [goal] * len(x))
            laws: list[BaseComponent] = [RateLaw(x.diff(context.t), (goal - x) / self._get_time(context))]
            noise = self._get_noise(context)
            if noise is not None:
                laws.append(NoiseLaw(x, noise))
            self.laws_ = laws
