"""Define leaky integration.

A leaky integrator brings a potential or an activity towards its input over a membrane time constant.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

from sage.all__sagemath_symbolics import SR, vector
from sage.modules.free_module_element import FreeModuleElement

from ....core.components import Tags
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.errors import ValidationError
from ....core.types import Role


class LeakyIntegrationInteraction(Interaction):
    """Leaky integration of a potential or an activity towards its input over a time constant.

    Neuroscience names it the leak of a membrane or of a population, as in the Wilson-Cowan and leaky integrate-and-fire
    models. The rate is `(goal(ns) - x) / time(ns)` of a state `x` of a target, with optional noise. With a constant
    goal and time and an additive noise of amplitude `sigma`, the state follows the Ornstein-Uhlenbeck process `dx =
    (goal - x) / time dt + sigma dW`, whose stationary variance is `sigma**2 time / 2`. A goal or a time function takes
    the namespace of the interaction, which reads the target as `ns.target` and the time as `ns.t`, and a number stands
    for a constant. A goal that is a number applies to every component of a vector state.

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
        >>> from skcomplex.systems.biology.neuroscience import LeakyIntegrationInteraction
        >>> relaxation = LeakyIntegrationInteraction('relaxation', target='stock', goal=10.0, time=0.5)
        >>> system = System(elements=[Element('stock', values={'x': 0.0})], interactions=[relaxation]).prepare()
        >>> round(float(system.simulate(5.0, dt=0.01).get_values('stock.x')[-1]), 3)
        10.0
    """

    law_classes = (RateLaw, NoiseLaw)

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

    def _get_tags(self) -> Tags:
        """Return the tags of the rate, and of the noise when there is one."""
        return self._get_variant_tags((RateLaw,) if self.noise is None else (RateLaw, NoiseLaw))

    def _check_arguments(self) -> None:
        """Raise an error for an argument the interaction cannot take.

        Raises:
            ValidationError:
                The state is not a name, or a function is neither callable nor a number.
        """
        super()._check_arguments()
        if not isinstance(self.state, str) or not self.state.isidentifier():
            error_msg = f'The state of {self.label!r} is {self.state!r}. Give the name of a state.'
            raise ValidationError(error_msg)
        for name in ('goal', 'time', 'noise'):
            value = getattr(self, name)
            if not (callable(value) or isinstance(value, int | float) or (name == 'noise' and value is None)):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace or a number.'
                raise ValidationError(error_msg)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the relaxation `(goal - x) / time` of the state, and its noise.

        Args:
            context:
                The namespace of the target.
        """
        super()._prepare(context)
        state = getattr(context.target, self.state)
        goal = self._broadcast(self._evaluate('goal', context), state)
        laws: list[BaseLaw] = [RateLaw(state, (goal - state) / self._evaluate('time', context))]
        if self.noise is not None:
            laws.append(NoiseLaw(state, self._evaluate('noise', context)))
        self.laws_ = laws

    def _evaluate(self, name: str, context: SimpleNamespace) -> object:
        """Return the value of a function of the namespace, or the symbol of a number."""
        value = getattr(self, name)
        return value(context) if callable(value) else getattr(context, name)

    @staticmethod
    def _broadcast(value: object, state: object) -> object:
        """Return a value with one component per component of a vector state, repeating a number."""
        if isinstance(state, FreeModuleElement) and not isinstance(value, FreeModuleElement | list | tuple):
            return vector(SR, [value] * len(state))
        return value
