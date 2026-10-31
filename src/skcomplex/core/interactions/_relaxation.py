"""Define relaxation, rates that bring a state towards a goal over a time, with optional noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from ...errors import ValidationError
from ...specs import RoleSpec
from ._rate import RateInteraction


class RelaxationInteraction(RateInteraction):
    """Relaxation `(goal(ns) - x) / time(ns)` of a state `x` of a target towards a goal, with optional noise.

    With a constant goal and time and an additive noise of amplitude `sigma`, the state follows the Ornstein-Uhlenbeck
    process `dx = (goal - x) / time dt + sigma dW`, whose stationary variance is `sigma**2 time / 2`. Without noise it
    is first-order smoothing, or adaptive expectation, towards the goal. A goal or a time function takes the namespace
    of the interaction, as for `RateInteraction`, and a number stands for a constant. A goal that is a number applies
    to every component of a vector state.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

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
        >>> from skcomplex.core.elements import CompartmentElement
        >>> from skcomplex.core.interactions import RelaxationInteraction
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> smoothing = RelaxationInteraction('smoothing', target='belief', state='n', goal=10.0, time=0.5)
        >>> system = System(elements=[CompartmentElement('belief', n=0.0)], interactions=[smoothing])
        >>> trajectory = Simulation(system, dt=0.01).check().prepare().run(until=5.0).trajectory_
        >>> round(float(trajectory.values('belief.n')[-1]), 3)
        10.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'x',
        goal: Callable[[SimpleNamespace], object] | float = 0.0,
        time: Callable[[SimpleNamespace], object] | float = 1.0,
        noise: Callable[[SimpleNamespace], object] | float | None = None,
    ) -> None:
        super().__init__(label, target=target, state=state, noise=noise)
        self.goal = goal
        self.time = time

    def _check_functions(self) -> None:
        """Raise an error for a state that is not a name, or a function that is neither callable nor a number."""
        super()._check_functions()
        for name in ('goal', 'time'):
            value = getattr(self, name)
            if not (callable(value) or isinstance(value, int | float)):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace or a number.'
                raise ValidationError(error_msg)

    def _get_goal(self, namespace: SimpleNamespace) -> object:
        """Return the goal, the hook a subclass overrides to derive it from its parameters."""
        return self._evaluate(self.goal, namespace)

    def _get_time(self, namespace: SimpleNamespace) -> object:
        """Return the relaxation time, the hook a subclass overrides to derive it from its parameters."""
        return self._evaluate(self.time, namespace)

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the relaxation `(goal - x) / time` of the state.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The rate of the state.
        """
        state = getattr(namespace.target, self.state)
        goal = self._broadcast(self._get_goal(namespace), state)
        return (goal - state) / self._get_time(namespace)
