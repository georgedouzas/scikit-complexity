"""Define the interaction that gives one state its equation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import ClassVar

from ....core.components import Tags
from ....core.components.base import BaseComponent
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.specs import Role


class EquationInteraction(Interaction):
    """An interaction that gives one state of an element its rate or its next value.

    In continuous time the equation is the rate of the state, or of its derivative of order `k` when the state is named
    with `k` primes, such as `"x''"` for an acceleration, and a noise amplitude adds a noise in the Itô or Stratonovich
    convention. In discrete time the equation is the next value of the state, read from the values at the start of the
    step. An equation and a noise are functions of the namespace whose `target` holds the values of the element and
    whose `t` holds the time.

    Unlike `Interaction`, which a subclass gives laws of its own, it takes the equation as a function. Choose it when
    one state follows one formula, with no class of its own.

    It appears in many fields. The decay of a tank, the logistic map of a population, and the income of an economy are
    each one equation of one state.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the element whose state the equation changes.

        state:
            The name of the state, with a prime for every order of the derivative the equation gives.

        equation:
            The rate or the next value as a function of the namespace, or `None` for a noise alone.

        noise:
            The noise amplitude as a function of the namespace or a number, or `None` for none.

        convention:
            The stochastic calculus of the noise, `'ito'` or `'stratonovich'`.

        time:
            The time of the equation, `'continuous'` for a rate or `'discrete'` for a next value.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.general.dynamics import EquationInteraction
        >>> decay = EquationInteraction('decay', target='tank', state='x', equation=lambda s: -s.target.k * s.target.x)
        >>> system = System(elements=[Element('tank', values={'x': 1.0, 'k': 0.5})], interactions=[decay])
        >>> round(float(system.prepare().simulate(until=2.0, dt=2.0).get_values('tank.x')[-1]), 4)
        0.3679
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw, NoiseLaw, UpdateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str,
        equation: Callable[[SimpleNamespace], object] | None = None,
        noise: Callable[[SimpleNamespace], object] | float | None = None,
        convention: str = 'ito',
        time: str = 'continuous',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.equation = equation
        self.noise = noise
        self.convention = convention
        self.time = time

    def _get_tags(self) -> Tags:
        """Return the tags of the update, or of the rate and the noise the equation has."""
        if self.time == 'discrete':
            return self._get_variant_tags((UpdateLaw,))
        classes = (RateLaw,) if self.equation is not None else ()
        return self._get_variant_tags((*classes, NoiseLaw) if self.noise is not None else classes)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate and the noise, or the next value, of the state.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the element.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        name = self.state.rstrip("'")
        order = len(self.state) - len(name)
        state = getattr(context.target, name)
        laws: list[BaseComponent] = []
        if self.time == 'discrete' and self.equation is not None:
            laws.append(UpdateLaw(state, self.equation(context)))
        elif self.equation is not None:
            laws.append(RateLaw(state.diff(context.t, max(order, 1)), self.equation(context)))
        if self.time != 'discrete' and self.noise is not None:
            noise = self.noise(context) if callable(self.noise) else self.noise
            laws.append(NoiseLaw(state.diff(context.t, max(order, 1)), noise, convention=self.convention))
        self.laws_: list[BaseComponent] = laws
