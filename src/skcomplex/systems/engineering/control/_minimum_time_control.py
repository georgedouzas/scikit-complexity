"""Minimum-time control: a linear plant steered to the origin.

A sampled bang-bang control steers it.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, abs_symbolic, sqrt
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import sign, where
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ParameterError
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.signals import PulseInteraction

PLANTS = {'free_particle': 0.0, 'harmonic_oscillator': -1.0, 'saddle': 1.0}
CIRCLES_END = 2


class BangBangPlantElement(Element):
    """Second-order linear plant under a bang-bang control.

    The control is held between samples, and it steers the plant to the origin. The plant follows `x' = y` and
    `y' = a x + u` with the control `u` of size 1, `a = 0` for a free particle, `a = -1` for a harmonic oscillator, and
    `a = 1` for a saddle. The control stays constant between samples, and at every sample takes `-1` above the switching
    curve `y = g(x)` of the plant and `1` below it. The curve is made of the two trajectories of the plant under `u = 1`
    and `u = -1` that reach the origin: `g(x) = -sign(x) sqrt(2 |x|)` for the free particle, the half circles
    `g(x) = sqrt(1 - (x + 1)^2)` on `[-2, 0]` and `-sqrt(1 - (x - 1)^2)` on `[0, 2]`, and 0 outside, for the harmonic
    oscillator, and `g(x) = sqrt(x^2 - 2 x)` for `x <= 0` and `-sqrt(x^2 + 2 x)` for `x >= 0` for the saddle.

    Args:
        label:
            The label of the plant.

        visual:
            How the plant is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        y:
            The initial velocity.

        u:
            The initial control, -1 or 1.

        a:
            The coefficient of the position in the acceleration, 0, -1, or 1.

    Examples:
        >>> from skcomplex.systems.engineering.control import BangBangPlantElement
        >>> BangBangPlantElement('plant', a=-1.0).a
        -1.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
        u: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([-1, 1]))] = 1.0,
        a: Annotated[float, ParameterSpec(domain=FiniteEnumeratedSet([-1, 0, 1]))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.u = u
        self.a = a

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the motion of the plant under the held control.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.x, s.y), RateLaw(s.y, s.a * s.x + s.u), RateLaw(s.u, 0)]

    @staticmethod
    def _get_switching_curve(plant: str, x: Expression) -> Expression:
        """Return the velocity `g(x)` of the switching curve of a plant at a position.

        Args:
            plant:
                The plant, `'free_particle'`, `'harmonic_oscillator'`, or `'saddle'`.

            x:
                The position, a number or a symbol.

        Returns:
            velocity:
                The velocity of the switching curve.

        Raises:
            ParameterError:
                The plant is unknown.
        """
        if plant == 'free_particle':
            curve: Expression = -sign(x) * sqrt(2 * abs_symbolic(x))
        elif plant == 'harmonic_oscillator':
            left = sqrt(abs_symbolic(1 - (x + 1) ** 2))
            right = -sqrt(abs_symbolic(1 - (x - 1) ** 2))
            curve = where(abs_symbolic(x) > CIRCLES_END, 0, where(x < 0, left, right))
        elif plant == 'saddle':
            curve = where(x <= 0, sqrt(abs_symbolic(x**2 - 2 * x)), -sqrt(abs_symbolic(x**2 + 2 * x)))
        else:
            error_msg = f'Unknown plant {plant!r}. Use one of {", ".join(PLANTS)}.'
            raise ParameterError(error_msg)
        return curve

    @staticmethod
    def _get_control(plant: str, x: Expression, y: Expression) -> Expression:
        """Return the bang-bang control of a plant.

        It is -1 above its switching curve and 1 below it.

        Args:
            plant:
                The plant, `'free_particle'`, `'harmonic_oscillator'`, or `'saddle'`.

            x:
                The position.

            y:
                The velocity.

        Returns:
            control:
                The control.
        """
        control: Expression = where(y > BangBangPlantElement._get_switching_curve(plant, x), -1, 1)
        return control


class MinimumTimeControlSystem(BaseSystem):
    """A linear plant steered to the origin of its phase space in minimum time.

    A sampled bang-bang control steers it. The plant follows `x'' = a x + u` with `|u| <= 1`: a free particle,
    `x'' = u`, a harmonic oscillator, `x'' + x = u`, or an unstable saddle, `x'' - x = u`. Pontryagin's maximum
    principle makes the minimum-time control bang-bang, `u = ±1`, with the switch on the curve of the two trajectories
    under `u = 1` and `u = -1` that reach the origin: the control is -1 above the curve and 1 below it. As in Dilão's
    examples, the controller samples the position and the velocity every `interval` and holds the control until the next
    sample, so the plant reaches a neighbourhood of the origin, whose size shrinks with the interval, and stays there. A
    free particle from `(x, v)` above the curve reaches the origin at the time `v + 2 sqrt(x + v^2 / 2)`, about 6.47 for
    the defaults. Only the states between the lines `y = -(x + 1)` and `y = -(x - 1)` of the saddle are controllable,
    and from the others the plant escapes to infinity.

    The system has the position `plant.x`, the velocity `plant.y`, and the control `plant.u`, which the samples `sample`
    set.

    Args:
        x:
            The initial position.

        v:
            The initial velocity.

        plant:
            The plant, `'free_particle'`, `'harmonic_oscillator'`, or `'saddle'`.

        interval:
            The time between two samples of the controller.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ParameterError:
            The plant is unknown, or the interval is not positive.

    Examples:
        >>> from skcomplex.systems.engineering.control import MinimumTimeControlSystem
        >>> MinimumTimeControlSystem(plant='saddle', x=0.5, v=0.0).prepare().system_.elements[0].u
        -1.0
    """

    field = 'engineering/control'
    topics = ('control', 'bang-bang', 'optimal control', 'pontryagin', 'events', 'ode')
    references = ('Pontryagin et al. 1962', 'Dilão 2023')
    until = 10.0
    dt = 0.01

    def __init__(
        self,
        x: float = 3.0,
        v: float = 2.0,
        *,
        plant: str = 'free_particle',
        interval: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.v = v
        self.plant = plant
        self.interval = interval

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        if self.plant not in PLANTS:
            error_msg = f'Unknown plant {self.plant!r}. Use one of {", ".join(PLANTS)}.'
            raise ParameterError(error_msg)
        if not self.interval > 0:
            error_msg = f'The interval of the samples is {self.interval!r}. Give a positive number.'
            raise ParameterError(error_msg)
        start = float(BangBangPlantElement._get_control(self.plant, self.x, self.v))
        sample = {'u': lambda s: BangBangPlantElement._get_control(self.plant, s.target.x, s.target.y)}
        return CompositeSystem(
            self.label,
            elements=[BangBangPlantElement('plant', x=self.x, y=self.v, u=start, a=PLANTS[self.plant])],
            interactions=[
                PulseInteraction('sample', target='plant', reset=sample, period=self.interval, start=self.interval),
            ],
        )
