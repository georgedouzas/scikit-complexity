"""Second-order linear plant under a bang-bang control, held between samples, that steers it to the origin."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, abs_symbolic, sqrt
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import sign, where
from ....core.errors import ParameterError
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

PLANTS = {'free_particle': 0.0, 'harmonic_oscillator': -1.0, 'saddle': 1.0}


CIRCLES_END = 2


class BangBangPlantElement(Element):
    """Second-order linear plant under a bang-bang control, held between samples, that steers it to the origin.

    The plant follows `x' = y` and `y' = a x + u` with the control `u` of size 1, `a = 0` for a free particle, `a = -1`
    for a harmonic oscillator, and `a = 1` for a saddle. The control stays constant between samples, and at every
    sample takes `-1` above the switching curve `y = g(x)` of the plant and `1` below it. The curve is made of the
    two trajectories of the plant under `u = 1` and `u = -1` that reach the origin: `g(x) = -sign(x) sqrt(2 |x|)` for
    the free particle, the half circles `g(x) = sqrt(1 - (x + 1)^2)` on `[-2, 0]` and `-sqrt(1 - (x - 1)^2)` on
    `[0, 2]`, and 0 outside, for the harmonic oscillator, and `g(x) = sqrt(x^2 - 2 x)` for `x <= 0` and
    `-sqrt(x^2 + 2 x)` for `x >= 0` for the saddle.

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
    def _switching_curve(plant: str, x: Expression) -> Expression:
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
    def _control(plant: str, x: Expression, y: Expression) -> Expression:
        """Return the bang-bang control of a plant, -1 above its switching curve and 1 below it.

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
        control: Expression = where(y > BangBangPlantElement._switching_curve(plant, x), -1, 1)
        return control
