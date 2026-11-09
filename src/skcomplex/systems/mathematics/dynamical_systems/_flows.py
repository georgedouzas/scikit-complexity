"""Dynamical systems of flows.

The module covers chaotic and conservative flows, delayed feedback, and singular growth.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import abs_symbolic, oo, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import delayed, derivative
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
from ..common.flows import LorenzElement
from ..common.flows._rossler import RosslerElement


class LangfordElement(Element):
    """Langford flow, a rotation in `x, y` with the height `z`.

    Its radius and height couple into a torus that breaks into chaos.

    The flow is `x' = (z - b) x - d y`, `y' = d x + (z - b) y`, and
    `z' = c + a z - z³ / 3 - (x² + y²)(1 + e z) + f z x³`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial height.

        a:
            The linear growth of the height.

        b:
            The height above which the rotation grows.

        c:
            The constant drive of the height.

        d:
            The angular frequency of the rotation.

        e:
            The coupling of the radius and the height.

        f:
            The term that breaks the rotational symmetry.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LangfordElement
        >>> LangfordElement('langford', d=3.0).d
        3.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.1,
        y: Annotated[object, StateSpec()] = 0.0,
        z: Annotated[object, StateSpec()] = 0.0,
        a: Annotated[object, ParameterSpec()] = 0.95,
        b: Annotated[object, ParameterSpec()] = 0.7,
        c: Annotated[object, ParameterSpec()] = 0.6,
        d: Annotated[object, ParameterSpec()] = 3.5,
        e: Annotated[object, ParameterSpec()] = 0.25,
        f: Annotated[object, ParameterSpec()] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the three coordinates.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        radius = s.x**2 + s.y**2
        return [
            RateLaw(s.x, (s.z - s.b) * s.x - s.d * s.y),
            RateLaw(s.y, s.d * s.x + (s.z - s.b) * s.y),
            RateLaw(s.z, s.c + s.a * s.z - s.z**3 / 3 - radius * (1 + s.e * s.z) + s.f * s.z * s.x**3),
        ]


class MixedFlowElement(Element):
    """Sprott's flow, conservative and dissipative at once.

    The flow is `x' = y + 2xy + xz`, `y' = 1 - 2x² + yz`, `z' = x - x² - y²`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import MixedFlowElement
        >>> MixedFlowElement('flow', x=2.0).x
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        y: Annotated[object, StateSpec()] = 0.0,
        z: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the three coordinates.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [
            RateLaw(s.x, s.y + 2 * s.x * s.y + s.x * s.z),
            RateLaw(s.y, 1 - 2 * s.x**2 + s.y * s.z),
            RateLaw(s.z, s.x - s.x**2 - s.y**2),
        ]


class NoseHooverElement(Element):
    """Nosé-Hoover oscillator `x' = y`, `y' = y z - x`, `z' = 1 - y²`.

    It is a conservative flow with tori and chaos.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        y:
            The initial momentum.

        z:
            The initial friction of the thermostat.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import NoseHooverElement
        >>> NoseHooverElement('oscillator', y=1.0).y
        1.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        y: Annotated[object, StateSpec()] = 0.1,
        z: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the position, the momentum, and the friction.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.y), RateLaw(s.y, s.y * s.z - s.x), RateLaw(s.z, 1 - s.y**2)]


class LorenzSystem(BaseSystem):
    """The Lorenz-63 system of convection.

    The system is chaotic on a butterfly-shaped attractor for its standard parameters.

    The system has the states `lorenz.x`, `lorenz.y`, and `lorenz.z`.

    Args:
        x:
            The initial convection intensity.

        y:
            The initial temperature difference between rising and falling currents.

        z:
            The initial deviation of the vertical temperature profile.

        sigma:
            The Prandtl number.

        rho:
            The Rayleigh number relative to its critical value.

        beta:
            The geometric factor.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LorenzSystem
        >>> LorenzSystem(rho=20.0).prepare().system_.elements[0].rho
        20.0
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'ode', 'strange attractor')
    references = ('Lorenz 1963', 'Datseris and Parlitz 2022')
    until = 50.0
    dt = 0.01

    def __init__(
        self,
        x: float = 1.0,
        y: float = 1.0,
        z: float = 1.0,
        *,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        return CompositeSystem(
            self.label,
            elements=[
                LorenzElement('lorenz', x=self.x, y=self.y, z=self.z, sigma=self.sigma, rho=self.rho, beta=self.beta),
            ],
        )


class RosslerSystem(BaseSystem):
    """The Rössler system.

    It is a chaotic flow with one folded band for its standard parameters.

    The system has the states `rossler.x`, `rossler.y`, and `rossler.z`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        a:
            The feedback of the second coordinate on itself.

        b:
            The constant drive of the third coordinate.

        c:
            The threshold of the third coordinate's growth.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import RosslerSystem
        >>> RosslerSystem(c=4.0).prepare().system_.elements[0].c
        4.0
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'ode', 'strange attractor')
    references = ('Rössler 1976',)
    until = 200.0
    dt = 0.05

    def __init__(
        self,
        x: float = 1.0,
        y: float = 1.0,
        z: float = 1.0,
        *,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.c = c

    def _build_system(self) -> CompositeSystem:
        """Return the Rössler system.

        It is a chaotic flow with one folded band for its standard parameters.
        """
        return CompositeSystem(
            self.label,
            elements=[RosslerElement('rossler', x=self.x, y=self.y, z=self.z, a=self.a, b=self.b, c=self.c)],
        )


class SprottJerkSystem(DynamicalSystem):
    """Sprott's jerk equation `x''' = -A x'' - x' + |x| - 1`, chaotic near A = 0.6.

    The system has the state `jerk.x` of order 3 and its derivatives `jerk.x_d1` and `jerk.x_d2`.

    Args:
        x:
            The initial position.

        velocity:
            The initial first derivative.

        acceleration:
            The initial second derivative.

        A:
            The damping of the second derivative.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SprottJerkSystem
        >>> SprottJerkSystem(A=0.5).prepare().system_.elements[0].get_value('A')
        0.5
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'ode', 'third order')
    references = ('Sprott 1997', 'Datseris and Parlitz 2022')
    until = 200.0
    dt = 0.1

    def __init__(
        self,
        x: float = 0.0,
        velocity: float = 0.0,
        acceleration: float = 1.0,
        *,
        A: float = 0.6,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.velocity = velocity
        self.acceleration = acceleration
        self.A = A

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'jerk'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'x': DeclaredVariableSpec(self.x, StateSpec(order=3, initial=(self.velocity, self.acceleration))),
            'A': DeclaredVariableSpec(self.A, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return the third derivative of the position."""
        return {
            'x': lambda s: (
                -s.target.A * derivative(s.target.x, 2) - derivative(s.target.x, 1) + abs_symbolic(s.target.x) - 1
            ),
        }


class FiniteTimeBlowUpSystem(DynamicalSystem):
    """Hyperbolic growth `x' = a x^p`.

    For p > 1 the growth diverges at the finite time `x0^(1 - p) / (a (p - 1))`.

    The system has the state `growth.x`.

    Args:
        x:
            The initial value, positive.

        a:
            The growth coefficient.

        p:
            The growth exponent.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import FiniteTimeBlowUpSystem
        >>> FiniteTimeBlowUpSystem(p=3.0).prepare().system_.elements[0].get_value('p')
        3.0
    """

    field = 'mathematics/dynamical_systems'
    topics = ('ode', 'singularity', 'hyperbolic growth')
    references = ('Forrest 2018',)
    until = 0.9
    dt = 0.01

    def __init__(
        self,
        x: float = 1.0,
        *,
        a: float = 1.0,
        p: float = 2.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.a = a
        self.p = p

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'growth'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'a': self.a,
            'p': self.p,
        }

    def _get_equations(self) -> Equations:
        """Return the rate of the value."""
        return {
            'x': lambda s: s.target.a * s.target.x**s.target.p,
        }


class HyperchaoticRosslerSystem(DynamicalSystem):
    """Rössler's four-dimensional flow.

    The flow is hyperchaotic with two positive Lyapunov exponents for its defaults.

    The flow `x' = -(y + z)`, `y' = x + a y + w`, `z' = b + x z`, `w' = c w - d z` stretches the attractor in two
    directions at once, with the exponents near 0.11 and 0.02 for the parameters of Dilão (2023, Exercise 4.18). The
    system has the states `rossler.x`, `rossler.y`, `rossler.z`, and `rossler.w`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        w:
            The initial fourth coordinate.

        a:
            The feedback of the second coordinate on itself.

        b:
            The constant drive of the third coordinate.

        c:
            The growth of the fourth coordinate.

        d:
            The damping of the fourth coordinate by the third.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HyperchaoticRosslerSystem
        >>> HyperchaoticRosslerSystem(c=0.1).prepare().system_.elements[0].get_value('c')
        0.1
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'ode', 'hyperchaos', 'strange attractor')
    references = ('Rössler 1979', 'Dilão 2023')
    until = 300.0
    dt = 0.05

    def __init__(
        self,
        x: float = -10.0,
        y: float = -6.0,
        z: float = 0.0,
        w: float = 10.0,
        *,
        a: float = 0.25,
        b: float = 3.0,
        c: float = 0.05,
        d: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'rossler'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'w': self.w,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'd': self.d,
        }

    def _get_equations(self) -> Equations:
        """Return the rates of the four coordinates."""
        return {
            'x': lambda s: -(s.target.y + s.target.z),
            'y': lambda s: s.target.x + s.target.a * s.target.y + s.target.w,
            'z': lambda s: s.target.b + s.target.x * s.target.z,
            'w': lambda s: s.target.c * s.target.w - s.target.d * s.target.z,
        }


class NoseHooverSystem(BaseSystem):
    """The Nosé-Hoover oscillator `x' = y`, `y' = y z - x`, `z' = 1 - y²`.

    The oscillator has tori next to a chaotic sea.

    The flow conserves volume on average and its state space is mixed. From `(0, 1.549934227822898, 0)` the orbit is
    periodic, from `(0, 1, 0)` it winds on a torus around that orbit, and from the default `(0, 0.1, 0)` it is chaotic
    (Datseris and Parlitz 2022, Exercise 2.11). Since `z' = 1 - y²`, every bounded orbit has `<y²> = 1`. The system has
    the states `oscillator.x`, `oscillator.y`, and `oscillator.z`.

    Args:
        x:
            The initial position.

        y:
            The initial momentum.

        z:
            The initial friction of the thermostat.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import NoseHooverSystem
        >>> NoseHooverSystem(y=1.0).prepare().system_.elements[0].y
        1.0
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'ode', 'conservative', 'thermostat', 'kam')
    references = ('Hoover 1995', 'Datseris and Parlitz 2022')
    until = 200.0
    dt = 0.05

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.1,
        z: float = 0.0,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        return CompositeSystem(self.label, elements=[NoseHooverElement('oscillator', x=self.x, y=self.y, z=self.z)])


class SprottMixedFlowSystem(BaseSystem):
    """Sprott's flow with a conservative torus and a dissipative chaotic attractor.

    The torus and the attractor lie side by side.

    The flow is `x' = y + 2xy + xz`, `y' = 1 - 2x² + yz`, `z' = x - x² - y²`, whose volumes change at the rate
    `2(y + z)`. From `(1, 0, 0)` the orbit winds on a 2-torus where this rate averages to zero, and from the default
    `(2, 0, 0)` it falls onto a chaotic attractor where it averages below zero (Datseris and Parlitz 2022, Exercises
    3.17 and 8.16).

    The system has the states `flow.x`, `flow.y`, and `flow.z`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SprottMixedFlowSystem
        >>> SprottMixedFlowSystem(1.0).prepare().system_.elements[0].x
        1.0
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'ode', 'conservative', 'strange attractor', 'torus')
    references = ('Sprott 2014', 'Datseris and Parlitz 2022')
    until = 200.0
    dt = 0.05

    def __init__(
        self,
        x: float = 2.0,
        y: float = 0.0,
        z: float = 0.0,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z

    def _build_system(self) -> CompositeSystem:
        """Return Sprott's flow with a conservative torus and a dissipative attractor.

        The torus and the attractor lie side by side.
        """
        return CompositeSystem(self.label, elements=[MixedFlowElement('flow', x=self.x, y=self.y, z=self.z)])


class LangfordSystem(BaseSystem):
    """The Langford flow, a rotation whose torus breaks into a strange attractor.

    The torus breaks for the defaults of Dilão.

    The flow is `x' = (z - b) x - d y`, `y' = d x + (z - b) y`, and
    `z' = c + a z - z³ / 3 - (x² + y²)(1 + e z) + f z x³`, with the reference parameters of Dilão (2023, Exercise
    3.22). Without `f` the flow is symmetric under rotations about the `z` axis.

    The system has the states `langford.x`, `langford.y`, and `langford.z`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial height.

        a:
            The linear growth of the height.

        b:
            The height above which the rotation grows.

        c:
            The constant drive of the height.

        d:
            The angular frequency of the rotation.

        e:
            The coupling of the radius and the height.

        f:
            The term that breaks the rotational symmetry.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LangfordSystem
        >>> LangfordSystem(f=0.0).prepare().system_.elements[0].f
        0.0
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'ode', 'torus', 'strange attractor')
    references = ('Langford 1984', 'Dilão 2023')
    until = 200.0
    dt = 0.05

    def __init__(
        self,
        x: float = 0.1,
        y: float = 0.0,
        z: float = 0.0,
        *,
        a: float = 0.95,
        b: float = 0.7,
        c: float = 0.6,
        d: float = 3.5,
        e: float = 0.25,
        f: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        return CompositeSystem(
            self.label,
            elements=[
                LangfordElement(
                    'langford',
                    x=self.x,
                    y=self.y,
                    z=self.z,
                    a=self.a,
                    b=self.b,
                    c=self.c,
                    d=self.d,
                    e=self.e,
                    f=self.f,
                ),
            ],
        )


class ThomasSystem(DynamicalSystem):
    """The Thomas flow `x' = -a x + sin y` with its cyclic permutations.

    The flow is chaotic for `a = 0.18`.

    The flow is symmetric under the cyclic permutation of its coordinates. As the damping `a` falls the fixed points
    give way to limit cycles and then to chaos, and at `a = 0` the orbits wander through a lattice of cells, Thomas's
    labyrinth chaos (Dilão 2023, Exercise 3.23). The system has the states `thomas.x`, `thomas.y`, and `thomas.z`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        a:
            The damping of every coordinate.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import ThomasSystem
        >>> ThomasSystem(a=0.2).prepare().system_.elements[0].get_value('a')
        0.2
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'ode', 'cyclic symmetry', 'strange attractor')
    references = ('Thomas 1999', 'Dilão 2023')
    until = 500.0
    dt = 0.1

    def __init__(
        self,
        x: float = 0.1,
        y: float = 0.0,
        z: float = 0.0,
        *,
        a: float = 0.18,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.z = z
        self.a = a

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'thomas'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'a': self.a,
        }

    def _get_equations(self) -> Equations:
        """Return the rates of the three coordinates."""
        return {
            'x': lambda s: -s.target.a * s.target.x + sin(s.target.y),
            'y': lambda s: -s.target.a * s.target.y + sin(s.target.z),
            'z': lambda s: -s.target.a * s.target.z + sin(s.target.x),
        }


class IkedaSystem(DynamicalSystem):
    """The Ikeda delay equation `x' = a sin x(t - tau)`.

    The equation is in the form of Dilão (2023, Exercise 6.4).

    The rest points `x = k π` alternate in stability for `a > 0`: those of odd `k` attract while `a tau < π / 2`, and
    lose stability to an oscillation in a Hopf bifurcation at `a tau = π / 2`. The history before the start is the
    initial value. The system has the state `ikeda.x`.

    Args:
        x:
            The initial value, also its history.

        a:
            The gain of the delayed feedback.

        tau:
            The delay.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import IkedaSystem
        >>> IkedaSystem(tau=1.0).prepare().system_.elements[0].get_value('tau')
        1.0
    """

    field = 'mathematics/dynamical_systems'
    topics = ('delay', 'dde', 'hopf bifurcation', 'oscillation')
    references = ('Ikeda and Matsumoto 1987', 'Dilão 2023')
    until = 100.0
    dt = 0.1

    def __init__(
        self,
        x: float = 2.0,
        *,
        a: float = 1.0,
        tau: float = 2.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.a = a
        self.tau = tau

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'ikeda'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'x': DeclaredVariableSpec(self.x, StateSpec()),
            'a': DeclaredVariableSpec(self.a, ParameterSpec()),
            'tau': DeclaredVariableSpec(self.tau, ParameterSpec(domain=RealSet.open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the delayed feedback of the state."""
        return {
            'x': lambda s: s.target.a * sin(delayed(s.target.x, s.target.tau)),
        }
