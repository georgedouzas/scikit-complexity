"""Dynamical systems of flows.

The module covers chaotic and conservative flows, delayed feedback, and singular growth.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from sage.all__sagemath_symbolics import abs_symbolic, sin

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.flows._rossler import RosslerInteraction


class LangfordInteraction(Interaction):
    """The laws of the Langford flow, a rotation in `x, y` with the height `z`.

    Its radius and height couple into a torus that breaks into chaos. The flow is `x' = (z - b) x - d y`,
    `y' = d x + (z - b) y`, and `z' = c + a z - z³ / 3 - (x² + y²)(1 + e z) + f z x³`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose coordinates `x`, `y`, and `z` the flow moves.

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
        >>> from skcomplex.systems.mathematics.dynamical_systems import LangfordInteraction
        >>> LangfordInteraction('flow', target='langford', d=3.0).d
        3.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'langford',
        a: float = 0.95,
        b: float = 0.7,
        c: float = 0.6,
        d: float = 3.5,
        e: float = 0.25,
        f: float = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the three coordinates as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, y, z = context.target.x, context.target.y, context.target.z
            a, b, c, d, e, f = context.a, context.b, context.c, context.d, context.e, context.f
            radius = x**2 + y**2
            self.laws_: list[BaseLaw] = [
                RateLaw(x, (z - b) * x - d * y),
                RateLaw(y, d * x + (z - b) * y),
                RateLaw(z, c + a * z - z**3 / 3 - radius * (1 + e * z) + f * z * x**3),
            ]


class MixedFlowInteraction(Interaction):
    """The laws of Sprott's flow, conservative and dissipative at once.

    The flow is `x' = y + 2xy + xz`, `y' = 1 - 2x² + yz`, `z' = x - x² - y²`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose coordinates `x`, `y`, and `z` the flow moves.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import MixedFlowInteraction
        >>> MixedFlowInteraction('mixed_flow', target='flow').target
        'flow'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role = 'flow') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the three coordinates as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, y, z = context.target.x, context.target.y, context.target.z
            self.laws_: list[BaseLaw] = [
                RateLaw(x, y + 2 * x * y + x * z),
                RateLaw(y, 1 - 2 * x**2 + y * z),
                RateLaw(z, x - x**2 - y**2),
            ]


class NoseHooverInteraction(Interaction):
    """The laws of the Nosé-Hoover oscillator `x' = y`, `y' = y z - x`, `z' = 1 - y²`.

    It is a conservative flow with tori and chaos.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose position `x`, momentum `y`, and friction `z` the flow moves.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import NoseHooverInteraction
        >>> NoseHooverInteraction('thermostat', target='oscillator').target
        'oscillator'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role = 'oscillator') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the position, the momentum, and the friction as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, y, z = context.target.x, context.target.y, context.target.z
            self.laws_: list[BaseLaw] = [RateLaw(x, y), RateLaw(y, y * z - x), RateLaw(z, 1 - y**2)]


class LorenzElement(Element):
    """An element with the three states of the Lorenz flow and its parameters.

    `LorenzInteraction` gives the flow `x' = sigma (y - x)`, `y' = x (rho - z) - y`, `z' = x y - beta z`.

    Args:
        label:
            The label of the element.

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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LorenzElement
        >>> LorenzElement('lorenz', rho=20.0).rho
        20.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: float = 1.0,
        y: float = 1.0,
        z: float = 1.0,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z
        self.sigma = sigma
        self.rho = rho
        self.beta = beta


class LorenzInteraction(Interaction):
    """The Lorenz flow of the three states of the element it binds.

    The flow is `x' = sigma (y - x)`, `y' = x (rho - z) - y`, `z' = x y - beta z`, with the parameters of the element.
    For `sigma = 10`, `rho = 28`, and `beta = 8/3` its runs never settle and never repeat, the strange attractor of
    Lorenz. Lorenz derived it from convection in the atmosphere, it models a laser and a water wheel, and it drives a
    response system in studies of chaotic synchronization.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element with the states `x`, `y`, and `z` and the parameters `sigma`, `rho`, and `beta`.

    Examples:
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.mathematics.dynamical_systems import LorenzElement, LorenzInteraction
        >>> lorenz = LorenzElement('lorenz', x=0.0, y=0.0, z=2.0, beta=0.5)
        >>> system = System(elements=[lorenz], interactions=[LorenzInteraction('flow', target='lorenz')]).prepare()
        >>> [round(float(value), 4) for value in system.simulate(until=1.0, dt=0.5).get_values('lorenz.z')]
        [2.0, 1.5576, 1.2131]
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role = 'lorenz') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the three modes as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            self.laws_: list[BaseLaw] = [
                RateLaw(s.x, s.sigma * (s.y - s.x)),
                RateLaw(s.y, s.x * (s.rho - s.z) - s.y),
                RateLaw(s.z, s.x * s.y - s.beta * s.z),
            ]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        lorenz = LorenzElement('lorenz', x=self.x, y=self.y, z=self.z, sigma=self.sigma, rho=self.rho, beta=self.beta)
        self.system_ = System(self.label, elements=[lorenz], interactions=[LorenzInteraction('flow')])


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import RosslerSystem
        >>> RosslerSystem(c=4.0).prepare().system_.interactions[0].c
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.c = c

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('rossler', values={'x': self.x, 'y': self.y, 'z': self.z})],
            interactions=[RosslerInteraction('flow', target='rossler', a=self.a, b=self.b, c=self.c)],
        )


class SprottJerkInteraction(Interaction):
    """The law of Sprott's jerk equation `x''' = -A x'' - x' + |x| - 1`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose position `x` the equation moves.

        A:
            The damping of the second derivative.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SprottJerkInteraction
        >>> SprottJerkInteraction('jerk_law', target='jerk', A=0.5).A
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role = 'jerk', A: float = 0.6) -> None:
        super().__init__(label)
        self.target = target
        self.A = A

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the third derivative of the position as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            jerk = -context.A * x.diff(t, 2) - x.diff(t) + abs_symbolic(x) - 1
            self.laws_: list[BaseLaw] = [RateLaw(x.diff(t, 3), jerk)]


class SprottJerkSystem(BaseSystem):
    """Sprott's jerk equation `x''' = -A x'' - x' + |x| - 1`, chaotic near A = 0.6.

    The system has the state `jerk.x` of order 3 and its derivatives `jerk.x'` and `jerk.x''`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SprottJerkSystem
        >>> SprottJerkSystem(A=0.5).prepare().system_.interactions[0].A
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.velocity = velocity
        self.acceleration = acceleration
        self.A = A

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the jerk equation."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('jerk', values={'x': self.x, "x'": self.velocity, "x''": self.acceleration})],
            interactions=[SprottJerkInteraction('jerk_law', target='jerk', A=self.A)],
        )


class FiniteTimeBlowUpInteraction(Interaction):
    """The laws of hyperbolic growth `x' = a x^p`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose state `x` the flow moves.

        a:
            The growth coefficient.

        p:
            The growth exponent.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import FiniteTimeBlowUpInteraction
        >>> FiniteTimeBlowUpInteraction('finite_time_blow_up', target='growth', a=0.5).a
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'growth',
        a: float = 1.0,
        p: float = 2.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.p = p

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of every state as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            a, p = context.a, context.p
            self.laws_: list[BaseLaw] = [
                RateLaw(x, a * x**p),
            ]


class FiniteTimeBlowUpSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import FiniteTimeBlowUpSystem
        >>> FiniteTimeBlowUpSystem(p=3.0).prepare().system_.interactions[0].p
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.a = a
        self.p = p

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('growth', values={'x': self.x})],
            interactions=[FiniteTimeBlowUpInteraction('finite_time_blow_up', target='growth', a=self.a, p=self.p)],
        )


class HyperchaoticRosslerInteraction(Interaction):
    """The laws of rössler's four-dimensional flow.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x`, `y`, `z`, and `w` the flow moves.

        a:
            The feedback of the second coordinate on itself.

        b:
            The constant drive of the third coordinate.

        c:
            The growth of the fourth coordinate.

        d:
            The damping of the fourth coordinate by the third.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HyperchaoticRosslerInteraction
        >>> HyperchaoticRosslerInteraction('hyperchaotic_rossler', target='rossler', a=0.5).a
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'rossler',
        a: float = 0.25,
        b: float = 3.0,
        c: float = 0.05,
        d: float = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of every state as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, y, z, w = context.target.x, context.target.y, context.target.z, context.target.w
            a, b, c, d = context.a, context.b, context.c, context.d
            self.laws_: list[BaseLaw] = [
                RateLaw(x, -(y + z)),
                RateLaw(y, x + a * y + w),
                RateLaw(z, b + x * z),
                RateLaw(w, c * w - d * z),
            ]


class HyperchaoticRosslerSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HyperchaoticRosslerSystem
        >>> HyperchaoticRosslerSystem(c=0.1).prepare().system_.interactions[0].c
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('rossler', values={'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w})],
            interactions=[
                HyperchaoticRosslerInteraction(
                    'hyperchaotic_rossler',
                    target='rossler',
                    a=self.a,
                    b=self.b,
                    c=self.c,
                    d=self.d,
                ),
            ],
        )


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import NoseHooverSystem
        >>> NoseHooverSystem(y=1.0).prepare().system_.elements[0].values['y']
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the oscillator and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('oscillator', values={'x': self.x, 'y': self.y, 'z': self.z})],
            interactions=[NoseHooverInteraction('thermostat', target='oscillator')],
        )


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SprottMixedFlowSystem
        >>> SprottMixedFlowSystem(1.0).prepare().system_.elements[0].values['x']
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('flow', values={'x': self.x, 'y': self.y, 'z': self.z})],
            interactions=[MixedFlowInteraction('mixed_flow', target='flow')],
        )


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LangfordSystem
        >>> LangfordSystem(f=0.0).prepare().system_.interactions[0].f
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('langford', values={'x': self.x, 'y': self.y, 'z': self.z})],
            interactions=[
                LangfordInteraction(
                    'flow',
                    target='langford',
                    a=self.a,
                    b=self.b,
                    c=self.c,
                    d=self.d,
                    e=self.e,
                    f=self.f,
                ),
            ],
        )


class ThomasInteraction(Interaction):
    """The laws of the Thomas flow `x' = -a x + sin y` with its cyclic permutations.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x`, `y`, and `z` the flow moves.

        a:
            The damping of every coordinate.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import ThomasInteraction
        >>> ThomasInteraction('flow', target='thomas', a=0.5).a
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'thomas',
        a: float = 0.18,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of every state as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, y, z = context.target.x, context.target.y, context.target.z
            a = context.a
            self.laws_: list[BaseLaw] = [
                RateLaw(x, -a * x + sin(y)),
                RateLaw(y, -a * y + sin(z)),
                RateLaw(z, -a * z + sin(x)),
            ]


class ThomasSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import ThomasSystem
        >>> ThomasSystem(a=0.2).prepare().system_.interactions[0].a
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z
        self.a = a

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('thomas', values={'x': self.x, 'y': self.y, 'z': self.z})],
            interactions=[ThomasInteraction('flow', target='thomas', a=self.a)],
        )


class IkedaInteraction(Interaction):
    """The law of the Ikeda delay equation `x' = a sin x(t - tau)`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose value `x` the delayed feedback moves.

        a:
            The gain of the delayed feedback.

        tau:
            The delay, a positive time.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import IkedaInteraction
        >>> IkedaInteraction('feedback', target='ikeda', tau=1.0).tau
        1.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'ikeda',
        a: float = 1.0,
        tau: float = 2.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.tau = tau

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the delayed feedback of the value as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_: list[BaseLaw] = [RateLaw(x, context.a * sin(x(t=t - context.tau)))]


class IkedaSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import IkedaSystem
        >>> IkedaSystem(tau=1.0).prepare().system_.interactions[0].tau
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.a = a
        self.tau = tau

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the delayed feedback."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('ikeda', values={'x': self.x})],
            interactions=[IkedaInteraction('feedback', target='ikeda', a=self.a, tau=self.tau)],
        )
