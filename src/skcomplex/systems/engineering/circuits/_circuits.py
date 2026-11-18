"""Electric circuits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import Expression, cos, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, ConstraintLaw, RateLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, ScipyODESolver
from ....core.errors import ParameterError
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
from ..common.signals import PulseInteraction

CHUA_NONLINEARITIES = ('piecewise-linear', 'sine')


class KirchhoffVoltageInteraction(Interaction):
    """Kirchhoff's voltage law of a series RLC loop driven by a sinusoidal source.

    The charge `q` of the loop follows `L q'' + R q' + q / C = V cos(omega t)`, with the inductance `L`, the resistance
    `R`, and the capacitance `C` of the loop. The same equation rings down a loop without a source, `V = 0`, and holds
    for a mass on a damped spring, with the mass, the damping, and the compliance in place of `L`, `R`, and `C`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the loop, which holds the charge and the values `L`, `R`, and `C`.

        state:
            The name of the charge of the loop.

        V:
            The amplitude of the source voltage.

        omega:
            The angular frequency of the source.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.engineering.circuits import KirchhoffVoltageInteraction
        >>> loop = Element('loop', values={'q': 1.0, 'L': 1.0, 'R': 0.1, 'C': 1.0})
        >>> System(elements=[loop], interactions=[KirchhoffVoltageInteraction('kvl', target='loop')]).prepare().states_
        ('loop.q', "loop.q'")
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (ConstraintLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'q',
        V: float = 0.0,
        omega: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.V = V
        self.omega = omega

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the voltage law of the loop."""
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        loop, t = context.target, context.t
        q = getattr(loop, self.state)
        law = ConstraintLaw(
            loop.L * q.diff(t, 2) + loop.R * q.diff(t) + q / loop.C == context.V * cos(context.omega * t),
        )
        self.laws_: list[BaseComponent] = [law]


class ChuaCircuitElement(Element):
    """Chua circuit, with the voltages `x` and `y` of its capacitors and the current `z` of its inductor.

    The interactions of `ChuaSystem` and `MixedModeChuaSystem` change the three states.

    Args:
        label:
            The label of the circuit.

        x:
            The initial voltage of the first capacitor.

        y:
            The initial voltage of the second capacitor.

        z:
            The initial current of the inductor.

    Examples:
        >>> from skcomplex.systems.engineering.circuits._circuits import ChuaCircuitElement
        >>> ChuaCircuitElement('circuit', x=1.0).x
        1.0
    """

    def __init__(self, label: str, *, x: float = 0.1, y: float = 0.0, z: float = 0.0) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z


class ChuaInteraction(Interaction):
    """Chua circuit with the piecewise-linear current of its diode, which gives the double scroll.

    The states follow `x' = alpha (y - x - m1 x - (m0 - m1) (|x + 1| - |x - 1|) / 2)`, `y' = x - y + z`, and
    `z' = -beta y`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the circuit.

        alpha:
            The ratio of the capacitances.

        beta:
            The ratio of the inductance to the second capacitance.

        m0:
            The inner slope of the diode.

        m1:
            The outer slope of the diode.

    Examples:
        >>> from skcomplex.systems.engineering.circuits._circuits import ChuaInteraction
        >>> ChuaInteraction('chua', target='circuit').alpha
        15.6
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        alpha: Annotated[float, RealSet.open(0, oo)] = 15.6,
        beta: Annotated[float, RealSet.open(0, oo)] = 28.0,
        m0: float = -8 / 7,
        m1: float = -5 / 7,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.m0 = m0
        self.m1 = m1

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the states of the circuit.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the circuit.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        s, x, y, z = context, context.target.x, context.target.y, context.target.z
        self.laws_: list[BaseComponent] = [
            RateLaw(x.diff(s.t), s.alpha * (y - x - s.m1 * x - (s.m0 - s.m1) * (abs(x + 1) - abs(x - 1)) / 2)),
            RateLaw(y.diff(s.t), x - y + z),
            RateLaw(z.diff(s.t), -s.beta * y),
        ]


class MultiscrollChuaInteraction(Interaction):
    """Chua circuit with a sine diode, which gives attractors of many scrolls.

    The states follow `x' = alpha (y - f(x))`, `y' = x - y + z`, and `z' = -beta y`, with
    `f(x) = -b sin(pi x / (2 a))`, continued linearly beyond `|x| = 2 a c`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the circuit.

        alpha:
            The ratio of the capacitances.

        beta:
            The ratio of the inductance to the second capacitance.

        a:
            The half period of the sine in `x`.

        b:
            The amplitude of the sine.

        c:
            The number of half periods on each side, beyond which the diode is linear.

    Examples:
        >>> from skcomplex.systems.engineering.circuits._circuits import MultiscrollChuaInteraction
        >>> MultiscrollChuaInteraction('chua', target='circuit').alpha
        10.82
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        alpha: Annotated[float, RealSet.open(0, oo)] = 10.82,
        beta: Annotated[float, RealSet.open(0, oo)] = 14.286,
        a: Annotated[float, RealSet.open(0, oo)] = 1.3,
        b: Annotated[float, RealSet.open(0, oo)] = 0.11,
        c: Annotated[float, RealSet.open(0, oo)] = 3.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.c = c

    @staticmethod
    def _scroll_nonlinearity(x: Expression, a: Expression, b: Expression, c: Expression) -> Expression:
        """Return the sine characteristic of the multiscroll Chua circuit.

        It is continued linearly beyond `|x| = 2 a c`.
        """
        edge = 2 * a * c
        slope = b * pi / (2 * a)
        return where(x >= edge, slope * (x - edge), where(x <= -edge, slope * (x + edge), -b * sin(pi * x / (2 * a))))

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the states of the circuit.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the circuit.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        s, x, y, z = context, context.target.x, context.target.y, context.target.z
        self.laws_: list[BaseComponent] = [
            RateLaw(x.diff(s.t), s.alpha * (y - self._scroll_nonlinearity(x, s.a, s.b, s.c))),
            RateLaw(y.diff(s.t), x - y + z),
            RateLaw(z.diff(s.t), -s.beta * y),
        ]


class MixedModeChuaInteraction(Interaction):
    """Slow-fast Chua circuit whose oscillations mix small and large amplitudes.

    The states follow `epsilon x' = -y + (a + b x) x^2`, `y' = alpha (x - R y - z)`, and `z' = c - beta y`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the circuit.

        epsilon:
            The first capacitance, which sets the fast time scale.

        a:
            The quadratic coefficient of the diode.

        b:
            The cubic coefficient of the diode.

        alpha:
            The inverse inductance.

        R:
            The resistance.

        beta:
            The gain of the controlled source over the second capacitance.

        c:
            The offset of the controlled source.

    Examples:
        >>> from skcomplex.systems.engineering.circuits._circuits import MixedModeChuaInteraction
        >>> MixedModeChuaInteraction('chua', target='circuit').c
        0.00053
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        epsilon: Annotated[float, RealSet.open(0, oo)] = 0.01,
        a: float = 1.5,
        b: float = -1.0,
        alpha: Annotated[float, RealSet.open(0, oo)] = 1.0,
        R: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        beta: Annotated[float, RealSet.open(0, oo)] = 0.005,
        c: float = 0.00053,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.alpha = alpha
        self.R = R
        self.beta = beta
        self.c = c

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the states of the circuit.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the circuit.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        s, x, y, z = context, context.target.x, context.target.y, context.target.z
        self.laws_: list[BaseComponent] = [
            RateLaw(x.diff(s.t), (-y + (s.a + s.b * x) * x**2) / s.epsilon),
            RateLaw(y.diff(s.t), s.alpha * (x - s.R * y - z)),
            RateLaw(z.diff(s.t), s.c - s.beta * y),
        ]


class RlcCircuitSystem(BaseSystem):
    """A series RLC circuit written as Kirchhoff's voltage law.

    The law is `L q'' + R q' + q / C = V cos(ω t)`. The system has the charge `circuit.q` and the current
    `circuit.q'`.

    Args:
        q:
            The initial charge of the capacitor.

        current:
            The initial current.

        L:
            The inductance.

        R:
            The resistance.

        C:
            The capacitance.

        V:
            The amplitude of the source voltage.

        omega:
            The angular frequency of the source.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import RlcCircuitSystem
        >>> RlcCircuitSystem(R=0.5).prepare().states_
        ('circuit.q', "circuit.q'")
    """

    field = 'engineering/circuits'
    topics = ('electromagnetism', 'circuit', 'ode', 'second order')
    references = ('Kirchhoff 1845',)
    until = 50.0
    dt = 0.05

    def __init__(
        self,
        q: float = 1.0,
        current: float = 0.0,
        *,
        L: float = 1.0,
        R: float = 0.1,
        C: float = 1.0,
        V: float = 0.0,
        omega: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.q = q
        self.current = current
        self.L = L
        self.R = R
        self.C = C
        self.V = V
        self.omega = omega

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the loop and its voltage law."""
        super()._prepare(context)
        values = {'q': self.q, "q'": self.current, 'L': self.L, 'R': self.R, 'C': self.C}
        self.system_ = System(
            self.label,
            elements=[Element('circuit', values=values)],
            interactions=[KirchhoffVoltageInteraction('kvl', target='circuit', V=self.V, omega=self.omega)],
        )


class ChuaSystem(BaseSystem):
    """Chua's circuit, whose double-scroll or multiscroll attractor is chaotic.

    The voltages of the two capacitors and the current of the inductor, in dimensionless form, follow `x' = alpha (y -
    f(x))`, `y' = x - y + z`, and `z' = -β y`. The piecewise-linear diode has `f(x) = x + m1 x + (m0 - m1) (|x + 1| - |x
    - 1|) / 2`, which gives the double scroll. The sine diode has `f(x) = -b sin(π x / (2 a))`, continued linearly
    beyond `|x| = 2 a c`, which gives several scrolls side by side. The system has the states `circuit.x`, `circuit.y`,
    and `circuit.z`.

    Args:
        x:
            The initial voltage of the first capacitor, or `None` for 0.1 with the piecewise-linear diode and 1 with
            the sine diode.

        y:
            The initial voltage of the second capacitor, or `None` for 0 with the piecewise-linear diode and 1 with the
            sine diode.

        z:
            The initial current of the inductor.

        nonlinearity:
            The characteristic of the diode, `'piecewise-linear'` for the double scroll or `'sine'` for multiple
            scrolls.

        alpha:
            The ratio of the capacitances, or `None` for 15.6 with the piecewise-linear diode and 10.82 with the sine
            diode.

        beta:
            The ratio of the inductance to the second capacitance, or `None` for 28 with the piecewise-linear diode and
            14.286 with the sine diode.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The nonlinearity is neither `'piecewise-linear'` nor `'sine'`.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import ChuaSystem
        >>> ChuaSystem(nonlinearity='sine').prepare().system_.interactions[0].alpha
        10.82
    """

    field = 'engineering/circuits'
    topics = ('circuit', 'chaos', 'ode', 'attractor')
    references = ('Chua, Komuro, and Matsumoto 1986', 'Tang et al. 2001', 'Trzaska 2023')
    until = 100.0
    dt = 0.05

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float = 0.0,
        *,
        nonlinearity: str = 'piecewise-linear',
        alpha: float | None = None,
        beta: float | None = None,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z
        self.nonlinearity = nonlinearity
        self.alpha = alpha
        self.beta = beta

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        if self.nonlinearity not in CHUA_NONLINEARITIES:
            error_msg = f"Unknown nonlinearity {self.nonlinearity!r}. Use 'piecewise-linear' or 'sine'."
            raise ParameterError(error_msg)
        sine = self.nonlinearity == 'sine'
        start = (1.0, 1.0) if sine else (0.1, 0.0)
        x = start[0] if self.x is None else self.x
        y = start[1] if self.y is None else self.y
        circuit = ChuaCircuitElement('circuit', x=x, y=y, z=self.z)
        kind = MultiscrollChuaInteraction if sine else ChuaInteraction
        settings = {
            name: value for name, value in {'alpha': self.alpha, 'beta': self.beta}.items() if value is not None
        }
        chua = kind('chua', target='circuit').set_params(**settings)
        self.system_ = System(self.label, elements=[circuit], interactions=[chua])


class MixedModeChuaSystem(BaseSystem):
    """A modified Chua circuit on three time scales.

    Its oscillations mix small and large amplitudes. The circuit follows `ε x' = -y + (a + b x) x^2`,
    `y' = alpha (x - R y - z)`, and `z' = c - β y`. For `ε = 0.01`, `a = 1.5`, `b = -1`, and `β = 0.005`, the circuit
    has small oscillations only for `c = 0`, large ones only for `c = 0.00105`, and mixed-mode oscillations for
    `c = 0.00053`. The system has the states `circuit.x`, `circuit.y`, and `circuit.z`.

    Args:
        x:
            The initial voltage of the first capacitor.

        y:
            The initial current of the inductor.

        z:
            The initial voltage of the second capacitor.

        epsilon:
            The first capacitance, which sets the fast time scale.

        a:
            The quadratic coefficient of the diode.

        b:
            The cubic coefficient of the diode.

        alpha:
            The inverse inductance.

        R:
            The resistance.

        beta:
            The gain of the controlled source over the second capacitance.

        c:
            The offset of the controlled source.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import MixedModeChuaSystem
        >>> MixedModeChuaSystem(c=0.0).prepare().system_.interactions[0].c
        0.0
    """

    field = 'engineering/circuits'
    topics = ('circuit', 'mixed-mode oscillations', 'canards', 'slow-fast', 'ode')
    references = ('Krupa, Popović, and Kopell 2008', 'Trzaska 2023')
    until = 200.0
    dt = 0.5

    def __init__(
        self,
        x: float = 0.01,
        y: float = 0.0,
        z: float = 0.0,
        *,
        epsilon: float = 0.01,
        a: float = 1.5,
        b: float = -1.0,
        alpha: float = 1.0,
        R: float = 0.0,
        beta: float = 0.005,
        c: float = 0.00053,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.alpha = alpha
        self.R = R
        self.beta = beta
        self.c = c

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        circuit = ChuaCircuitElement('circuit', x=self.x, y=self.y, z=self.z)
        chua = MixedModeChuaInteraction(
            'chua',
            target='circuit',
            epsilon=self.epsilon,
            a=self.a,
            b=self.b,
            alpha=self.alpha,
            R=self.R,
            beta=self.beta,
            c=self.c,
        )
        self.system_ = System(
            self.label,
            elements=[circuit],
            interactions=[chua],
            solver=ScipyODESolver(method='Radau'),
        )


class MemristorOscillatorSystem(DynamicalSystem):
    """The simplest memristor oscillator.

    It is a capacitor, an inductor, and a memristor in series. The capacitor voltage, the inductor current, and the
    internal state of the memristor follow `C v' = i`, `L i' = -v - β (w^2 - 1) i`, and `w' = -i - alpha w + w i`. The
    memristor has the memristance `β (w^2 - 1)`, and the circuit oscillates chaotically for the default values. The
    system has the states `circuit.v`, `circuit.i`, and `circuit.w`.

    Args:
        v:
            The initial voltage of the capacitor.

        i:
            The initial current.

        w:
            The initial internal state of the memristor.

        C:
            The capacitance.

        L:
            The inductance.

        alpha:
            The decay rate of the memristor state.

        beta:
            The scale of the memristance.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import MemristorOscillatorSystem
        >>> MemristorOscillatorSystem(L=2.0).prepare().system_.elements[0].values['L']
        2.0
    """

    field = 'engineering/circuits'
    topics = ('circuit', 'memristor', 'chaos', 'ode')
    references = ('Muthuswamy and Chua 2010', 'Trzaska 2023')
    until = 200.0
    dt = 0.1

    def __init__(
        self,
        v: float = 0.1,
        i: float = 0.0,
        w: float = 0.1,
        *,
        C: float = 1.0,
        L: float = 3.0,
        alpha: float = 0.6,
        beta: float = 1.5,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.v = v
        self.i = i
        self.w = w
        self.C = C
        self.L = L
        self.alpha = alpha
        self.beta = beta

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'circuit'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'v': self.v,
            'i': self.i,
            'w': self.w,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'C': self.C,
            'L': self.L,
            'alpha': self.alpha,
            'beta': self.beta,
        }

    def _get_equations(self) -> Equations:
        """Return the rates of the states of the circuit."""
        return {
            'v': lambda s: s.target.i / s.target.C,
            'i': lambda s: -(s.target.v + s.target.beta * (s.target.w**2 - 1) * s.target.i) / s.target.L,
            'w': lambda s: -s.target.i - s.target.alpha * s.target.w + s.target.w * s.target.i,
        }


class TunnelDiodeOscillatorSystem(DynamicalSystem):
    """A tunnel-diode oscillator.

    It is an inductor and a capacitor in parallel with a negative-resistance diode. The diode current is
    `h(V) = gamma V^3 - alpha V`, so Kirchhoff's laws give `C V' = -h(V) - i` and `L i' = V`, which is the van der Pol
    oscillator with `ε = alpha sqrt(L / C)`. The voltage settles on a limit cycle of amplitude near
    `2 sqrt(alpha / (3 gamma))`. The system has the voltage `circuit.V` and the inductor current `circuit.i`.

    Args:
        V:
            The initial voltage.

        i:
            The initial current of the inductor.

        L:
            The inductance.

        C:
            The capacitance.

        alpha:
            The negative conductance of the diode at zero voltage.

        gamma:
            The cubic coefficient of the diode.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import TunnelDiodeOscillatorSystem
        >>> TunnelDiodeOscillatorSystem(alpha=2.0).prepare().system_.elements[0].values['alpha']
        2.0
    """

    field = 'engineering/circuits'
    topics = ('circuit', 'oscillation', 'limit cycle', 'ode')
    references = ('van der Pol 1926', 'Trzaska 2023')
    until = 50.0
    dt = 0.05

    def __init__(
        self,
        V: float = 0.1,
        i: float = 0.0,
        *,
        L: float = 1.0,
        C: float = 1.0,
        alpha: float = 1.0,
        gamma: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.V = V
        self.i = i
        self.L = L
        self.C = C
        self.alpha = alpha
        self.gamma = gamma

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'circuit'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'V': self.V,
            'i': self.i,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'L': self.L,
            'C': self.C,
            'alpha': self.alpha,
            'gamma': self.gamma,
        }

    def _get_equations(self) -> Equations:
        """Return the rates of the states of the circuit."""
        return {
            'V': lambda s: -(s.target.gamma * s.target.V**3 - s.target.alpha * s.target.V + s.target.i) / s.target.C,
            'i': lambda s: s.target.V / s.target.L,
        }


class ImpulsiveCircuitsSystem(BaseSystem):
    """An RLC circuit driven by a periodic train of current impulses.

    It follows `L x'' + R x' + x / C = (A / C) Σ δ(t - k T)`. Every impulse is an event that raises the rate of
    change `x'` by `A / (L C)`, and between impulses the circuit rings down freely. The response settles on a periodic,
    non-sinusoidal waveform of period `T`. The system has the response `circuit.x` and its rate `circuit.x'`, kicked by
    the impulses `impulse`.

    Args:
        x:
            The initial value of the response, the current of the coil.

        current:
            The initial rate of change of the response.

        L:
            The inductance.

        R:
            The resistance.

        C:
            The capacitance.

        A:
            The strength of every impulse.

        T:
            The period of the impulses.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import ImpulsiveCircuitsSystem
        >>> ImpulsiveCircuitsSystem(T=3.0).prepare().system_.interactions[0].period
        3.0
    """

    field = 'engineering/circuits'
    topics = ('circuit', 'impulses', 'events', 'hybrid')
    references = ('Trzaska 2023',)
    until = 40.0
    dt = 0.05

    def __init__(
        self,
        x: float = 0.0,
        current: float = 0.0,
        *,
        L: float = 1.0,
        R: float = 0.1,
        C: float = 1.0,
        A: float = 1.0,
        T: float = 2.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.current = current
        self.L = L
        self.R = R
        self.C = C
        self.A = A
        self.T = T

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the circuit, the impulses, and the voltage law of the free circuit."""
        super()._prepare(context)
        values = {'x': self.x, "x'": self.current, 'L': self.L, 'R': self.R, 'C': self.C}
        kick = {"x'": lambda s: s.target.x.diff(s.t) + self.A / (s.target.L * s.target.C)}
        self.system_ = System(
            self.label,
            elements=[Element('circuit', values=values)],
            interactions=[
                PulseInteraction('impulse', target='circuit', reset=kick, period=self.T),
                KirchhoffVoltageInteraction('free', target='circuit', state='x'),
            ],
        )
