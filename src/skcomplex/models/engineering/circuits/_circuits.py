"""Electric circuits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, Any

from sage.all__sagemath_symbolics import Expression, cos, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.errors import ParameterError
from ....core.laws import BaseLaw, CrossingLaw, EquationLaw, RateLaw
from ....core.laws.utils import derivative, where
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.solvers import ScipyODESolver
from ..._catalog import register_model

CHUA_NONLINEARITIES = ('piecewise-linear', 'sine')


class CircuitElement(Element):
    """Circuit element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        q: Annotated[object, StateSpec(order=2)] = 1.0,
        q_d1: float = 0.0,
        L: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        R: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        V: Annotated[object, ParameterSpec()] = 0.0,
        omega: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.q = q
        self.q_d1 = q_d1
        self.L = L
        self.R = R
        self.C = C
        self.V = V
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the circuit element."""
        return [
            EquationLaw(s.L * derivative(s.q, 2) + s.R * derivative(s.q, 1) + s.q / s.C == s.V * cos(s.omega * s.t)),
        ]


@register_model(
    field='engineering/circuits',
    tags=('electromagnetism', 'circuit', 'ode', 'second order'),
    references=('Kirchhoff 1845',),
    defaults={'until': 50.0, 'dt': 0.05},
)
def rlc_circuit(
    q: float = 1.0,
    current: float = 0.0,
    *,
    L: float = 1.0,
    R: float = 0.1,
    C: float = 1.0,
    V: float = 0.0,
    omega: float = 1.0,
) -> System:
    """Return a series RLC circuit written as Kirchhoff's voltage law, `L q'' + R q' + q / C = V cos(ω t)`.

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

    Returns:
        system:
            The model, with the circuit `circuit`, whose charge has order 2 and current `circuit.q_d1`.

    Examples:
        >>> from skcomplex.models.engineering.circuits import rlc_circuit
        >>> rlc_circuit(R=0.5).elements[0].R
        0.5
    """
    return System(elements=[CircuitElement('circuit', q=q, q_d1=current, L=L, R=R, C=C, V=V, omega=omega)])


def _scroll_nonlinearity(x: Expression, a: Expression, b: Expression, c: Expression) -> Expression:
    """Return the sine characteristic of the multiscroll Chua circuit, continued linearly beyond `|x| = 2 a c`."""
    edge = 2 * a * c
    slope = b * pi / (2 * a)
    return where(x >= edge, slope * (x - edge), where(x <= -edge, slope * (x + edge), -b * sin(pi * x / (2 * a))))


class ChuaCircuitElement(Element):
    """Chua circuit element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.1,
        y: Annotated[object, StateSpec()] = 0.0,
        z: Annotated[object, StateSpec()] = 0.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 15.6,
        beta: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 28.0,
        m0: Annotated[object, ParameterSpec()] = -8 / 7,
        m1: Annotated[object, ParameterSpec()] = -5 / 7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.alpha = alpha
        self.beta = beta
        self.m0 = m0
        self.m1 = m1

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the chua circuit element."""
        return [
            RateLaw(s.x, s.alpha * (s.y - s.x - s.m1 * s.x - (s.m0 - s.m1) * (abs(s.x + 1) - abs(s.x - 1)) / 2)),
            RateLaw(s.y, s.x - s.y + s.z),
            RateLaw(s.z, -s.beta * s.y),
        ]


class MultiscrollChuaCircuitElement(Element):
    """Multiscroll chua circuit element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        y: Annotated[object, StateSpec()] = 1.0,
        z: Annotated[object, StateSpec()] = 0.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.82,
        beta: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 14.286,
        a: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.3,
        b: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.11,
        c: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.c = c

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the multiscroll chua circuit element."""
        return [
            RateLaw(s.x, s.alpha * (s.y - _scroll_nonlinearity(s.x, s.a, s.b, s.c))),
            RateLaw(s.y, s.x - s.y + s.z),
            RateLaw(s.z, -s.beta * s.y),
        ]


class MixedModeChuaCircuitElement(Element):
    """Mixed mode chua circuit element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.01,
        y: Annotated[object, StateSpec()] = 0.0,
        z: Annotated[object, StateSpec()] = 0.0,
        epsilon: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        a: Annotated[object, ParameterSpec()] = 1.5,
        b: Annotated[object, ParameterSpec()] = -1.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        R: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        beta: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.005,
        c: Annotated[object, ParameterSpec()] = 0.00053,
    ) -> None:
        super().__init__(label, visual=visual)
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the mixed mode chua circuit element."""
        return [
            RateLaw(s.x, (-s.y + (s.a + s.b * s.x) * s.x**2) / s.epsilon),
            RateLaw(s.y, s.alpha * (s.x - s.R * s.y - s.z)),
            RateLaw(s.z, s.c - s.beta * s.y),
        ]


class MemristorCircuitElement(Element):
    """Memristor circuit element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[object, StateSpec()] = 0.1,
        i: Annotated[object, StateSpec()] = 0.0,
        w: Annotated[object, StateSpec()] = 0.1,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        L: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        beta: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.i = i
        self.w = w
        self.C = C
        self.L = L
        self.alpha = alpha
        self.beta = beta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the memristor circuit element."""
        return [
            RateLaw(s.v, s.i / s.C),
            RateLaw(s.i, -(s.v + s.beta * (s.w**2 - 1) * s.i) / s.L),
            RateLaw(s.w, -s.i - s.alpha * s.w + s.w * s.i),
        ]


class TunnelDiodeCircuitElement(Element):
    """Tunnel diode circuit element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        V: Annotated[object, StateSpec()] = 0.1,
        i: Annotated[object, StateSpec()] = 0.0,
        L: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        gamma: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.V = V
        self.i = i
        self.L = L
        self.C = C
        self.alpha = alpha
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the tunnel diode circuit element."""
        return [
            RateLaw(s.V, -(s.gamma * s.V**3 - s.alpha * s.V + s.i) / s.C),
            RateLaw(s.i, s.V / s.L),
        ]


class KickedCircuitElement(Element):
    """Kicked circuit element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        clock: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        L: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        R: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        A: Annotated[object, ParameterSpec()] = 1.0,
        T: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.clock = clock
        self.L = L
        self.R = R
        self.C = C
        self.A = A
        self.T = T

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the kicked circuit element."""
        return [
            EquationLaw(s.L * s.C * derivative(s.x, 2) + s.R * s.C * derivative(s.x, 1) + s.x == 0),
            RateLaw(s.clock, 1),
            CrossingLaw(
                s.clock - s.T,
                {s.clock: 0, derivative(s.x, 1): derivative(s.x, 1) + s.A / (s.L * s.C)},
                direction=1,
                label='impulse',
            ),
        ]


@register_model(
    field='engineering/circuits',
    tags=('circuit', 'chaos', 'ode', 'attractor'),
    references=('Chua, Komuro, and Matsumoto 1986', 'Tang et al. 2001', 'Trzaska 2023'),
    defaults={'until': 100.0, 'dt': 0.05},
)
def chua(
    x: float | None = None,
    y: float | None = None,
    z: float = 0.0,
    *,
    nonlinearity: str = 'piecewise-linear',
    alpha: float | None = None,
    beta: float | None = None,
) -> System:
    """Return Chua's circuit, whose double-scroll or multiscroll attractor is chaotic.

    The voltages of the two capacitors and the current of the inductor, in dimensionless form, follow
    `x' = alpha (y - f(x))`, `y' = x - y + z`, and `z' = -β y`. The piecewise-linear diode has
    `f(x) = x + m1 x + (m0 - m1) (|x + 1| - |x - 1|) / 2`, which gives the double scroll. The sine diode has
    `f(x) = -b sin(π x / (2 a))`, continued linearly beyond `|x| = 2 a c`, which gives several scrolls side by side.

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

    Returns:
        system:
            The model, with the states `circuit.x`, `circuit.y`, and `circuit.z`.

    Raises:
        ParameterError:
            The nonlinearity is neither `'piecewise-linear'` nor `'sine'`.

    Examples:
        >>> from skcomplex.models.engineering.circuits import chua
        >>> chua(nonlinearity='sine').elements[0].alpha
        10.82
    """
    if nonlinearity not in CHUA_NONLINEARITIES:
        error_msg = f"Unknown nonlinearity {nonlinearity!r}. Use 'piecewise-linear' or 'sine'."
        raise ParameterError(error_msg)
    element = ChuaCircuitElement if nonlinearity == 'piecewise-linear' else MultiscrollChuaCircuitElement
    initial = {'x': x, 'y': y, 'alpha': alpha, 'beta': beta}
    values: dict[str, Any] = {name: value for name, value in initial.items() if value is not None}
    return System(elements=[element('circuit', z=z, **values)])


@register_model(
    field='engineering/circuits',
    tags=('circuit', 'mixed-mode oscillations', 'canards', 'slow-fast', 'ode'),
    references=('Krupa, Popović, and Kopell 2008', 'Trzaska 2023'),
    defaults={'until': 200.0, 'dt': 0.5, 'solver': ScipyODESolver(method='Radau')},
)
def mixed_mode_chua(
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
) -> System:
    """Return a modified Chua circuit on three time scales, whose oscillations mix small and large amplitudes.

    The circuit follows `ε x' = -y + (a + b x) x^2`, `y' = alpha (x - R y - z)`, and `z' = c - β y`. For `ε = 0.01`,
    `a = 1.5`, `b = -1`, and `β = 0.005`, the circuit has small oscillations only for `c = 0`, large ones only for
    `c = 0.00105`, and mixed-mode oscillations for `c = 0.00053`.

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

    Returns:
        system:
            The model, with the states `circuit.x`, `circuit.y`, and `circuit.z`.

    Examples:
        >>> from skcomplex.models.engineering.circuits import mixed_mode_chua
        >>> mixed_mode_chua(c=0.0).elements[0].c
        0.0
    """
    circuit = MixedModeChuaCircuitElement(
        'circuit',
        x=x,
        y=y,
        z=z,
        epsilon=epsilon,
        a=a,
        b=b,
        alpha=alpha,
        R=R,
        beta=beta,
        c=c,
    )
    return System(elements=[circuit])


@register_model(
    field='engineering/circuits',
    tags=('circuit', 'memristor', 'chaos', 'ode'),
    references=('Muthuswamy and Chua 2010', 'Trzaska 2023'),
    defaults={'until': 200.0, 'dt': 0.1},
)
def memristor_oscillator(
    v: float = 0.1,
    i: float = 0.0,
    w: float = 0.1,
    *,
    C: float = 1.0,
    L: float = 3.0,
    alpha: float = 0.6,
    beta: float = 1.5,
) -> System:
    """Return the simplest memristor oscillator: a capacitor, an inductor, and a memristor in series.

    The capacitor voltage, the inductor current, and the internal state of the memristor follow `C v' = i`,
    `L i' = -v - β (w^2 - 1) i`, and `w' = -i - alpha w + w i`. The memristor has the memristance `β (w^2 - 1)`, and the
    circuit oscillates chaotically for the default values.

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

    Returns:
        system:
            The model, with the states `circuit.v`, `circuit.i`, and `circuit.w`.

    Examples:
        >>> from skcomplex.models.engineering.circuits import memristor_oscillator
        >>> memristor_oscillator(L=2.0).elements[0].L
        2.0
    """
    return System(elements=[MemristorCircuitElement('circuit', v=v, i=i, w=w, C=C, L=L, alpha=alpha, beta=beta)])


@register_model(
    field='engineering/circuits',
    tags=('circuit', 'oscillation', 'limit cycle', 'ode'),
    references=('van der Pol 1926', 'Trzaska 2023'),
    defaults={'until': 50.0, 'dt': 0.05},
)
def tunnel_diode_oscillator(
    V: float = 0.1,
    i: float = 0.0,
    *,
    L: float = 1.0,
    C: float = 1.0,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> System:
    """Return a tunnel-diode oscillator: an inductor and a capacitor in parallel with a negative-resistance diode.

    The diode current is `h(V) = gamma V^3 - alpha V`, so Kirchhoff's laws give `C V' = -h(V) - i` and `L i' = V`,
    which is the van der Pol oscillator with `ε = alpha sqrt(L / C)`. The voltage settles on a limit cycle of
    amplitude near `2 sqrt(alpha / (3 gamma))`.

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

    Returns:
        system:
            The model, with the voltage `circuit.V` and the inductor current `circuit.i`.

    Examples:
        >>> from skcomplex.models.engineering.circuits import tunnel_diode_oscillator
        >>> tunnel_diode_oscillator(alpha=2.0).elements[0].alpha
        2.0
    """
    return System(elements=[TunnelDiodeCircuitElement('circuit', V=V, i=i, L=L, C=C, alpha=alpha, gamma=gamma)])


@register_model(
    field='engineering/circuits',
    tags=('circuit', 'impulses', 'events', 'hybrid'),
    references=('Trzaska 2023',),
    defaults={'until': 40.0, 'dt': 0.05},
)
def impulsive_circuits(
    x: float = 0.0,
    current: float = 0.0,
    *,
    L: float = 1.0,
    R: float = 0.1,
    C: float = 1.0,
    A: float = 1.0,
    T: float = 2.0,
) -> System:
    """Return an RLC circuit driven by a periodic train of current impulses, `L C x'' + R C x' + x = A Σ δ(t - k T)`.

    Every impulse is an event that raises the rate of change `x'` by `A / (L C)`, and between impulses the circuit
    rings down freely. The response settles on a periodic, non-sinusoidal waveform of period `T`.

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

    Returns:
        system:
            The model, with the response `circuit.x`, its rate `circuit.x_d1`, and the time since the last impulse
            `circuit.clock`.

    Examples:
        >>> from skcomplex.models.engineering.circuits import impulsive_circuits
        >>> impulsive_circuits(T=3.0).elements[0].T
        3.0
    """
    return System(elements=[KickedCircuitElement('circuit', x=x, x_d1=current, clock=0.0, L=L, R=R, C=C, A=A, T=T)])
