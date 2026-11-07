"""Electric circuits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.interactions import ImpulseInteraction
from ....core.components.system import System
from ....core.engine.solvers import ScipyODESolver
from ....core.errors import ParameterError
from ..._catalog import register_system
from ...physics.synchronization import DiffusiveCouplingInteraction
from ._chua_circuit import ChuaCircuitElement
from ._circuit import CircuitElement
from ._colpitts_oscillator import ColpittsOscillatorElement
from ._kicked_circuit import KickedCircuitElement
from ._memristive_circuit import MemristiveCircuitElement
from ._memristor_circuit import MemristorCircuitElement
from ._mixed_mode_chua_circuit import MixedModeChuaCircuitElement
from ._multiscroll_chua_circuit import MultiscrollChuaCircuitElement
from ._tunnel_diode_circuit import TunnelDiodeCircuitElement

CHUA_NONLINEARITIES = ('piecewise-linear', 'sine')


@register_system(
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
            The system, with the circuit `circuit`, whose charge has order 2 and current `circuit.q_d1`.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import rlc_circuit
        >>> rlc_circuit(R=0.5).elements[0].R
        0.5
    """
    return System(elements=[CircuitElement('circuit', q=q, q_d1=current, L=L, R=R, C=C, V=V, omega=omega)])


@register_system(
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
            The system, with the states `circuit.x`, `circuit.y`, and `circuit.z`.

    Raises:
        ParameterError:
            The nonlinearity is neither `'piecewise-linear'` nor `'sine'`.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import chua
        >>> chua(nonlinearity='sine').elements[0].alpha
        10.82
    """
    if nonlinearity not in CHUA_NONLINEARITIES:
        error_msg = f"Unknown nonlinearity {nonlinearity!r}. Use 'piecewise-linear' or 'sine'."
        raise ParameterError(error_msg)
    element = ChuaCircuitElement if nonlinearity == 'piecewise-linear' else MultiscrollChuaCircuitElement
    initial = {'x': x, 'y': y, 'alpha': alpha, 'beta': beta}
    values: dict[str, float] = {name: value for name, value in initial.items() if value is not None}
    return System(elements=[element('circuit', z=z).set_params(**values)])


@register_system(
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
            The system, with the states `circuit.x`, `circuit.y`, and `circuit.z`.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import mixed_mode_chua
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


@register_system(
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
            The system, with the states `circuit.v`, `circuit.i`, and `circuit.w`.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import memristor_oscillator
        >>> memristor_oscillator(L=2.0).elements[0].L
        2.0
    """
    return System(elements=[MemristorCircuitElement('circuit', v=v, i=i, w=w, C=C, L=L, alpha=alpha, beta=beta)])


@register_system(
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
            The system, with the voltage `circuit.V` and the inductor current `circuit.i`.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import tunnel_diode_oscillator
        >>> tunnel_diode_oscillator(alpha=2.0).elements[0].alpha
        2.0
    """
    return System(elements=[TunnelDiodeCircuitElement('circuit', V=V, i=i, L=L, C=C, alpha=alpha, gamma=gamma)])


@register_system(
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
            The system, with the response `circuit.x` and its rate `circuit.x_d1`, kicked by the impulses `impulse`.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import impulsive_circuits
        >>> impulsive_circuits(T=3.0).interactions[0].period
        3.0
    """
    circuit = KickedCircuitElement('circuit', x=x, x_d1=current, L=L, R=R, C=C, A=A)
    kick = {'x_d1': lambda s: s.target.x_d1 + s.target.A / (s.target.L * s.target.C)}
    impulse = ImpulseInteraction('impulse', target='circuit', reset=kick, period=T)
    return System(elements=[circuit], interactions=[impulse])


@register_system(
    field='engineering/circuits',
    tags=('circuit', 'memristor', 'mixed-mode oscillations', 'slow-fast', 'ode'),
    references=('Trzaska 2023',),
    defaults={'until': 150.0, 'dt': 0.02, 'solver': ScipyODESolver(method='LSODA')},
)
def memristive_mixed_mode_oscillator(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    w: float = 0.0,
    *,
    epsilon: float = 0.01,
    R: float = 2.0,
    alpha: float = 1.55,
    beta: float = 0.15,
    e: float = -0.04,
    a: float = -0.34,
    b: float = 0.05,
) -> System:
    """Return a slow-fast circuit with a cubic flux-controlled memristor, whose oscillations mix two amplitudes.

    The memristor has the charge `q = a w + b w^3` of its flux `w`, and with the voltage `x` of the memristor, the
    current `y` of the inductor, and the voltage `z` of the second capacitor, the circuit follows
    `epsilon x' = -y - (a + 3 b w^2) x`, `y' = alpha (x - R y - z + e)`, `z' = -β y`, and `w' = x`. Trzaska prints the
    memductance as `a + 3 b w`, but only the derivative `a + 3 b w^2` of the characteristic gives the response he
    reports: for the defaults, from rest, the voltage settles on the mixed-mode oscillation `2^11`, two large spikes
    and eleven small oscillations in a period of about 7.8.

    Args:
        x:
            The initial voltage of the memristor.

        y:
            The initial current of the inductor.

        z:
            The initial voltage of the second capacitor.

        w:
            The initial flux of the memristor.

        epsilon:
            The first capacitance, which sets the fast time scale.

        R:
            The resistance.

        alpha:
            The inverse inductance.

        beta:
            The gain of the controlled source over the second capacitance.

        e:
            The voltage of the source.

        a:
            The linear coefficient of the memristor characteristic.

        b:
            The cubic coefficient of the memristor characteristic.

    Returns:
        system:
            The system, with the states `circuit.x`, `circuit.y`, `circuit.z`, and `circuit.w`.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import memristive_mixed_mode_oscillator
        >>> memristive_mixed_mode_oscillator(a=-0.5).elements[0].a
        -0.5
    """
    circuit = MemristiveCircuitElement(
        'circuit',
        x=x,
        y=y,
        z=z,
        w=w,
        epsilon=epsilon,
        R=R,
        alpha=alpha,
        beta=beta,
        e=e,
        a=a,
        b=b,
    )
    return System(elements=[circuit])


@register_system(
    field='engineering/circuits',
    tags=('circuit', 'synchronization', 'chaos', 'ode'),
    references=('Kennedy 1994', 'Datseris and Parlitz 2022'),
    defaults={'until': 500.0, 'dt': 0.05},
)
def coupled_colpitts(
    k: float = 0.4,
    *,
    omegas: tuple[float, float] = (1.02, 0.98),
    a: float = 30.0,
    b: float = 1.0,
    c: float = 20.0,
    d: float = 1.0,
    epsilon: float = 1.0,
) -> System:
    """Return two detuned chaotic Colpitts oscillators coupled through the difference of their second variables.

    Oscillator `i` follows `x' = omega_i y - a F(z)`, `y' = c - omega_i x - z - b y + k (y_j - y_i)`, and
    `z' = epsilon (y - d)`, with the transistor characteristic `F(z) = max(0, -1 - z)`. The phase of an oscillator
    is the angle of its orbit in the plane of `y` and `z`, around the means of both. Without coupling the two
    oscillators run at different mean frequencies and their phases drift apart, while for `k` near 0.3 to 0.5 their
    phases lock although their amplitudes stay irregular.

    Args:
        k:
            The coupling strength.

        omegas:
            The natural frequencies of the two oscillators.

        a:
            The gain of the transistor.

        b:
            The damping of the second variable.

        c:
            The bias of the second variable.

        d:
            The offset of the third variable.

        epsilon:
            The time scale of the third variable.

    Returns:
        system:
            The system, with the states `oscillator_0.x`, `oscillator_0.y`, `oscillator_0.z`, and the same states of
            `oscillator_1`.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import coupled_colpitts
        >>> [element.omega for element in coupled_colpitts().elements]
        [1.02, 0.98]
    """
    starts = ((0.1, 0.0, 0.0), (0.0, 0.1, 0.2))
    oscillators = [
        ColpittsOscillatorElement(
            f'oscillator_{index}',
            x=start[0],
            y=start[1],
            z=start[2],
            omega=omega,
            a=a,
            b=b,
            c=c,
            d=d,
            epsilon=epsilon,
        )
        for index, (omega, start) in enumerate(zip(omegas, starts, strict=True))
    ]
    couplings = [
        DiffusiveCouplingInteraction(f'coupling_{target}', source=source, target=target, state='y', k=k)
        for source, target in (('oscillator_1', 'oscillator_0'), ('oscillator_0', 'oscillator_1'))
    ]
    return System(elements=oscillators, interactions=couplings)
