"""Neuroscience models: conductance-based and reduced excitable neurons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, Expression, parameter, state
from ....core.model import Element, Event, Law, Rate, Symbols, System
from ....core.runtime import register_model

type GateRates = dict[str, tuple[Expression, Expression]]


def _gate_rates(v: Expression) -> GateRates:
    """Return the opening and closing rates per ms of the gates `n`, `m`, and `h` at a potential in mV."""
    return {
        'n': (0.01 * (v + 55) / (1 - exp(-(v + 55) / 10)), 0.125 * exp(-(v + 65) / 80)),
        'm': (0.1 * (v + 40) / (1 - exp(-(v + 40) / 10)), 4 * exp(-(v + 65) / 18)),
        'h': (0.07 * exp(-(v + 65) / 20), 1 / (1 + exp(-(v + 35) / 10))),
    }


def _resting_gates(v: float) -> dict[str, float]:
    """Return the steady-state opening of every gate at a fixed potential in mV."""
    return {gate: float(opening / (opening + closing)) for gate, (opening, closing) in _gate_rates(v).items()}


class HodgkinHuxley(Element):
    """Squid giant axon membrane of Hodgkin and Huxley, with sodium, potassium, and leak currents.

    The potential follows `C v' = I - g_Na m³ h (v - E_Na) - g_K n⁴ (v - E_K) - g_L (v - E_L)`, and each gate `x` of
    `n`, `m`, and `h` relaxes as `x' = alpha_x(v) (1 - x) - beta_x(v) x`, in the modern convention with rest near -65
    mV. Upward crossings of 0 mV are recorded as `spike` events. Units are mV, ms, mS/cm², and µA/cm².

    Args:
        label:
            The label of the neuron.

        v:
            The initial membrane potential.

        n:
            The initial potassium activation.

        m:
            The initial sodium activation.

        h:
            The initial sodium inactivation.

        C:
            The membrane capacitance.

        g_na:
            The maximal sodium conductance.

        g_k:
            The maximal potassium conductance.

        g_l:
            The leak conductance.

        e_na:
            The sodium reversal potential.

        e_k:
            The potassium reversal potential.

        e_l:
            The leak reversal potential.

        I:
            The applied current.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import HodgkinHuxley
        >>> HodgkinHuxley('axon', I=10.0).g_na
        120.0
    """

    v: float = state(default=-65.0)
    n: float = state(default=0.3177, domain=Domain.NONNEGATIVE)
    m: float = state(default=0.0529, domain=Domain.NONNEGATIVE)
    h: float = state(default=0.5961, domain=Domain.NONNEGATIVE)
    C: float = parameter(default=1.0, domain=Domain.POSITIVE)
    g_na: float = parameter(default=120.0, domain=Domain.NONNEGATIVE)
    g_k: float = parameter(default=36.0, domain=Domain.NONNEGATIVE)
    g_l: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    e_na: float = parameter(default=50.0)
    e_k: float = parameter(default=-77.0)
    e_l: float = parameter(default=-54.387)
    I: float = parameter(default=10.0)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the membrane equation, the gate kinetics, and the spike events.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        currents = s.g_na * s.m**3 * s.h * (s.v - s.e_na) + s.g_k * s.n**4 * (s.v - s.e_k) + s.g_l * (s.v - s.e_l)
        gates = [
            Rate(getattr(s, gate), opening * (1 - getattr(s, gate)) - closing * getattr(s, gate))
            for gate, (opening, closing) in _gate_rates(s.v).items()
        ]
        return [Rate(s.v, (s.I - currents) / s.C), *gates, Event(s.v, None, direction=1, label='spike')]


class FitzHughNagumoNeuron(Element):
    """FitzHugh-Nagumo neuron, a fast potential with a cubic nullcline and a slow linear recovery.

    The potential follows `v' = v - v³/3 - w + I` and the recovery `w' = epsilon (v + a - b w)`.

    Args:
        label:
            The label of the neuron.

        v:
            The initial potential.

        w:
            The initial recovery.

        a:
            The offset of the recovery.

        b:
            The damping of the recovery.

        epsilon:
            The time scale of the recovery.

        I:
            The applied current.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import FitzHughNagumoNeuron
        >>> FitzHughNagumoNeuron('neuron').epsilon
        0.08
    """

    v: float = state(default=-1.2)
    w: float = state(default=-0.6)
    a: float = parameter(default=0.7)
    b: float = parameter(default=0.8, domain=Domain.NONNEGATIVE)
    epsilon: float = parameter(default=0.08, domain=Domain.POSITIVE)
    I: float = parameter(default=0.5)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the fast potential and the slow recovery.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.v, s.v - s.v**3 / 3 - s.w + s.I), Rate(s.w, s.epsilon * (s.v + s.a - s.b * s.w))]


@register_model(
    field='biology/neuroscience',
    tags=('neuron', 'excitability', 'spiking', 'conductance'),
    references=('Hodgkin and Huxley 1952', 'Rinzel and Miller 1980', 'Trzaska 2023', 'arXiv:2112.12147'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def hodgkin_huxley(
    v: float = -65.0,
    *,
    I: float = 10.0,
    C: float = 1.0,
    g_na: float = 120.0,
    g_k: float = 36.0,
    g_l: float = 0.3,
) -> System:
    """Return the Hodgkin-Huxley neuron, whose sodium and potassium channels produce action potentials.

    The gates start at their steady state at the initial potential, and the current is switched on at the start. A
    constant current above about 6.2 µA/cm² makes the neuron fire periodically, and a weaker one gives at most a few
    spikes before it settles.

    Args:
        v:
            The initial membrane potential in mV.

        I:
            The applied current in µA/cm².

        C:
            The membrane capacitance in µF/cm².

        g_na:
            The maximal sodium conductance in mS/cm².

        g_k:
            The maximal potassium conductance in mS/cm².

        g_l:
            The leak conductance in mS/cm².

    Returns:
        system:
            The model, with the potential `neuron.v`, the gates `neuron.n`, `neuron.m`, and `neuron.h`, and its spikes
            as `spike` events.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import hodgkin_huxley
        >>> round(hodgkin_huxley().elements[0].h, 4)
        0.5961
    """
    gates = _resting_gates(v)
    return System(
        elements=[
            HodgkinHuxley(
                'neuron',
                v=v,
                I=I,
                C=C,
                g_na=g_na,
                g_k=g_k,
                g_l=g_l,
                n=gates['n'],
                m=gates['m'],
                h=gates['h'],
            ),
        ],
    )


@register_model(
    field='biology/neuroscience',
    tags=('neuron', 'excitability', 'relaxation oscillations'),
    references=('FitzHugh 1961', 'Nagumo, Arimoto, and Yoshizawa 1962', 'Trzaska 2023', 'Datseris and Parlitz 2022'),
    defaults={'until': 100.0, 'dt': 0.2},
)
def fitzhugh_nagumo_ode(
    v: float = -1.2,
    w: float = -0.6,
    *,
    a: float = 0.7,
    b: float = 0.8,
    epsilon: float = 0.08,
    I: float = 0.5,
) -> System:
    """Return the FitzHugh-Nagumo neuron, the two-variable reduction of Hodgkin and Huxley's excitable membrane.

    Without current the neuron rests on a stable fixed point and answers a large enough kick with one spike. A current
    `I` between about 0.33 and 1.42 destabilizes the rest and the neuron fires periodically.

    Args:
        v:
            The initial potential.

        w:
            The initial recovery.

        a:
            The offset of the recovery.

        b:
            The damping of the recovery.

        epsilon:
            The time scale of the recovery.

        I:
            The applied current.

    Returns:
        system:
            The model, with the potential `neuron.v` and the recovery `neuron.w`.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import fitzhugh_nagumo_ode
        >>> fitzhugh_nagumo_ode(I=0.0).elements[0].I
        0.0
    """
    return System(elements=[FitzHughNagumoNeuron('neuron', v=v, w=w, a=a, b=b, epsilon=epsilon, I=I)])
