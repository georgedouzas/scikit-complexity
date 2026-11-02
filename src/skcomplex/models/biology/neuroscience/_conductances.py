"""Neuroscience models: conductance-based and reduced excitable neurons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, exp

from ....core.interactions import CrossingInteraction, RateInteraction
from ....core.system import System
from ..._catalog import register_model
from ..common.excitable import FitzHughNagumoInteraction
from ._fitz_hugh_nagumo_neuron import FitzHughNagumoNeuronElement
from ._hodgkin_huxley import HodgkinHuxleyElement
from .types import GateRates


def _gate_rates(v: Expression) -> GateRates:
    """Return the opening and closing rates per ms of the gates `n`, `m`, and `h` at a potential in mV."""
    return {
        'n': (0.01 * (v + 55) / (1 - exp(-(v + 55) / 10)), 0.125 * exp(-(v + 65) / 80)),
        'm': (0.1 * (v + 40) / (1 - exp(-(v + 40) / 10)), 4 * exp(-(v + 65) / 18)),
        'h': (0.07 * exp(-(v + 65) / 20), 1 / (1 + exp(-(v + 35) / 10))),
    }


def _gate_kinetics(gate: str) -> Callable[[SimpleNamespace], Expression]:
    """Return the rate `alpha(v) (1 - x) - beta(v) x` of a gate `x` as a function of the namespace of the neuron."""

    def kinetics(s: SimpleNamespace) -> Expression:
        opening, closing = _gate_rates(s.target.v)[gate]
        x = getattr(s.target, gate)
        rate: Expression = opening * (1 - x) - closing * x
        return rate

    return kinetics


def _resting_gates(v: float) -> dict[str, float]:
    """Return the steady-state opening of every gate at a fixed potential in mV."""
    return {gate: float(opening / (opening + closing)) for gate, (opening, closing) in _gate_rates(v).items()}


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
            HodgkinHuxleyElement(
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
        interactions=[
            RateInteraction('applied_current', target='neuron', state='v', rate=lambda s: s.target.I, divisor='C'),
            RateInteraction(
                'sodium_current',
                target='neuron',
                state='v',
                rate=lambda s: -s.target.g_na * s.target.m**3 * s.target.h * (s.target.v - s.target.e_na),
                divisor='C',
            ),
            RateInteraction(
                'potassium_current',
                target='neuron',
                state='v',
                rate=lambda s: -s.target.g_k * s.target.n**4 * (s.target.v - s.target.e_k),
                divisor='C',
            ),
            RateInteraction(
                'leak_current',
                target='neuron',
                state='v',
                rate=lambda s: -s.target.g_l * (s.target.v - s.target.e_l),
                divisor='C',
            ),
            *[
                RateInteraction(f'{gate}_gating', target='neuron', state=gate, rate=_gate_kinetics(gate))
                for gate in ('n', 'm', 'h')
            ],
            CrossingInteraction('spike', target='neuron', guard=lambda s: s.target.v, direction=1),
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
        >>> fitzhugh_nagumo_ode(I=0.0).interactions[0].I
        0.0
    """
    kinetics = FitzHughNagumoInteraction('kinetics', target='neuron', a=a, b=b, epsilon=epsilon, I=I)
    return System(elements=[FitzHughNagumoNeuronElement('neuron', v=v, w=w)], interactions=[kinetics])
