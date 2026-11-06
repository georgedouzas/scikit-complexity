"""Neuroscience systems: firing rates, phase neurons, and adaptive neurons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, sin
from scipy import stats

from ....core.components.elements import GroupElement
from ....core.components.interactions import RateInteraction, RelaxationInteraction
from ....core.components.laws.utils import logistic
from ....core.components.system import System
from ..._catalog import register_system
from ._mutual_inhibition import MutualInhibitionInteraction
from ._polyhomeostatic_neuron import PolyhomeostaticNeuronElement
from ._pulse_coupling import PulseCouplingInteraction
from ._rate_unit import RateUnitElement
from ._theta_neuron import ThetaNeuronElement
from ._wilson_cowan import WilsonCowanElement


def _lorentzian_quantiles(center: float, width: float, size: int) -> list[float]:
    """Return evenly spaced quantiles of a Lorentzian distribution."""
    levels = (np.arange(size) + 0.5) / size
    return [float(value) for value in center + width * stats.cauchy.ppf(levels)]


def _response(x: Expression, gain: Expression, threshold: Expression) -> Expression:
    """Return the response `logistic(x) - logistic(0)` of a sigmoid of a gain and a threshold, 0 at no input."""
    response: Expression = logistic(x, threshold, 1 / gain) - logistic(0, threshold, 1 / gain)
    return response


def _saturation(gain: Expression, threshold: Expression) -> Expression:
    """Return the largest value of the response of a sigmoid of a gain and a threshold."""
    saturation: Expression = 1 - logistic(0, threshold, 1 / gain)
    return saturation


def _excitation(s: SimpleNamespace) -> Expression:
    """Return the response of the excitatory population of Wilson and Cowan to its input."""
    c = s.target
    excitation: Expression = (_saturation(c.a_e, c.theta_e) - c.r_e * c.E) * _response(
        c.c1 * c.E - c.c2 * c.I + c.P,
        c.a_e,
        c.theta_e,
    )
    return excitation


def _inhibition(s: SimpleNamespace) -> Expression:
    """Return the response of the inhibitory population of Wilson and Cowan to its input."""
    c = s.target
    inhibition: Expression = (_saturation(c.a_i, c.theta_i) - c.r_i * c.I) * _response(
        c.c3 * c.E - c.c4 * c.I + c.Q,
        c.a_i,
        c.theta_i,
    )
    return inhibition


def _kl_gradient(s: SimpleNamespace) -> Expression:
    """Return the factor `d = 1 - (2 + 1/mu) y + y²/mu` of the gradient of the polyhomeostatic neuron."""
    neuron = s.target
    rate = logistic(neuron.x, neuron.b, 1 / neuron.a)
    gradient: Expression = 1 - (2 + 1 / neuron.mu) * rate + rate**2 / neuron.mu
    return gradient


@register_system(
    field='biology/neuroscience',
    tags=('neural mass', 'oscillations', 'excitation-inhibition'),
    references=('Wilson and Cowan 1972', 'arXiv:2112.12147'),
    defaults={'until': 100.0, 'dt': 0.2},
)
def wilson_cowan(
    E: float = 0.1,
    I: float = 0.05,
    *,
    P: float = 1.25,
    Q: float = 0.0,
    c1: float = 16.0,
    c2: float = 12.0,
    c3: float = 15.0,
    c4: float = 3.0,
) -> System:
    """Return the Wilson-Cowan system of interacting excitatory and inhibitory populations of neurons.

    With the published couplings, a moderate excitatory input `P = 1.25` gives a limit cycle of population activity,
    and no input lets the activity decay to rest.

    Args:
        E:
            The initial excitatory activity.

        I:
            The initial inhibitory activity.

        P:
            The external input to the excitatory population.

        Q:
            The external input to the inhibitory population.

        c1:
            The excitatory to excitatory coupling.

        c2:
            The inhibitory to excitatory coupling.

        c3:
            The excitatory to inhibitory coupling.

        c4:
            The inhibitory to inhibitory coupling.

    Returns:
        system:
            The system, with the activities `cortex.E` and `cortex.I`.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import wilson_cowan
        >>> wilson_cowan(P=0.0).elements[0].P
        0.0
    """
    cortex = WilsonCowanElement('cortex', E=E, I=I, P=P, Q=Q, c1=c1, c2=c2, c3=c3, c4=c4)
    relaxations = [
        RelaxationInteraction(
            'excitatory_relaxation',
            target='cortex',
            state='E',
            goal=_excitation,
            time=lambda s: s.target.tau_e,
        ),
        RelaxationInteraction(
            'inhibitory_relaxation',
            target='cortex',
            state='I',
            goal=_inhibition,
            time=lambda s: s.target.tau_i,
        ),
    ]
    return System(elements=[cortex], interactions=relaxations)


@register_system(
    field='biology/neuroscience',
    tags=('neuron', 'phase oscillators', 'mean field', 'network'),
    references=('Ermentrout and Kopell 1986', 'Luke, Barreto, and So 2013', 'arXiv:1902.05307'),
    defaults={'until': 20.0, 'dt': 0.1},
)
def theta_neurons(
    n_neurons: int = 50,
    *,
    eta: float = -0.2,
    delta: float = 0.1,
    kappa: float = 1.5,
    n: int = 2,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a network of theta neurons, equivalently quadratic integrate-and-fire neurons, coupled all to all.

    Neuron `i` follows `theta_i' = (1 - cos theta_i) + (1 + cos theta_i)(eta_i + kappa I)`, where `I` is the mean pulse
    `a_n (1 - cos theta_j)^n` of all neurons at the current state. The excitabilities are evenly spaced quantiles of a
    Lorentzian distribution of center `eta` and half-width `delta`, the distribution for which Ott and Antonsen's
    ansatz reduces the network exactly to one complex equation. Strong excitatory coupling switches the network from a
    quiet state to collective firing.

    Args:
        n_neurons:
            The number of neurons.

        eta:
            The center of the excitabilities.

        delta:
            The half-width of the excitabilities.

        kappa:
            The coupling strength.

        n:
            The sharpness of the pulses, a positive integer.

        random_state:
            The seed or generator of the initial phases.

    Returns:
        system:
            The system, with the phases `neurons.theta` as its only state.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import theta_neurons
        >>> theta_neurons(20).elements[0].prepare().size_
        20
    """
    phases = np.random.default_rng(random_state).uniform(0.0, 2 * math.pi, n_neurons)
    neurons = GroupElement(
        'neurons',
        element=ThetaNeuronElement,
        size=n_neurons,
        values={'theta': phases, 'eta': np.asarray(_lorentzian_quantiles(eta, delta, n_neurons))},
    )
    return System(
        elements=[neurons],
        interactions=[PulseCouplingInteraction('coupling', neurons='neurons', kappa=kappa, n=n)],
    )


@register_system(
    field='biology/neuroscience',
    tags=('neural network', 'competition', 'winner-take-all'),
    references=('Fukai and Tanaka 1997',),
    defaults={'until': 40.0, 'dt': 0.2},
)
def winners_take_all(
    inputs: Sequence[float] = (1.0, 1.1, 0.9, 1.05, 0.95),
    *,
    beta: float = 2.0,
    tau: float = 1.0,
    x: float = 0.1,
) -> System:
    """Return a winner-take-all network of units with self-limited growth and mutual inhibition.

    Unit `i` follows `tau x_i' = x_i (I_i - x_i - beta sum_{j != i} x_j)`, the competitive network of Fukai and
    Tanaka. One inhibition interaction binds every unit as an element set and reads the total activity of the set, so
    the network holds one interaction whatever the number of units. When the inhibition `beta` exceeds 1, only the unit
    with the largest input stays active, at activity `I_max`. When it is below 1, several winners share the activity.
    The winners-take-all network of Gros (2015, section 10.4), with reservoir variables and clique encoding, is a
    different model.

    Args:
        inputs:
            The input of every unit.

        beta:
            The strength of the mutual inhibition.

        tau:
            The time constant of the units.

        x:
            The initial activity of every unit.

    Returns:
        system:
            The system, with the activities `unit_<i>.x`.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import winners_take_all
        >>> len(winners_take_all((1.0, 2.0, 3.0)).interactions)
        1
    """
    units = [RateUnitElement(f'unit_{index}', x=x, I=value, tau=tau) for index, value in enumerate(inputs)]
    inhibition = MutualInhibitionInteraction('inhibition', units=[unit.label for unit in units], beta=beta)
    return System(elements=units, interactions=[inhibition])


@register_system(
    field='biology/neuroscience',
    tags=('neuron', 'homeostasis', 'adaptation', 'information'),
    references=('Triesch 2005', 'Markovic and Gros 2010', 'Gros 2015'),
    defaults={'until': 400.0, 'dt': 1.0},
)
def polyhomeostatic_neuron(
    *,
    mu: float = 0.2,
    eps_a: float = 0.05,
    eps_b: float = 0.05,
    I0: float = 0.0,
    A: float = 1.0,
    omega: float = 1.0,
) -> System:
    """Return a neuron that adapts its gain and threshold so its firing rate approaches an exponential distribution.

    Polyhomeostatic optimization targets the whole distribution of the firing rate, the one of maximal entropy for a
    given mean `mu`, rather than its mean alone. The neuron is driven by a periodic input, and the time average of its
    rate approaches `mu` as gain and threshold adapt.

    Args:
        mu:
            The target mean firing rate.

        eps_a:
            The adaptation rate of the gain.

        eps_b:
            The adaptation rate of the threshold.

        I0:
            The mean input.

        A:
            The amplitude of the periodic input.

        omega:
            The angular frequency of the periodic input.

    Returns:
        system:
            The system, with the potential `neuron.x`, the gain `neuron.a`, and the threshold `neuron.b`.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import polyhomeostatic_neuron
        >>> polyhomeostatic_neuron(mu=0.1).elements[0].mu
        0.1
    """
    neuron = PolyhomeostaticNeuronElement('neuron', mu=mu, eps_a=eps_a, eps_b=eps_b, I0=I0, A=A, omega=omega)
    interactions = [
        RelaxationInteraction(
            'membrane',
            target='neuron',
            state='x',
            goal=lambda s: s.target.I0 + s.target.A * sin(s.target.omega * s.t),
            time=lambda s: 1 / s.target.Gamma,
        ),
        RateInteraction(
            'gain_plasticity',
            target='neuron',
            state='a',
            rate=lambda s: s.target.eps_a * (1 / s.target.a + (s.target.x - s.target.b) * _kl_gradient(s)),
        ),
        RateInteraction(
            'threshold_plasticity',
            target='neuron',
            state='b',
            rate=lambda s: -s.target.eps_b * s.target.a * _kl_gradient(s),
        ),
    ]
    return System(elements=[neuron], interactions=interactions)
