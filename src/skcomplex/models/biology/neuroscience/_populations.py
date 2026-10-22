"""Neuroscience models: firing rates, phase neurons, and adaptive neurons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import numpy as np
from sage.all__sagemath_symbolics import cos, exp, sin

from ....core.base import Domain, Expression, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Law, Rate, Symbols, mean, total
from ....core.runtime import register_model
from ....core.system import System


def _sigmoid(x: Expression, gain: Expression, threshold: Expression) -> Expression:
    """Return the Wilson-Cowan response `1 / (1 + exp(-gain (x - threshold))) - 1 / (1 + exp(gain threshold))`."""
    return 1 / (1 + exp(-gain * (x - threshold))) - 1 / (1 + exp(gain * threshold))


def _saturation(gain: Expression, threshold: Expression) -> Expression:
    """Return the largest value of the Wilson-Cowan response."""
    return 1 - 1 / (1 + exp(gain * threshold))


def _lorentzian_quantiles(center: float, width: float, size: int) -> list[float]:
    """Return evenly spaced quantiles of a Lorentzian distribution."""
    return [center + width * math.tan(math.pi * ((index + 0.5) / size - 0.5)) for index in range(size)]


class WilsonCowan(Element):
    """Excitatory and inhibitory populations of Wilson and Cowan, as fractions of active cells.

    The excitatory activity follows `tau_e E' = -E + (k_e - r_e E) S_e(c1 E - c2 I + P)` and the inhibitory activity
    `tau_i I' = -I + (k_i - r_i I) S_i(c3 E - c4 I + Q)`, where `S(x) = 1 / (1 + exp(-a (x - theta))) - 1 / (1 +
    exp(a theta))` is a sigmoid shifted to vanish at zero input and `k` is its largest value.

    Args:
        label:
            The label of the populations.

        E:
            The initial excitatory activity.

        I:
            The initial inhibitory activity.

        c1:
            The excitatory to excitatory coupling.

        c2:
            The inhibitory to excitatory coupling.

        c3:
            The excitatory to inhibitory coupling.

        c4:
            The inhibitory to inhibitory coupling.

        a_e:
            The gain of the excitatory response.

        theta_e:
            The threshold of the excitatory response.

        a_i:
            The gain of the inhibitory response.

        theta_i:
            The threshold of the inhibitory response.

        r_e:
            The refractory period of excitatory cells.

        r_i:
            The refractory period of inhibitory cells.

        P:
            The external input to the excitatory population.

        Q:
            The external input to the inhibitory population.

        tau_e:
            The time constant of the excitatory population.

        tau_i:
            The time constant of the inhibitory population.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import WilsonCowan
        >>> WilsonCowan('cortex', P=1.25).c1
        16.0
    """

    E: float = state(default=0.1, domain=Domain.NONNEGATIVE)
    I: float = state(default=0.05, domain=Domain.NONNEGATIVE)
    c1: float = parameter(default=16.0, domain=Domain.NONNEGATIVE)
    c2: float = parameter(default=12.0, domain=Domain.NONNEGATIVE)
    c3: float = parameter(default=15.0, domain=Domain.NONNEGATIVE)
    c4: float = parameter(default=3.0, domain=Domain.NONNEGATIVE)
    a_e: float = parameter(default=1.3, domain=Domain.POSITIVE)
    theta_e: float = parameter(default=4.0)
    a_i: float = parameter(default=2.0, domain=Domain.POSITIVE)
    theta_i: float = parameter(default=3.7)
    r_e: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    r_i: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    P: float = parameter(default=1.25)
    Q: float = parameter(default=0.0)
    tau_e: float = parameter(default=1.0, domain=Domain.POSITIVE)
    tau_i: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the relaxation of both activities toward their responses.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        excitation = (_saturation(s.a_e, s.theta_e) - s.r_e * s.E) * _sigmoid(
            s.c1 * s.E - s.c2 * s.I + s.P,
            s.a_e,
            s.theta_e,
        )
        inhibition = (_saturation(s.a_i, s.theta_i) - s.r_i * s.I) * _sigmoid(
            s.c3 * s.E - s.c4 * s.I + s.Q,
            s.a_i,
            s.theta_i,
        )
        return [Rate(s.E, (-s.E + excitation) / s.tau_e), Rate(s.I, (-s.I + inhibition) / s.tau_i)]


class ThetaNeuron(Element):
    """Theta neuron of Ermentrout and Kopell, `theta' = (1 - cos theta) + (1 + cos theta) eta`, a spiking phase.

    The neuron fires when its phase passes π. With `eta > 0` it fires periodically with period `π / sqrt(eta)`, and
    with `eta < 0` it is excitable. The change of variable `V = tan(theta / 2)` turns it into the quadratic
    integrate-and-fire neuron `V' = V² + eta`.

    Args:
        label:
            The label of the neuron or the group.

        theta:
            The initial phase.

        eta:
            The excitability.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import ThetaNeuron
        >>> ThetaNeuron.group('neurons', 10, eta=0.5).size
        10
    """

    theta: float = state(default=0.0, domain=Domain.ANGLE)
    eta: float = parameter(default=1.0)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the phase dynamics.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.theta, 1 - cos(s.theta) + (1 + cos(s.theta)) * s.eta)]


class PulseCoupling(Interaction):
    """Global coupling of theta neurons through the mean of the smooth pulses that every neuron emits.

    Every neuron receives the current `(1 + cos theta) kappa I`, where `I = mean_j a_n (1 - cos theta_j)^n` is the mean
    pulse of all neurons at the current state. The pulse peaks when a neuron fires, and `a_n` gives it a unit mean over
    the phase. With `kappa > 0` the coupling is excitatory.

    Args:
        label:
            The label of the interaction.

        neurons:
            The path of the group of theta neurons.

        kappa:
            The coupling strength.

        n:
            The sharpness of the pulses, a positive integer.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import PulseCoupling
        >>> PulseCoupling('coupling', neurons='neurons', kappa=2.0).kappa
        2.0
    """

    neurons: str = role()
    kappa: float = parameter(default=1.0)
    n: int = 2

    def laws(self, s: Symbols) -> list[Law]:
        """Return the current of the mean pulse into every neuron.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        theta = s.neurons.theta
        normalization = 2**self.n * math.factorial(self.n) ** 2 / math.factorial(2 * self.n)
        pulses = mean(normalization * (1 - cos(theta)) ** self.n)
        return [Rate(theta, (1 + cos(theta)) * s.kappa * pulses)]


class RateUnit(Element):
    """Neural unit of a competitive network, whose activity grows with its input and saturates, `tau x' = x (I - x)`.

    Args:
        label:
            The label of the unit.

        x:
            The initial activity.

        I:
            The input.

        tau:
            The time constant.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import RateUnit
        >>> RateUnit('unit', I=1.2).I
        1.2
    """

    x: float = state(default=0.1, domain=Domain.NONNEGATIVE)
    I: float = parameter(default=1.0)
    tau: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the self-limited growth of the activity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.x, s.x * (s.I - s.x) / s.tau)]


class MutualInhibition(Interaction):
    """Inhibition of every unit by the summed activity of all the other units, `-beta x_i sum_{j != i} x_j / tau_i`.

    Args:
        label:
            The label of the interaction.

        units:
            The paths of the units, an element set, or the path of a group of units.

        beta:
            The strength of the inhibition.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import MutualInhibition
        >>> MutualInhibition('inhibition', units=['a', 'b'], beta=2.0).beta
        2.0
    """

    units: str | Sequence[str] = role()
    beta: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `-beta x_i (sum_j x_j - x_i) / tau_i`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        unit = s.units
        return [Rate(unit.x, -s.beta * unit.x * (total(unit.x) - unit.x) / unit.tau)]


class PolyhomeostaticNeuron(Element):
    """Rate neuron that adapts its gain and threshold so that its firing rate becomes exponentially distributed.

    The membrane potential `x` integrates the input `I0 + A sin(omega t)` with rate `Gamma`, and the neuron fires at
    rate `y = 1 / (1 + exp(a (b - x)))`. Stochastic gradient descent on the Kullback-Leibler divergence between the
    distribution of `y` and an exponential of mean `mu` gives `a' = eps_a (1/a + (x - b) d)` and `b' = -eps_b a d`
    with `d = 1 - (2 + 1/mu) y + y²/mu`.

    Args:
        label:
            The label of the neuron.

        x:
            The initial membrane potential.

        a:
            The initial gain.

        b:
            The initial threshold.

        mu:
            The target mean firing rate.

        eps_a:
            The adaptation rate of the gain.

        eps_b:
            The adaptation rate of the threshold.

        Gamma:
            The leak rate of the membrane potential.

        I0:
            The mean input.

        A:
            The amplitude of the periodic input.

        omega:
            The angular frequency of the periodic input.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import PolyhomeostaticNeuron
        >>> PolyhomeostaticNeuron('neuron').mu
        0.2
    """

    x: float = state(default=0.0)
    a: float = state(default=1.0, domain=Domain.POSITIVE)
    b: float = state(default=0.0)
    mu: float = parameter(default=0.2, domain=Domain.POSITIVE)
    eps_a: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    eps_b: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    Gamma: float = parameter(default=1.0, domain=Domain.POSITIVE)
    I0: float = parameter(default=0.0)
    A: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    omega: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the leaky membrane and the adaptation of gain and threshold.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        rate = 1 / (1 + exp(s.a * (s.b - s.x)))
        gradient = 1 - (2 + 1 / s.mu) * rate + rate**2 / s.mu
        return [
            Rate(s.x, s.Gamma * (s.I0 + s.A * sin(s.omega * s.t) - s.x)),
            Rate(s.a, s.eps_a * (1 / s.a + (s.x - s.b) * gradient)),
            Rate(s.b, -s.eps_b * s.a * gradient),
        ]


@register_model(
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
    """Return the Wilson-Cowan model of interacting excitatory and inhibitory populations of neurons.

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
            The model, with the activities `cortex.E` and `cortex.I`.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import wilson_cowan
        >>> wilson_cowan(P=0.0).elements[0].P
        0.0
    """
    return System(components=[WilsonCowan('cortex', E=E, I=I, P=P, Q=Q, c1=c1, c2=c2, c3=c3, c4=c4)])


@register_model(
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
            The model, with the phases `neurons.theta` as its only state.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import theta_neurons
        >>> theta_neurons(20).elements[0].size
        20
    """
    phases = np.random.default_rng(random_state).uniform(0.0, 2 * math.pi, n_neurons)
    neurons = ThetaNeuron.group(
        'neurons',
        n_neurons,
        theta=phases,
        eta=np.asarray(_lorentzian_quantiles(eta, delta, n_neurons)),
    )
    return System(components=[neurons], interactions=[PulseCoupling('coupling', neurons='neurons', kappa=kappa, n=n)])


@register_model(
    field='biology/neuroscience',
    tags=('neural network', 'competition', 'winner-take-all'),
    references=('Fukai and Tanaka 1997', 'Gros 2015'),
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
            The model, with the activities `unit_<i>.x`.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import winners_take_all
        >>> len(winners_take_all((1.0, 2.0, 3.0)).interactions)
        1
    """
    units = [RateUnit(f'unit_{index}', x=x, I=value, tau=tau) for index, value in enumerate(inputs)]
    inhibition = MutualInhibition('inhibition', units=[unit.label for unit in units], beta=beta)
    return System(components=units, interactions=[inhibition])


@register_model(
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
            The model, with the potential `neuron.x`, the gain `neuron.a`, and the threshold `neuron.b`.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import polyhomeostatic_neuron
        >>> polyhomeostatic_neuron(mu=0.1).elements[0].mu
        0.1
    """
    neuron = PolyhomeostaticNeuron('neuron', mu=mu, eps_a=eps_a, eps_b=eps_b, I0=I0, A=A, omega=omega)
    return System(components=[neuron])
