"""Neuroscience systems: firing rates, phase neurons, and adaptive neurons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, cos, oo, sin
from sage.sets.real_set import RealSet
from scipy import stats

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import logistic, mean
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ..common.excitable import ThetaNeuronElement, ThetaNeuronInteraction
from ._leaky_integration import LeakyIntegrationInteraction


class PolyhomeostaticNeuronElement(Element):
    """Rate neuron that adapts its gain and threshold.

    The adaptation makes its firing rate exponentially distributed.

    The membrane potential `x` integrates the input `I0 + A sin(omega t)` with rate `Gamma`, and the neuron fires at
    rate `y = 1 / (1 + exp(a (b - x)))`. Stochastic gradient descent on the Kullback-Leibler divergence between the
    distribution of `y` and an exponential of mean `mu` gives `a' = eps_a (1/a + (x - b) d)` and `b' = -eps_b a d`
    with `d = 1 - (2 + 1/mu) y + y²/mu`. The element holds the states and parameters and has no laws of its own: the
    relaxation and plasticity interactions of `PolyhomeostaticNeuronSystem` read its parameters.

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
        >>> from skcomplex.systems.biology.neuroscience import PolyhomeostaticNeuronElement
        >>> PolyhomeostaticNeuronElement('neuron').mu
        0.2
    """

    def __init__(
        self,
        label: str,
        *,
        x: float = 0.0,
        a: Annotated[float, RealSet.open(0, oo)] = 1.0,
        b: float = 0.0,
        mu: Annotated[float, RealSet.open(0, oo)] = 0.2,
        eps_a: Annotated[float, RealSet.closed_open(0, oo)] = 0.05,
        eps_b: Annotated[float, RealSet.closed_open(0, oo)] = 0.05,
        Gamma: Annotated[float, RealSet.open(0, oo)] = 1.0,
        I0: float = 0.0,
        A: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        omega: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.a = a
        self.b = b
        self.mu = mu
        self.eps_a = eps_a
        self.eps_b = eps_b
        self.Gamma = Gamma
        self.I0 = I0
        self.A = A
        self.omega = omega


class RateUnitElement(Element):
    """Neural unit of a competitive network, whose activity grows with its input and saturates.

    A `SaturatingGrowthInteraction` grows it as `tau x' = x (I - x)`.

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
        >>> from skcomplex.systems.biology.neuroscience import RateUnitElement
        >>> RateUnitElement('unit', I=1.2).I
        1.2
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        I: float = 1.0,
        tau: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.I = I
        self.tau = tau


class SaturatingGrowthInteraction(Interaction):
    """Self-limited growth of the activity of a neural unit, `tau x' = x (I - x)`.

    The input `I` and the time constant `tau` are values of the unit, a `RateUnitElement`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the unit.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import SaturatingGrowthInteraction
        >>> SaturatingGrowthInteraction('growth', target='unit').target
        'unit'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the self-limited growth of the activity.

        Args:
            context:
                The namespace of the unit.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            self.laws_: list[BaseLaw] = [RateLaw(s.x, s.x * (s.I - s.x) / s.tau)]


class WilsonCowanElement(Element):
    """Excitatory and inhibitory populations of Wilson and Cowan.

    The populations are fractions of active cells.

    The excitatory activity follows `tau_e E' = -E + (k_e - r_e E) S_e(c1 E - c2 I + P)` and the inhibitory activity
    `tau_i I' = -I + (k_i - r_i I) S_i(c3 E - c4 I + Q)`, where `S(x) = 1 / (1 + exp(-a (x - theta))) - 1 / (1 +
    exp(a theta))` is a sigmoid shifted to vanish at zero input and `k` is its largest value. The element holds the
    states and parameters and has no laws of its own: the relaxations of `WilsonCowanSystem` read its parameters.

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
        >>> from skcomplex.systems.biology.neuroscience import WilsonCowanElement
        >>> WilsonCowanElement('cortex', P=1.25).c1
        16.0
    """

    def __init__(
        self,
        label: str,
        *,
        E: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        I: Annotated[float, RealSet.closed_open(0, oo)] = 0.05,
        c1: Annotated[float, RealSet.closed_open(0, oo)] = 16.0,
        c2: Annotated[float, RealSet.closed_open(0, oo)] = 12.0,
        c3: Annotated[float, RealSet.closed_open(0, oo)] = 15.0,
        c4: Annotated[float, RealSet.closed_open(0, oo)] = 3.0,
        a_e: Annotated[float, RealSet.open(0, oo)] = 1.3,
        theta_e: float = 4.0,
        a_i: Annotated[float, RealSet.open(0, oo)] = 2.0,
        theta_i: float = 3.7,
        r_e: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        r_i: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        P: float = 1.25,
        Q: float = 0.0,
        tau_e: Annotated[float, RealSet.open(0, oo)] = 1.0,
        tau_i: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.E = E
        self.I = I
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.a_e = a_e
        self.theta_e = theta_e
        self.a_i = a_i
        self.theta_i = theta_i
        self.r_e = r_e
        self.r_i = r_i
        self.P = P
        self.Q = Q
        self.tau_e = tau_e
        self.tau_i = tau_i


class MutualInhibitionInteraction(Interaction):
    """Inhibition of every unit by the summed activity of all the other units.

    The inhibition is `-beta x_i sum_{j != i} x_j / tau_i`.

    It is the coupling of separate units to each other, with the term `-beta x_i x_j / tau_i` of every other unit `j`.

    Args:
        label:
            The label of the interaction.

        units:
            The paths of the units.

        beta:
            The strength of the inhibition.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import MutualInhibitionInteraction
        >>> MutualInhibitionInteraction('inhibition', units=['a', 'b'], beta=2.0).beta
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        units: tuple[Role, ...],
        beta: Annotated[float, RealSet.closed_open(0, oo)] = 2.0,
    ) -> None:
        super().__init__(label)
        self.units = units
        self.beta = beta

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the inhibition of every unit by the other units, term by term.

        Args:
            context:
                The namespace of the units.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        laws: list[BaseLaw] = []
        for unit in context.units:
            parts = [
                SR(1) * (-context.beta * unit.x * other.x / unit.tau) for other in context.units if other is not unit
            ]
            laws.append(RateLaw(unit.x, sum(parts[1:], parts[0])))
        self.laws_ = laws


class PulseCouplingInteraction(Interaction):
    """Global coupling of theta neurons through the mean of their pulses.

    Every neuron emits smooth pulses.

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
        >>> from skcomplex.systems.biology.neuroscience import PulseCouplingInteraction
        >>> PulseCouplingInteraction('coupling', neurons='neurons', kappa=2.0).kappa
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        neurons: Role,
        kappa: float = 1.0,
        n: int = 2,
    ) -> None:
        super().__init__(label)
        self.neurons = neurons
        self.kappa = kappa
        self.n = n

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the current of the mean pulse into every neuron.

        Args:
            context:
                The namespace of the symbols the laws can use.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        s = context
        theta = s.neurons.theta
        normalization = 2**self.n * math.factorial(self.n) ** 2 / math.factorial(2 * self.n)
        pulses = mean(normalization * (1 - cos(theta)) ** self.n)
        self.laws_: list[BaseLaw] = [RateLaw(theta, (1 + cos(theta)) * s.kappa * pulses)]


class WilsonCowanSystem(BaseSystem):
    """The Wilson-Cowan system of interacting excitatory and inhibitory populations.

    The populations are made of neurons.

    With the published couplings, a moderate excitatory input `P = 1.25` gives a limit cycle of population activity, and
    no input lets the activity decay to rest. The system has the activities `cortex.E` and `cortex.I`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import WilsonCowanSystem
        >>> WilsonCowanSystem(P=0.0).prepare().system_.elements[0].P
        0.0
    """

    field = 'biology/neuroscience'
    topics = ('neural mass', 'oscillations', 'excitation-inhibition')
    references = ('Wilson and Cowan 1972', 'arXiv:2112.12147')
    until = 100.0
    dt = 0.2

    def __init__(
        self,
        E: float = 0.1,
        I: float = 0.05,
        *,
        P: float = 1.25,
        Q: float = 0.0,
        c1: float = 16.0,
        c2: float = 12.0,
        c3: float = 15.0,
        c4: float = 3.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.E = E
        self.I = I
        self.P = P
        self.Q = Q
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4

    @staticmethod
    def _get_response(x: Expression, gain: Expression, threshold: Expression) -> Expression:
        """Return the response `logistic(x) - logistic(0)` of a sigmoid.

        The sigmoid has a gain and a threshold, and the response is 0 at no input.
        """
        response: Expression = logistic(x, threshold, 1 / gain) - logistic(0, threshold, 1 / gain)
        return response

    @staticmethod
    def _get_saturation(gain: Expression, threshold: Expression) -> Expression:
        """Return the largest value of the response of a sigmoid of a gain and a threshold."""
        saturation: Expression = 1 - logistic(0, threshold, 1 / gain)
        return saturation

    def _get_excitation(self, s: SimpleNamespace) -> Expression:
        """Return the response of the Wilson-Cowan excitatory population to its input."""
        c = s.target
        excitation: Expression = (self._get_saturation(c.a_e, c.theta_e) - c.r_e * c.E) * self._get_response(
            c.c1 * c.E - c.c2 * c.I + c.P,
            c.a_e,
            c.theta_e,
        )
        return excitation

    def _get_inhibition(self, s: SimpleNamespace) -> Expression:
        """Return the response of the Wilson-Cowan inhibitory population to its input."""
        c = s.target
        inhibition: Expression = (self._get_saturation(c.a_i, c.theta_i) - c.r_i * c.I) * self._get_response(
            c.c3 * c.E - c.c4 * c.I + c.Q,
            c.a_i,
            c.theta_i,
        )
        return inhibition

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the Wilson-Cowan system of interacting excitatory and inhibitory populations.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        cortex = WilsonCowanElement(
            'cortex',
            E=self.E,
            I=self.I,
            P=self.P,
            Q=self.Q,
            c1=self.c1,
            c2=self.c2,
            c3=self.c3,
            c4=self.c4,
        )
        relaxations = [
            LeakyIntegrationInteraction(
                'excitatory_relaxation',
                target='cortex',
                state='E',
                goal=self._get_excitation,
                time=lambda s: s.target.tau_e,
            ),
            LeakyIntegrationInteraction(
                'inhibitory_relaxation',
                target='cortex',
                state='I',
                goal=self._get_inhibition,
                time=lambda s: s.target.tau_i,
            ),
        ]
        self.system_ = System(self.label, elements=[cortex], interactions=relaxations)


class ThetaNeuronsSystem(BaseSystem):
    """A network of theta neurons coupled all to all.

    Theta neurons are equivalently quadratic integrate-and-fire neurons.

    Neuron `i` follows `theta_i' = (1 - cos theta_i) + (1 + cos theta_i)(eta_i + kappa I)`, where `I` is the mean pulse
    `a_n (1 - cos theta_j)^n` of all neurons at the current state. The excitabilities are evenly spaced quantiles of a
    Lorentzian distribution of center `eta` and half-width `delta`, the distribution for which Ott and Antonsen's ansatz
    reduces the network exactly to one complex equation. Strong excitatory coupling switches the network from a quiet
    state to collective firing. The system has the phases `neurons.theta` as its only state.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import ThetaNeuronsSystem
        >>> ThetaNeuronsSystem(20).prepare().system_.elements[0].size
        20
    """

    field = 'biology/neuroscience'
    topics = ('neuron', 'phase oscillators', 'mean field', 'network')
    references = ('Ermentrout and Kopell 1986', 'Luke, Barreto, and So 2013', 'arXiv:1902.05307')
    until = 20.0
    dt = 0.1

    def __init__(
        self,
        n_neurons: int = 50,
        *,
        eta: float = -0.2,
        delta: float = 0.1,
        kappa: float = 1.5,
        n: int = 2,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_neurons = n_neurons
        self.eta = eta
        self.delta = delta
        self.kappa = kappa
        self.n = n
        self.random_state = random_state

    @staticmethod
    def _get_lorentzian_quantiles(center: float, width: float, size: int) -> list[float]:
        """Return evenly spaced quantiles of a Lorentzian distribution."""
        levels = (np.arange(size) + 0.5) / size
        return [float(value) for value in center + width * stats.cauchy.ppf(levels)]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a network of theta neurons.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        phases = np.random.default_rng(self.random_state).uniform(0.0, 2 * math.pi, self.n_neurons)
        neurons = ThetaNeuronElement(
            'neurons',
            theta=phases,
            eta=np.asarray(self._get_lorentzian_quantiles(self.eta, self.delta, self.n_neurons)),
            size=self.n_neurons,
        )
        self.system_ = System(
            self.label,
            elements=[neurons],
            interactions=[
                ThetaNeuronInteraction('phases', target='neurons'),
                PulseCouplingInteraction('coupling', neurons='neurons', kappa=self.kappa, n=self.n),
            ],
        )


class WinnersTakeAllSystem(BaseSystem):
    """A winner-take-all network of units.

    The units have self-limited growth and mutual inhibition.

    Unit `i` follows `tau x_i' = x_i (I_i - x_i - beta sum_{j != i} x_j)`, the competitive network of Fukai and Tanaka.
    One inhibition interaction binds every unit as an element set and reads the total activity of the set, so the
    network holds one interaction whatever the number of units. When the inhibition `beta` exceeds 1, only the unit with
    the largest input stays active, at activity `I_max`. When it is below 1, several winners share the activity. The
    winners-take-all network of Gros (2015, section 10.4), with reservoir variables and clique encoding, is a different
    model. The system has the activities `unit_<i>.x`.

    Args:
        inputs:
            The input of every unit.

        beta:
            The strength of the mutual inhibition.

        tau:
            The time constant of the units.

        x:
            The initial activity of every unit.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import WinnersTakeAllSystem
        >>> [interaction.label for interaction in WinnersTakeAllSystem((1.0, 2.0)).prepare().system_.interactions]
        ['growth_unit_0', 'growth_unit_1', 'inhibition']
    """

    field = 'biology/neuroscience'
    topics = ('neural network', 'competition', 'winner-take-all')
    references = ('Fukai and Tanaka 1997',)
    until = 40.0
    dt = 0.2

    def __init__(
        self,
        inputs: Sequence[float] = (1.0, 1.1, 0.9, 1.05, 0.95),
        *,
        beta: float = 2.0,
        tau: float = 1.0,
        x: float = 0.1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.inputs = inputs
        self.beta = beta
        self.tau = tau
        self.x = x

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a winner-take-all network of units.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        units = [
            RateUnitElement(f'unit_{index}', x=self.x, I=value, tau=self.tau) for index, value in enumerate(self.inputs)
        ]
        growth = [SaturatingGrowthInteraction(f'growth_{unit.label}', target=unit.label) for unit in units]
        inhibition = MutualInhibitionInteraction(
            'inhibition',
            units=tuple(unit.label for unit in units),
            beta=self.beta,
        )
        self.system_ = System(self.label, elements=units, interactions=[*growth, inhibition])


class PolyhomeostaticNeuronSystem(BaseSystem):
    """A neuron that adapts its gain and threshold.

    The adaptation makes its firing rate approach an exponential distribution.

    Polyhomeostatic optimization targets the whole distribution of the firing rate, the one of maximal entropy for a
    given mean `mu`, rather than its mean alone. The neuron is driven by a periodic input, and the time average of its
    rate approaches `mu` as gain and threshold adapt. The system has the potential `neuron.x`, the gain `neuron.a`, and
    the threshold `neuron.b`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import PolyhomeostaticNeuronSystem
        >>> PolyhomeostaticNeuronSystem(mu=0.1).prepare().system_.elements[0].mu
        0.1
    """

    field = 'biology/neuroscience'
    topics = ('neuron', 'homeostasis', 'adaptation', 'information')
    references = ('Triesch 2005', 'Markovic and Gros 2010', 'Gros 2015')
    until = 400.0
    dt = 1.0

    def __init__(
        self,
        *,
        mu: float = 0.2,
        eps_a: float = 0.05,
        eps_b: float = 0.05,
        I0: float = 0.0,
        A: float = 1.0,
        omega: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.mu = mu
        self.eps_a = eps_a
        self.eps_b = eps_b
        self.I0 = I0
        self.A = A
        self.omega = omega

    @staticmethod
    def _get_kl_gradient(s: SimpleNamespace) -> Expression:
        """Return the factor `d = 1 - (2 + 1/mu) y + y²/mu` of the gradient.

        The gradient is the one of the polyhomeostatic neuron.
        """
        neuron = s.target
        rate = logistic(neuron.x, neuron.b, 1 / neuron.a)
        gradient: Expression = 1 - (2 + 1 / neuron.mu) * rate + rate**2 / neuron.mu
        return gradient

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a neuron.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        neuron = PolyhomeostaticNeuronElement(
            'neuron',
            mu=self.mu,
            eps_a=self.eps_a,
            eps_b=self.eps_b,
            I0=self.I0,
            A=self.A,
            omega=self.omega,
        )
        interactions = [
            LeakyIntegrationInteraction(
                'membrane',
                target='neuron',
                state='x',
                goal=lambda s: s.target.I0 + s.target.A * sin(s.target.omega * s.t),
                time=lambda s: 1 / s.target.Gamma,
            ),
            Interaction(
                'gain_plasticity',
                roles={'target': 'neuron'},
                laws=lambda s: [
                    RateLaw(
                        s.target.a,
                        s.target.eps_a * (1 / s.target.a + (s.target.x - s.target.b) * self._get_kl_gradient(s)),
                    ),
                ],
            ),
            Interaction(
                'threshold_plasticity',
                roles={'target': 'neuron'},
                laws=lambda s: [RateLaw(s.target.b, -s.target.eps_b * s.target.a * self._get_kl_gradient(s))],
            ),
        ]
        self.system_ = System(self.label, elements=[neuron], interactions=interactions)
