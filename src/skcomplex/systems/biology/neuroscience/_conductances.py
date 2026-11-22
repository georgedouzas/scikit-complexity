"""Neuroscience systems: conductance-based and reduced excitable neurons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, EventLaw, RateLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ..common.excitable import FitzHughNagumoInteraction
from .types import GateRates


class FitzHughNagumoNeuronElement(Element):
    """FitzHugh-Nagumo neuron, with a fast potential `v` and a slow recovery `w`.

    The neuron has no laws of its own. FitzHugh-Nagumo kinetics change it.

    Args:
        label:
            The label of the neuron.

        v:
            The initial potential.

        w:
            The initial recovery.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import FitzHughNagumoNeuronElement
        >>> FitzHughNagumoNeuronElement('neuron').w
        -0.6
    """

    def __init__(
        self,
        label: str,
        *,
        v: float = -1.2,
        w: float = -0.6,
    ) -> None:
        super().__init__(label)
        self.v = v
        self.w = w


class HodgkinHuxleyElement(Element):
    """Squid giant axon membrane of Hodgkin and Huxley.

    The membrane has sodium, potassium, and leak currents.

    The potential follows `C v' = I - g_Na m³ h (v - E_Na) - g_K n⁴ (v - E_K) - g_L (v - E_L)`, and each gate `x` of
    `n`, `m`, and `h` relaxes as `x' = alpha_x(v) (1 - x) - beta_x(v) x`, in the modern convention with rest near -65
    mV. Upward crossings of 0 mV are recorded as `spike` events. Units are mV, ms, mS/cm², and µA/cm². The element holds
    the states and parameters and has no laws of its own: the current, gating, and crossing interactions of
    `HodgkinHuxleySystem` read its parameters.

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
        >>> from skcomplex.systems.biology.neuroscience import HodgkinHuxleyElement
        >>> HodgkinHuxleyElement('axon', I=10.0).g_na
        120.0
    """

    def __init__(
        self,
        label: str,
        *,
        v: float = -65.0,
        n: Annotated[float, RealSet.closed_open(0, oo)] = 0.3177,
        m: Annotated[float, RealSet.closed_open(0, oo)] = 0.0529,
        h: Annotated[float, RealSet.closed_open(0, oo)] = 0.5961,
        C: Annotated[float, RealSet.open(0, oo)] = 1.0,
        g_na: Annotated[float, RealSet.closed_open(0, oo)] = 120.0,
        g_k: Annotated[float, RealSet.closed_open(0, oo)] = 36.0,
        g_l: Annotated[float, RealSet.closed_open(0, oo)] = 0.3,
        e_na: float = 50.0,
        e_k: float = -77.0,
        e_l: float = -54.387,
        I: float = 10.0,
    ) -> None:
        super().__init__(label)
        self.v = v
        self.n = n
        self.m = m
        self.h = h
        self.C = C
        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l
        self.e_na = e_na
        self.e_k = e_k
        self.e_l = e_l
        self.I = I


class HodgkinHuxleySystem(BaseSystem):
    """The Hodgkin-Huxley neuron.

    Its sodium and potassium channels produce action potentials.

    The gates start at their steady state at the initial potential, and the current is switched on at the start. A
    constant current above about 6.2 µA/cm² makes the neuron fire periodically, and a weaker one gives at most a few
    spikes before it settles. The system has the potential `neuron.v`, the gates `neuron.n`, `neuron.m`, and `neuron.h`,
    and its spikes as `spike` events.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import HodgkinHuxleySystem
        >>> round(HodgkinHuxleySystem().prepare().system_.elements[0].h, 4)
        0.5961
    """

    field = 'biology/neuroscience'
    topics = ('neuron', 'excitability', 'spiking', 'conductance')
    references = ('Hodgkin and Huxley 1952', 'Rinzel and Miller 1980', 'Trzaska 2023', 'arXiv:2112.12147')
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        v: float = -65.0,
        *,
        I: float = 10.0,
        C: float = 1.0,
        g_na: float = 120.0,
        g_k: float = 36.0,
        g_l: float = 0.3,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.v = v
        self.I = I
        self.C = C
        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l

    @staticmethod
    def _get_gate_rates(v: Expression) -> GateRates:
        """Return the opening and closing rates per ms of the gates `n`, `m`, and `h`.

        The rates are at a potential in mV.
        """
        return {
            'n': (0.01 * (v + 55) / (1 - exp(-(v + 55) / 10)), 0.125 * exp(-(v + 65) / 80)),
            'm': (0.1 * (v + 40) / (1 - exp(-(v + 40) / 10)), 4 * exp(-(v + 65) / 18)),
            'h': (0.07 * exp(-(v + 65) / 20), 1 / (1 + exp(-(v + 35) / 10))),
        }

    def _get_gate_kinetics(self, gate: str) -> Callable[[SimpleNamespace], list[BaseLaw]]:
        """Return the rate `alpha(v) (1 - x) - beta(v) x` of a gate `x`.

        The rate is a function of the namespace of the neuron.
        """

        def kinetics(s: SimpleNamespace) -> list[BaseLaw]:
            opening, closing = self._get_gate_rates(s.target.v)[gate]
            x = getattr(s.target, gate)
            return [RateLaw(x, opening * (1 - x) - closing * x)]

        return kinetics

    def _get_resting_gates(self, v: float) -> dict[str, float]:
        """Return the steady-state opening of every gate at a fixed potential in mV."""
        return {
            gate: float(opening / (opening + closing)) for gate, (opening, closing) in self._get_gate_rates(v).items()
        }

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Hodgkin-Huxley neuron.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        gates = self._get_resting_gates(self.v)
        self.system_ = System(
            self.label,
            elements=[
                HodgkinHuxleyElement(
                    'neuron',
                    v=self.v,
                    I=self.I,
                    C=self.C,
                    g_na=self.g_na,
                    g_k=self.g_k,
                    g_l=self.g_l,
                    n=gates['n'],
                    m=gates['m'],
                    h=gates['h'],
                ),
            ],
            interactions=[
                Interaction(
                    'applied_current',
                    roles={'target': 'neuron'},
                    laws=lambda s: [RateLaw(s.target.v, s.target.I / s.target.C)],
                ),
                Interaction(
                    'sodium_current',
                    roles={'target': 'neuron'},
                    laws=lambda s: [
                        RateLaw(
                            s.target.v,
                            -s.target.g_na * s.target.m**3 * s.target.h * (s.target.v - s.target.e_na) / s.target.C,
                        ),
                    ],
                ),
                Interaction(
                    'potassium_current',
                    roles={'target': 'neuron'},
                    laws=lambda s: [
                        RateLaw(s.target.v, -s.target.g_k * s.target.n**4 * (s.target.v - s.target.e_k) / s.target.C),
                    ],
                ),
                Interaction(
                    'leak_current',
                    roles={'target': 'neuron'},
                    laws=lambda s: [RateLaw(s.target.v, -s.target.g_l * (s.target.v - s.target.e_l) / s.target.C)],
                ),
                *[
                    Interaction(f'{gate}_gating', roles={'target': 'neuron'}, laws=self._get_gate_kinetics(gate))
                    for gate in ('n', 'm', 'h')
                ],
                Interaction(
                    'spike',
                    roles={'target': 'neuron'},
                    laws=lambda s: [EventLaw(SR(s.target.v), None, 1, False, label='spike')],
                ),
            ],
        )


class FitzhughNagumoODESystem(BaseSystem):
    """The FitzHugh-Nagumo neuron.

    It is the two-variable reduction of Hodgkin and Huxley's excitable membrane.

    Without current the neuron rests on a stable fixed point and answers a large enough kick with one spike. A current
    `I` between about 0.33 and 1.42 destabilizes the rest and the neuron fires periodically. The system has the
    potential `neuron.v` and the recovery `neuron.w`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import FitzhughNagumoODESystem
        >>> FitzhughNagumoODESystem(I=0.0).prepare().system_.interactions[0].I
        0.0
    """

    field = 'biology/neuroscience'
    topics = ('neuron', 'excitability', 'relaxation oscillations')
    references = ('FitzHugh 1961', 'Nagumo, Arimoto, and Yoshizawa 1962', 'Trzaska 2023', 'Datseris and Parlitz 2022')
    until = 100.0
    dt = 0.2

    def __init__(
        self,
        v: float = -1.2,
        w: float = -0.6,
        *,
        a: float = 0.7,
        b: float = 0.8,
        epsilon: float = 0.08,
        I: float = 0.5,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.v = v
        self.w = w
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.I = I

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the FitzHugh-Nagumo neuron.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        kinetics = FitzHughNagumoInteraction(
            'kinetics',
            target='neuron',
            a=self.a,
            b=self.b,
            epsilon=self.epsilon,
            I=self.I,
        )
        self.system_ = System(
            self.label,
            elements=[FitzHughNagumoNeuronElement('neuron', v=self.v, w=self.w)],
            interactions=[kinetics],
        )
