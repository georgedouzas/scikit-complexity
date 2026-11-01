"""Excitatory and inhibitory populations of Wilson and Cowan, as fractions of active cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class WilsonCowanElement(Element):
    """Excitatory and inhibitory populations of Wilson and Cowan, as fractions of active cells.

    The excitatory activity follows `tau_e E' = -E + (k_e - r_e E) S_e(c1 E - c2 I + P)` and the inhibitory activity
    `tau_i I' = -I + (k_i - r_i I) S_i(c3 E - c4 I + Q)`, where `S(x) = 1 / (1 + exp(-a (x - theta))) - 1 / (1 +
    exp(a theta))` is a sigmoid shifted to vanish at zero input and `k` is its largest value. The element holds the
    states and parameters and has no laws of its own: the relaxations of `wilson_cowan` read its parameters.

    Args:
        label:
            The label of the populations.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.biology.neuroscience import WilsonCowanElement
        >>> WilsonCowanElement('cortex', P=1.25).c1
        16.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        I: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 16.0,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 12.0,
        c3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
        c4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        a_e: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.3,
        theta_e: Annotated[float, ParameterSpec()] = 4.0,
        a_i: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        theta_i: Annotated[float, ParameterSpec()] = 3.7,
        r_e: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        r_i: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        P: Annotated[float, ParameterSpec()] = 1.25,
        Q: Annotated[float, ParameterSpec()] = 0.0,
        tau_e: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        tau_i: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
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
