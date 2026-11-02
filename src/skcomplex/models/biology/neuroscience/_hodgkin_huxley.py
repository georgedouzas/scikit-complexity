"""Squid giant axon membrane of Hodgkin and Huxley, with sodium, potassium, and leak currents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class HodgkinHuxleyElement(Element):
    """Squid giant axon membrane of Hodgkin and Huxley, with sodium, potassium, and leak currents.

    The potential follows `C v' = I - g_Na m³ h (v - E_Na) - g_K n⁴ (v - E_K) - g_L (v - E_L)`, and each gate `x` of
    `n`, `m`, and `h` relaxes as `x' = alpha_x(v) (1 - x) - beta_x(v) x`, in the modern convention with rest near -65
    mV. Upward crossings of 0 mV are recorded as `spike` events. Units are mV, ms, mS/cm², and µA/cm². The element holds
    the states and parameters and has no laws of its own: the current, gating, and crossing interactions of
    `hodgkin_huxley` read its parameters.

    Args:
        label:
            The label of the neuron.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.biology.neuroscience import HodgkinHuxleyElement
        >>> HodgkinHuxleyElement('axon', I=10.0).g_na
        120.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec()] = -65.0,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.3177,
        m: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0529,
        h: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5961,
        C: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g_na: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 120.0,
        g_k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 36.0,
        g_l: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        e_na: Annotated[float, ParameterSpec()] = 50.0,
        e_k: Annotated[float, ParameterSpec()] = -77.0,
        e_l: Annotated[float, ParameterSpec()] = -54.387,
        I: Annotated[float, ParameterSpec()] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
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
