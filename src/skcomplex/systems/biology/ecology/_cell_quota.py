"""Ecology systems: phytoplankton growth limited by its internal nutrient quota."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class DroopCellQuotaSystem(DynamicalSystem):
    """The Droop system of a phytoplankton that grows on its stored nutrient.

    The growth rate of the phytoplankton follows the nutrient stored in its cells.

    The growth rate `r_inf (1 - q / Q)` rises with the quota `Q` and saturates at `r_inf`. Under a constant external
    nutrient the quota settles at `Q* = q + v / r_inf`, where `v = vmax R / (g + R)` is the uptake, and the biomass then
    grows exponentially at `r_inf (1 - q / Q*) - d`. The book gives no values for this system, so the defaults are in
    units of the minimum quota, the maximum growth rate, and the half saturation, with time in days. The system has the
    quota `algae.Q` and the biomass `algae.P`.

    Args:
        Q:
            The initial cell quota.

        P:
            The initial biomass.

        r_inf:
            The growth rate for an infinite quota.

        q:
            The minimum quota, at which growth stops.

        vmax:
            The maximum uptake rate.

        g:
            The external nutrient at which the uptake is half its maximum.

        R:
            The external nutrient concentration.

        d:
            The death rate.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import DroopCellQuotaSystem
        >>> DroopCellQuotaSystem(R=4.0).prepare().system_.elements[0].get_value('R')
        4.0
    """

    field = 'biology/ecology'
    topics = ('population growth', 'ode', 'nutrient', 'cell quota', 'phytoplankton')
    references = ('Padmanabhan, Meskin, and Al Moustafa 2021', 'Droop 1973')
    until = 20.0
    dt = 0.1

    def __init__(
        self,
        Q: float = 1.5,
        P: float = 1.0,
        *,
        r_inf: float = 1.0,
        q: float = 1.0,
        vmax: float = 2.0,
        g: float = 1.0,
        R: float = 1.0,
        d: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.Q = Q
        self.P = P
        self.r_inf = r_inf
        self.q = q
        self.vmax = vmax
        self.g = g
        self.R = R
        self.d = d

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'algae'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'Q': DeclaredVariableSpec(self.Q, StateSpec(domain=RealSet.open(0, oo))),
            'P': DeclaredVariableSpec(self.P, StateSpec(domain=RealSet.closed_open(0, oo))),
            'r_inf': DeclaredVariableSpec(self.r_inf, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'q': DeclaredVariableSpec(self.q, ParameterSpec(domain=RealSet.open(0, oo))),
            'vmax': DeclaredVariableSpec(self.vmax, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'g': DeclaredVariableSpec(self.g, ParameterSpec(domain=RealSet.open(0, oo))),
            'R': DeclaredVariableSpec(self.R, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'd': DeclaredVariableSpec(self.d, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the uptake and dilution of the quota and the growth and death of biomass."""
        return {
            'Q': lambda s: s.target.vmax * s.target.R / (s.target.g + s.target.R) - self._get_growth(s) * s.target.Q,
            'P': lambda s: (self._get_growth(s) - s.target.d) * s.target.P,
        }

    def _get_growth(self, s: SimpleNamespace) -> Expression:
        """Return `growth` from the namespace of an equation."""
        return s.target.r_inf * (1 - s.target.q / s.target.Q)
