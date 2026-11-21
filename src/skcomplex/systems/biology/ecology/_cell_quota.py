"""Ecology systems: phytoplankton growth limited by its internal nutrient quota."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.ecology import DroopCellQuotaSystem
        >>> DroopCellQuotaSystem(R=4.0).prepare().system_.elements[0].values['R']
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
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

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'Q': self.Q,
            'P': self.P,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'r_inf': self.r_inf,
            'q': self.q,
            'vmax': self.vmax,
            'g': self.g,
            'R': self.R,
            'd': self.d,
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
