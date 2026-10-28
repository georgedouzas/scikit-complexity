"""Ecology models: phytoplankton growth limited by its internal nutrient quota."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.system import System
from ....engine.runtime import register_model
from ....specs import ParameterSpec, StateSpec, VisualSpec


class QuotaPhytoplanktonElement(Element):
    """Phytoplankton that grows on the nutrient stored in its cells, the Droop cell-quota model.

    The quota follows `dQ/dt = vmax R / (g + R) - r_inf (1 - q / Q) Q`, the Michaelis-Menten uptake from the
    external nutrient `R` less the dilution by growth, and the biomass `dP/dt = r_inf (1 - q / Q) P - d P`.

    Args:
        label:
            The label of the population.

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

    Examples:
        >>> from skcomplex.models.biology.ecology import QuotaPhytoplanktonElement
        >>> QuotaPhytoplanktonElement('algae', R=2.0).R
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Q: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.5,
        P: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        r_inf: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        q: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        vmax: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        R: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Q = Q
        self.P = P
        self.r_inf = r_inf
        self.q = q
        self.vmax = vmax
        self.g = g
        self.R = R
        self.d = d

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the uptake and dilution of the quota and the growth and death of the biomass.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        growth = s.r_inf * (1 - s.q / s.Q)
        return [
            RateLaw(s.Q, s.vmax * s.R / (s.g + s.R) - growth * s.Q),
            RateLaw(s.P, (growth - s.d) * s.P),
        ]


@register_model(
    field='biology/ecology',
    tags=('population growth', 'ode', 'nutrient', 'cell quota', 'phytoplankton'),
    references=('Padmanabhan, Meskin, and Al Moustafa 2021', 'Droop 1973'),
    defaults={'until': 20.0, 'dt': 0.1},
)
def droop_cell_quota(
    Q: float = 1.5,
    P: float = 1.0,
    *,
    r_inf: float = 1.0,
    q: float = 1.0,
    vmax: float = 2.0,
    g: float = 1.0,
    R: float = 1.0,
    d: float = 0.1,
) -> System:
    """Return the Droop model, a phytoplankton whose growth rate follows the nutrient stored in its cells.

    The growth rate `r_inf (1 - q / Q)` rises with the quota `Q` and saturates at `r_inf`. Under a constant external
    nutrient the quota settles at `Q* = q + v / r_inf`, where `v = vmax R / (g + R)` is the uptake, and the biomass
    then grows exponentially at `r_inf (1 - q / Q*) - d`. The book gives no values for this model, so the defaults are
    in units of the minimum quota, the maximum growth rate, and the half saturation, with time in days.

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

    Returns:
        system:
            The model, with the quota `algae.Q` and the biomass `algae.P`.

    Examples:
        >>> from skcomplex.models.biology.ecology import droop_cell_quota
        >>> droop_cell_quota(R=4.0).elements[0].R
        4.0
    """
    algae = QuotaPhytoplanktonElement('algae', Q=Q, P=P, r_inf=r_inf, q=q, vmax=vmax, g=g, R=R, d=d)
    return System(elements=[algae])
