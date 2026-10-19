"""Ecology models: phytoplankton growth limited by its internal nutrient quota."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import Element, Law, Rate, Symbols, System
from ....core.runtime import register_model


class QuotaPhytoplankton(Element):
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
        >>> from skcomplex.models.biology.ecology import QuotaPhytoplankton
        >>> QuotaPhytoplankton('algae', R=2.0).R
        2.0
    """

    Q: float = state(default=1.5, domain=Domain.POSITIVE)
    P: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    r_inf: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    q: float = parameter(default=1.0, domain=Domain.POSITIVE)
    vmax: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    g: float = parameter(default=1.0, domain=Domain.POSITIVE)
    R: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    d: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
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
            Rate(s.Q, s.vmax * s.R / (s.g + s.R) - growth * s.Q),
            Rate(s.P, (growth - s.d) * s.P),
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
    algae = QuotaPhytoplankton('algae', Q=Q, P=P, r_inf=r_inf, q=q, vmax=vmax, g=g, R=R, d=d)
    return System(elements=[algae])
