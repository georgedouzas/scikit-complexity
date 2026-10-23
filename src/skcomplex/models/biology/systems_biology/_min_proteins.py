"""Min-protein patterns: the skeleton model of MinD and MinE cycling between the membrane and the cytosol."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
from scipy.optimize import brentq

from ....core.base import Domain, parameter, state
from ....core.components import Element, Field
from ....core.laws import Law, Rate, Symbols, laplacian
from ....core.runtime import register_model
from ....core.solvers import ScipyODE
from ....core.space import LatticeSpace
from ....core.system import System


class MinSkeleton(Element):
    """Site of the Min skeleton model, with membrane MinD and MinDE and cytosolic MinD-ATP, MinD-ADP, and MinE.

    The reactions are the attachment and self-recruitment of MinD-ATP to the membrane, `(k_d + k_dd md) cdt`, the
    recruitment of MinE by membrane MinD, `k_de md ce`, the dissociation of MinDE complexes, `k_off mde`, which releases
    MinD-ADP and MinE to the cytosol, and the nucleotide exchange `lam cdd` from MinD-ADP to MinD-ATP. They conserve
    the totals of MinD, `md + mde + cdt + cdd`, and of MinE, `mde + ce`.

    Args:
        label:
            The label of the site or the field.

        md:
            The initial membrane MinD.

        mde:
            The initial membrane MinDE complexes.

        cdt:
            The initial cytosolic MinD-ATP.

        cdd:
            The initial cytosolic MinD-ADP.

        ce:
            The initial cytosolic MinE.

        Dm:
            The diffusivity on the membrane.

        Dc:
            The diffusivity in the cytosol.

        k_d:
            The attachment rate of MinD-ATP.

        k_dd:
            The recruitment rate of MinD-ATP by membrane MinD.

        k_de:
            The recruitment rate of MinE by membrane MinD.

        k_off:
            The dissociation rate of MinDE complexes.

        lam:
            The nucleotide exchange rate.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import MinSkeleton
        >>> MinSkeleton('site').lam
        6.0
    """

    md: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    mde: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    cdt: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    cdd: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    ce: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    Dm: float = parameter(default=0.013, domain=Domain.POSITIVE)
    Dc: float = parameter(default=60.0, domain=Domain.POSITIVE)
    k_d: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    k_dd: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    k_de: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    k_off: float = parameter(default=0.4, domain=Domain.NONNEGATIVE)
    lam: float = parameter(default=6.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the reactions and the diffusion of the five species.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        attach: Any = (s.k_d + s.k_dd * s.md) * s.cdt
        recruit = s.k_de * s.md * s.ce
        release = s.k_off * s.mde
        exchange = s.lam * s.cdd
        return [
            Rate(s.md, s.Dm * laplacian(s.md) + attach - recruit),
            Rate(s.mde, s.Dm * laplacian(s.mde) + recruit - release),
            Rate(s.cdt, s.Dc * laplacian(s.cdt) - attach + exchange),
            Rate(s.cdd, s.Dc * laplacian(s.cdd) + release - exchange),
            Rate(s.ce, s.Dc * laplacian(s.ce) - recruit + release),
        ]


def _uniform_state(element: MinSkeleton, total_D: float, total_E: float) -> dict[str, float]:
    """Return the uniform steady state of the skeleton model with the totals of MinD and MinE."""

    def others(md: float) -> tuple[float, float, float, float]:
        mde = total_E * element.k_de * md / (element.k_off + element.k_de * md)
        return (
            mde,
            element.k_off * mde / (element.k_d + element.k_dd * md),
            element.k_off * mde / element.lam,
            total_E - mde,
        )

    md = brentq(lambda md: md + sum(others(md)[:3]) - total_D, 0.0, total_D)
    mde, cdt, cdd, ce = others(md)
    return {'md': md, 'mde': mde, 'cdt': cdt, 'cdd': cdd, 'ce': ce}


@register_model(
    field='biology/systems_biology',
    tags=('field', 'min proteins', 'mass conserving', 'reaction-diffusion', 'traveling waves', 'pde'),
    references=('Brauns and Marchetti 2024', 'Halatek and Frey 2018', 'Huang, Meir, and Wingreen 2003'),
    defaults={'until': 400.0, 'dt': 4.0, 'solver': ScipyODE(method='BDF', jacobian='exact')},
)
def min_protein_skeleton(
    n_sites: int = 128,
    *,
    length: float = 100.0,
    total_D: float = 200.0,
    total_E: float = 120.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the skeleton model of the Min proteins on a periodic line, a mass-conserving reaction-diffusion system.

    MinD and MinE cycle between the membrane, where they diffuse slowly, and the cytosol, where they diffuse fast,
    with the reactions of `MinSkeleton`. The reactions conserve the totals of MinD and MinE, so only diffusion moves
    mass, and the local reactive equilibria map the dynamics onto conserved fields with nonreciprocal cross-diffusion
    (Brauns and Marchetti, App. F). With the default totals, the uniform state is unstable to an oscillatory band of
    wavelengths near 17, and narrow bands of membrane MinD travel along the line and meet head-on.
    The fast cytosolic diffusion and the fast recruitment make the system stiff, so the catalog registers the implicit
    `BDF` method with the exact Jacobian.

    Args:
        n_sites:
            The number of sites of the periodic line.

        length:
            The length of the line.

        total_D:
            The mean total density of MinD.

        total_E:
            The mean total density of MinE.

        random_state:
            The seed or generator of the 1% noise that moves MinD between the membrane and the cytosol.

    Returns:
        system:
            The model, with the membrane fields `cell.md` and `cell.mde` and the cytosolic fields `cell.cdt`,
            `cell.cdd`, and `cell.ce`.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import min_protein_skeleton
        >>> values = min_protein_skeleton(32).elements[0].values
        >>> round(float(values['ce'].mean() + values['mde'].mean()), 6)
        120.0
    """
    rng = np.random.default_rng(random_state)
    uniform = _uniform_state(MinSkeleton('site'), total_D, total_E)
    values: dict[str, Any] = {name: np.full(n_sites, value) for name, value in uniform.items()}
    noise = 0.01 * uniform['md'] * rng.uniform(-1.0, 1.0, n_sites)
    values['md'] = values['md'] + noise
    values['cdt'] = values['cdt'] - noise
    space = LatticeSpace(n_sites, spacing=length / n_sites)
    return System(components=[Field('cell', MinSkeleton, space, values=values)])
