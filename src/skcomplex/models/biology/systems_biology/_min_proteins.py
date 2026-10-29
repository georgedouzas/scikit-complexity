"""Min-protein patterns: the skeleton model of MinD and MinE cycling between the membrane and the cytosol."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from scipy.optimize import brentq

from ....core.elements import Element, FieldElement
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import laplacian
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.solvers import ScipyODESolver
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model


class MinSkeletonElement(Element):
    """Site of the Min skeleton model, with membrane MinD and MinDE and cytosolic MinD-ATP, MinD-ADP, and MinE.

    The reactions are the attachment and self-recruitment of MinD-ATP to the membrane, `(k_d + k_dd md) cdt`, the
    recruitment of MinE by membrane MinD, `k_de md ce`, the dissociation of MinDE complexes, `k_off mde`, which releases
    MinD-ADP and MinE to the cytosol, and the nucleotide exchange `lam cdd` from MinD-ADP to MinD-ATP. They conserve
    the totals of MinD, `md + mde + cdt + cdd`, and of MinE, `mde + ce`.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.biology.systems_biology import MinSkeletonElement
        >>> MinSkeletonElement('site').lam
        6.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        md: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        mde: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        cdt: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        cdd: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        ce: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Dm: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.013,
        Dc: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 60.0,
        k_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        k_dd: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        k_de: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        k_off: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 6.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.md = md
        self.mde = mde
        self.cdt = cdt
        self.cdd = cdd
        self.ce = ce
        self.Dm = Dm
        self.Dc = Dc
        self.k_d = k_d
        self.k_dd = k_dd
        self.k_de = k_de
        self.k_off = k_off
        self.lam = lam

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the reactions and the diffusion of the five species.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        attach = (s.k_d + s.k_dd * s.md) * s.cdt
        recruit = s.k_de * s.md * s.ce
        release = s.k_off * s.mde
        exchange = s.lam * s.cdd
        return [
            RateLaw(s.md, s.Dm * laplacian(s.md) + attach - recruit),
            RateLaw(s.mde, s.Dm * laplacian(s.mde) + recruit - release),
            RateLaw(s.cdt, s.Dc * laplacian(s.cdt) - attach + exchange),
            RateLaw(s.cdd, s.Dc * laplacian(s.cdd) + release - exchange),
            RateLaw(s.ce, s.Dc * laplacian(s.ce) - recruit + release),
        ]


def _uniform_state(element: MinSkeletonElement, total_D: float, total_E: float) -> dict[str, float]:
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
    defaults={'until': 400.0, 'dt': 4.0, 'solver': ScipyODESolver(method='BDF', jacobian='exact')},
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
    with the reactions of `MinSkeletonElement`. The reactions conserve the totals of MinD and MinE, so only diffusion
    moves mass, and the local reactive equilibria map the dynamics onto conserved fields with nonreciprocal
    cross-diffusion (Brauns and Marchetti, App. F). With the default totals, the uniform state is unstable to an
    oscillatory band of wavelengths near 17, and narrow bands of membrane MinD travel along the line and meet head-on.
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
    uniform = _uniform_state(MinSkeletonElement('site'), total_D, total_E)
    values: dict[str, Array] = {name: np.full(n_sites, value) for name, value in uniform.items()}
    noise = 0.01 * uniform['md'] * rng.uniform(-1.0, 1.0, n_sites)
    values['md'] = values['md'] + noise
    values['cdt'] = values['cdt'] - noise
    space = LatticeSpace(n_sites, spacing=length / n_sites)
    return System(elements=[FieldElement('cell', element=MinSkeletonElement, space=space, values=values)])
