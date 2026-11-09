"""Min-protein patterns.

The skeleton system has MinD and MinE cycling between the membrane and the cytosol.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from scipy.optimize import brentq

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import DiffusionInteraction, ReactionInteraction
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import ScipyODESolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array


class MinSkeletonElement(Element):
    """Site of the Min skeleton system.

    It holds membrane MinD and MinDE and cytosolic MinD-ATP, MinD-ADP, and MinE.

    The reactions are the attachment and self-recruitment of MinD-ATP to the membrane, `(k_d + k_dd md) cdt`, the
    recruitment of MinE by membrane MinD, `k_de md ce`, the dissociation of MinDE complexes, `k_off mde`, which releases
    MinD-ADP and MinE to the cytosol, and the nucleotide exchange `lam cdd` from MinD-ADP to MinD-ATP. They conserve
    the totals of MinD, `md + mde + cdt + cdd`, and of MinE, `mde + ce`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `MinProteinSkeletonSystem` read its parameters.

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
        >>> from skcomplex.systems.biology.systems_biology import MinSkeletonElement
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


class MinProteinSkeletonSystem(BaseSystem):
    """The skeleton system of the Min proteins on a periodic line.

    It is a mass-conserving reaction-diffusion system.

    MinD and MinE cycle between the membrane, where they diffuse slowly, and the cytosol, where they diffuse fast, with
    the reactions of `MinSkeletonElement`. The reactions conserve the totals of MinD and MinE, so only diffusion moves
    mass, and the local reactive equilibria map the dynamics onto conserved fields with nonreciprocal cross-diffusion
    (Brauns and Marchetti, App. F). With the default totals, the uniform state is unstable to an oscillatory band of
    wavelengths near 17, and narrow bands of membrane MinD travel along the line and meet head-on. The fast cytosolic
    diffusion and the fast recruitment make the system stiff, so the catalog registers the implicit `BDF` method with
    the exact Jacobian. The system has the membrane fields `cell.md` and `cell.mde` and the cytosolic fields `cell.cdt`,
    `cell.cdd`, and `cell.ce`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import MinProteinSkeletonSystem
        >>> values = MinProteinSkeletonSystem(32).prepare().system_.elements[0].values
        >>> round(float(values['ce'].mean() + values['mde'].mean()), 6)
        120.0
    """

    field = 'biology/systems_biology'
    topics = ('field', 'min proteins', 'mass conserving', 'reaction-diffusion', 'traveling waves', 'pde')
    references = ('Brauns and Marchetti 2024', 'Halatek and Frey 2018', 'Huang, Meir, and Wingreen 2003')
    until = 400.0
    dt = 4.0
    solver = ScipyODESolver(method='BDF', jacobian='exact')

    def __init__(
        self,
        n_sites: int = 128,
        *,
        length: float = 100.0,
        total_D: float = 200.0,
        total_E: float = 120.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_sites = n_sites
        self.length = length
        self.total_D = total_D
        self.total_E = total_E
        self.random_state = random_state

    @staticmethod
    def _uniform_state(element: MinSkeletonElement, total_D: float, total_E: float) -> dict[str, float]:
        """Return the uniform steady state of the skeleton system.

        It has the totals of MinD and MinE.
        """

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

    def _build_system(self) -> System:
        """Return the parts of the skeleton system of the Min proteins on a periodic line."""
        rng = np.random.default_rng(self.random_state)
        uniform = self._uniform_state(MinSkeletonElement('site'), self.total_D, self.total_E)
        values: dict[str, Array] = {name: np.full(self.n_sites, value) for name, value in uniform.items()}
        noise = 0.01 * uniform['md'] * rng.uniform(-1.0, 1.0, self.n_sites)
        values['md'] = values['md'] + noise
        values['cdt'] = values['cdt'] - noise
        space = LatticeSpace(self.n_sites, spacing=self.length / self.n_sites)
        cell = FieldElement('cell', element=MinSkeletonElement, space=space, values=values)
        reactions = [
            ReactionInteraction(
                'attachment',
                target='cell',
                reactants={'cdt': 1},
                products={'md': 1},
                propensity=lambda s: (s.target.k_d + s.target.k_dd * s.target.md) * s.target.cdt,
            ),
            ReactionInteraction(
                'recruitment',
                target='cell',
                reactants={'md': 1, 'ce': 1},
                products={'mde': 1},
                propensity=lambda s: s.target.k_de * s.target.md * s.target.ce,
            ),
            ReactionInteraction(
                'release',
                target='cell',
                reactants={'mde': 1},
                products={'cdd': 1, 'ce': 1},
                propensity=lambda s: s.target.k_off * s.target.mde,
            ),
            ReactionInteraction(
                'exchange',
                target='cell',
                reactants={'cdd': 1},
                products={'cdt': 1},
                propensity=lambda s: s.target.lam * s.target.cdd,
            ),
        ]
        membrane = [
            DiffusionInteraction(f'diffusion_{state}', target='cell', state=state, coefficient=lambda s: s.target.Dm)
            for state in ('md', 'mde')
        ]
        cytosol = [
            DiffusionInteraction(f'diffusion_{state}', target='cell', state=state, coefficient=lambda s: s.target.Dc)
            for state in ('cdt', 'cdd', 'ce')
        ]
        return System(self.label, elements=[cell], interactions=[*reactions, *membrane, *cytosol])
