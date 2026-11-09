"""Gauge arbitrage: exchange rates on links as lattice electrodynamics.

Traders who move a good on sites source them.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import NeighborInteraction, RateInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import codifferential, exterior_derivative
from ....core.components.spaces import CellComplexSpace, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem, RunSettings
from ....core.engine.solvers import RK4Solver
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array

SOLVER_TOLERANCE = 1e-15


class GoodSiteElement(Element):
    """Site of a lattice with a density `rho` of a good.

    Args:
        label:
            The label of the site or the group on the sites.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        rho:
            The initial density of the good.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.economics.finance import GoodSiteElement
        >>> GoodSiteElement('site', rho=2.0).rho
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho


class ExchangeLinkElement(Element):
    """Link of a lattice with an exchange-rate potential `A` and its rate `E`.

    The rate `E` is the rate of change of the potential.

    Args:
        label:
            The label of the link or the group on the links.

        visual:
            How the link is drawn, or `None` for `default_visual_`.

        A:
            The initial exchange-rate potential.

        E:
            The initial value of `F_0i` of Schwichtenberg (2020 Eq. 2.22), the rate of change of the potential.

    Attributes:
        default_visual_:
            The visual of the link when it sets none.

    Examples:
        >>> from skcomplex.systems.economics.finance import ExchangeLinkElement
        >>> ExchangeLinkElement('link', A=0.5).A
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec()] = 0.0,
        E: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.E = E


class TraderCurrentInteraction(NeighborInteraction):
    """Trader current along the links, down the covariant gradient.

    The current is `J = -D (rho_t - rho_s - q A (rho_s + rho_t) / 2)`. The covariant derivative is `D_mu = d_mu - A_mu`
    of Schwichtenberg (2020 Eq. 2.55), with the charge `q` of its Eq. 4.24, and the current is minus the diffusion
    constant times this derivative of the good, as in Eq. 2.56. The current moves the good from the source of a link to
    its target (footnote 18 of §2.1), and it is the source of the field of the link with the sign of the Ampere-Maxwell
    law of Eq. 2.39, `E' = ... - J`, so the Gauss law holds as the good moves. These signs give the potential a mass,
    `A'' = ... - D q rho A`, for a positive density and charge, so the fields stay bounded.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group on the sites.

        links:
            The path of the group on the edges.

        D:
            The diffusivity of the good.

        q:
            The charge of the good.

    Examples:
        >>> from skcomplex.systems.economics.finance import TraderCurrentInteraction
        >>> TraderCurrentInteraction('current', target='sites', links='links', q=0.5).q
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        q: Annotated[float, ParameterSpec()] = 0.2,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.links = links
        self.D = D
        self.q = q

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the good at both ends of a link and of its electric field.

        Args:
            namespace:
                The namespace of an edge, with the ends `source` and `target` and the member `edge`.

        Returns:
            laws:
                The rates.
        """
        source, target, edge = namespace.source, namespace.target, namespace.edge
        middle = (source.rho + target.rho) / 2
        current = -namespace.D * (target.rho - source.rho - namespace.q * edge.A * middle)
        return [RateLaw(source.rho, -current), RateLaw(target.rho, current), RateLaw(edge.E, -current)]


def _total_good(t: float, values: Mapping[str, object]) -> float:
    """Return the total of the good."""
    del t
    return float(np.sum(np.asarray(values['sites.rho'], dtype=float)))


def _gauss_residual(edges: scipy.sparse.csr_array, t: float, values: Mapping[str, object]) -> float:
    """Return the largest absolute difference of the codifferential and the excess.

    The codifferential is that of the field, and the excess that of the good.

    The excess is that of the good.
    """
    del t
    density = np.ravel(np.asarray(values['sites.rho'], dtype=float))
    electric = np.ravel(np.asarray(values['links.E'], dtype=float))
    return float(np.abs(-(edges.T @ electric) - (density - density.mean())).max())


def _market_observables(system: CompositeSystem) -> list[ObservableSpec]:
    """Return the total of the good and the Gauss residual against its excess density."""
    space = system.elements[0].space.prepare()
    return [
        ObservableSpec('total_good', function=_total_good),
        ObservableSpec('gauss_residual', function=partial(_gauss_residual, space.incidences_[0])),
    ]


class GaugeArbitrageMarketSystem(BaseSystem):
    """Exchange-rate potentials on links as lattice Maxwell fields.

    Traders who move a good on sites source them.

    The potentials follow `A_t = E` and `E_t = -codifferential(dA) - J`, and the good follows `rho_t =
    -codifferential(J)` with the current `J = -D (d rho - q A rho)` along its covariant gradient, after Schwichtenberg
    (2020 §2.1 and §2.2.1). Here `E` is `F_0i` of the book, so `codifferential(E)` equals the excess of the good over
    its mean, as in its Gauss law, Eq. 2.37, with a uniform background of the mean density that keeps the periodic
    lattice neutral. The system departs from the book in two places. The book writes Eq. 2.39 with no metric, which
    gives `A_tt = +codifferential(dA) - J`, where every mode with a curl grows, so the system takes the sign of a wave.
    And Eq. 2.56 applies the covariant derivative twice, which does not conserve the good, so the system applies it once
    inside the divergence of the current. The book gives no values of `D` and `q` and sets `mu_0` by no number, so `D`,
    `q`, and `mu_0 = 1` are choices of this system.

    Args:
        size:
            The number of sites along both axes.

        D:
            The diffusivity of the good.

        q:
            The charge of the good.

        bump:
            The height of the random part of the initial density above a density of 1.

        seed:
            The seed of the initial density.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.finance import GaugeArbitrageMarketSystem
        >>> [element.label for element in GaugeArbitrageMarketSystem(4).prepare().system_.elements]
        ['sites', 'links']
    """

    field = 'economics/finance'
    topics = ('gauge', 'arbitrage', 'exchange rates', 'lattice', 'maxwell', 'cell complex')
    references = ('Schwichtenberg 2020',)
    until = 20.0
    dt = 0.5
    solver = RK4Solver(dt=0.02)

    def __init__(
        self,
        size: int = 16,
        *,
        D: float = 0.5,
        q: float = 0.2,
        bump: float = 1.0,
        seed: int = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.D = D
        self.q = q
        self.bump = bump
        self.seed = seed

    @staticmethod
    def _potential_rate(namespace: SimpleNamespace) -> Expression:
        """Return the rate of the potential, the field `F_0i` of the link."""
        return namespace.target.E

    @staticmethod
    def _field_rate(namespace: SimpleNamespace) -> Expression:
        """Return the rate of the field without the current.

        It is minus the codifferential of the exterior derivative.
        """
        return -codifferential(exterior_derivative(namespace.target.A))

    @staticmethod
    def _initial_field(edges: scipy.sparse.csr_array, excess: Array) -> Array:
        """Return the electric field of the smallest norm with a given codifferential.

        Its codifferential is the excess of the good.
        """
        field = scipy.sparse.linalg.lsqr(
            -edges.T,
            excess,
            atol=SOLVER_TOLERANCE,
            btol=SOLVER_TOLERANCE,
            conlim=0.0,
            iter_lim=100 * excess.size,
        )[0]
        return np.asarray(field, dtype=float)

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        space = CellComplexSpace(LatticeSpace((self.size, self.size))).prepare()
        density = 1.0 + self.bump * np.random.default_rng(self.seed).random(self.size * self.size)
        field = self._initial_field(space.incidences_[0], density - density.mean())
        sites = GroupElement(
            'sites',
            element=GoodSiteElement,
            space=space,
            on='sites',
            values={'rho': density},
            visual=VisualSpec(),
        )
        links = GroupElement(
            'links',
            element=ExchangeLinkElement,
            space=space,
            on='edges',
            values={'E': field},
            visual=VisualSpec('cell', {'state': 'A'}),
        )
        interactions = [
            RateInteraction('potential', target='links', state='A', rate=self._potential_rate),
            RateInteraction('field', target='links', state='E', rate=self._field_rate),
            TraderCurrentInteraction('current', target='sites', links='links', D=self.D, q=self.q),
        ]
        return CompositeSystem(self.label, elements=[sites, links], interactions=interactions)

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The observables of the parts are built from the model parameters.
        """
        settings = super().get_run_settings()
        settings['observables'] = _market_observables(self._get_pass_parameters())
        return settings
