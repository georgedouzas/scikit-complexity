"""Gauge arbitrage: exchange rates on links as lattice electrodynamics, sourced by traders who move a good on sites."""

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

from ....core.elements import Element, GroupElement
from ....core.interactions import NeighborInteraction, RateInteraction
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import codifferential, exterior_derivative
from ....core.spaces import CellComplexSpace, LatticeSpace
from ....core.system import System
from ....engine.solvers import RK4Solver
from ....specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model

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
        >>> from skcomplex.models.economics.finance import GoodSiteElement
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
    """Link of a lattice with an exchange-rate potential `A` and its rate of change `E`.

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
        >>> from skcomplex.models.economics.finance import ExchangeLinkElement
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
    """Trader current `J = -D (rho_t - rho_s - q A (rho_s + rho_t) / 2)` along the links, down the covariant gradient.

    The covariant derivative is `D_mu = d_mu - A_mu` of Schwichtenberg (2020 Eq. 2.55), with the charge `q` of its
    Eq. 4.24, and the current is minus the diffusion constant times this derivative of the good, as in Eq. 2.56. The
    current moves the good from the source of a link to its target (footnote 18 of §2.1), and it is the source of the
    field of the link with the sign of the Ampere-Maxwell law of Eq. 2.39, `E' = ... - J`, so the Gauss law holds as the
    good moves. These signs give the potential a mass, `A'' = ... - D q rho A`, for a positive density and charge, so
    the fields stay bounded.

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
        >>> from skcomplex.models.economics.finance import TraderCurrentInteraction
        >>> TraderCurrentInteraction('current', target='sites', links='links', q=0.5).q
        0.5
    """

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
        """Return the rates of the good at both ends of a link and of the electric field of the link.

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


def _potential_rate(namespace: SimpleNamespace) -> Expression:
    """Return the rate of the potential, the field `F_0i` of the link."""
    return namespace.target.E


def _field_rate(namespace: SimpleNamespace) -> Expression:
    """Return the rate of the field without the current, minus the codifferential of the exterior derivative."""
    return -codifferential(exterior_derivative(namespace.target.A))


def _total_good(t: float, values: Mapping[str, object]) -> float:
    """Return the total of the good."""
    del t
    return float(np.sum(np.asarray(values['sites.rho'], dtype=float)))


def _gauss_residual(edges: scipy.sparse.csr_array, t: float, values: Mapping[str, object]) -> float:
    """Return the largest absolute difference of the codifferential of the field and the excess of the good."""
    del t
    density = np.ravel(np.asarray(values['sites.rho'], dtype=float))
    electric = np.ravel(np.asarray(values['links.E'], dtype=float))
    return float(np.abs(-(edges.T @ electric) - (density - density.mean())).max())


def _market_observables(system: System) -> list[ObservableSpec]:
    """Return the total of the good and the Gauss residual against its excess density."""
    space = system.elements[0].space.prepare()
    return [
        ObservableSpec('total_good', function=_total_good),
        ObservableSpec('gauss_residual', function=partial(_gauss_residual, space.incidences_[0])),
    ]


def _initial_field(edges: scipy.sparse.csr_array, excess: Array) -> Array:
    """Return the electric field of the smallest norm whose codifferential is the excess of the good."""
    field = scipy.sparse.linalg.lsqr(
        -edges.T,
        excess,
        atol=SOLVER_TOLERANCE,
        btol=SOLVER_TOLERANCE,
        conlim=0.0,
        iter_lim=100 * excess.size,
    )[0]
    return np.asarray(field, dtype=float)


@register_model(
    field='economics/finance',
    tags=('gauge', 'arbitrage', 'exchange rates', 'lattice', 'maxwell', 'cell complex'),
    references=('Schwichtenberg 2020',),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': RK4Solver(dt=0.02)},
    observables=_market_observables,
)
def gauge_arbitrage_market(
    size: int = 16,
    *,
    D: float = 0.5,
    q: float = 0.2,
    bump: float = 1.0,
    seed: int = 0,
) -> System:
    """Return exchange-rate potentials on links as lattice Maxwell fields, sourced by traders who move a good on sites.

    The potentials follow `A_t = E` and `E_t = -codifferential(dA) - J`, and the good follows `rho_t =
    -codifferential(J)` with the current `J = -D (d rho - q A rho)` along its covariant gradient, after Schwichtenberg
    (2020 §2.1 and §2.2.1). Here `E` is `F_0i` of the book, so `codifferential(E)` equals the excess of the good over
    its mean, as in its Gauss law, Eq. 2.37, with a uniform background of the mean density that keeps the periodic
    lattice neutral. The model departs from the book in two places. The book writes Eq. 2.39 with no metric, which gives
    `A_tt = +codifferential(dA) - J`, where every mode with a curl grows, so the model takes the sign of a wave. And
    Eq. 2.56 applies the covariant derivative twice, which does not conserve the good, so the model applies it once
    inside the divergence of the current. The book gives no values of `D` and `q` and sets `mu_0` by no number, so `D`,
    `q`, and `mu_0 = 1` are choices of this model.

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

    Returns:
        system:
            The system.

    Examples:
        >>> from skcomplex.models.economics.finance import gauge_arbitrage_market
        >>> [element.label for element in gauge_arbitrage_market(4).elements]
        ['sites', 'links']
    """
    space = CellComplexSpace(LatticeSpace((size, size))).prepare()
    density = 1.0 + bump * np.random.default_rng(seed).random(size * size)
    field = _initial_field(space.incidences_[0], density - density.mean())
    sites = GroupElement('sites', element=GoodSiteElement, space=space, on='sites', values={'rho': density})
    links = GroupElement(
        'links',
        element=ExchangeLinkElement,
        space=space,
        on='edges',
        values={'E': field},
        visual=VisualSpec('cell', {'state': 'A'}),
    )
    interactions = [
        RateInteraction('potential', target='links', state='A', rate=_potential_rate),
        RateInteraction('field', target='links', state='E', rate=_field_rate),
        TraderCurrentInteraction('current', target='sites', links='links', D=D, q=q),
    ]
    return System(elements=[sites, links], interactions=interactions)
