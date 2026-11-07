"""Ecology systems: consumers that interact only through the resources they share."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.system import System
from ....core.errors import ValidationError
from ..._catalog import register_system
from ._consumer import ConsumerElement
from ._consumer_growth import ConsumerGrowthInteraction
from ._resource import ResourceElement
from ._resource_turnover import ResourceTurnoverInteraction

UPTAKE = ((0.5, 0.3), (0.4, 0.6))
CAPACITIES = (4.8, 2.85)


@register_system(
    field='biology/ecology',
    tags=('consumer-resource', 'competitive exclusion', 'niche theory', 'ode'),
    references=('MacArthur 1970', 'Tilman 1982', 'Mehta and Blumenthal 2024', 'Gros 2015'),
    defaults={'until': 100.0, 'dt': 0.5},
)
def consumer_resource(
    consumers: Sequence[float] | None = None,
    resources: Sequence[float] | None = None,
    *,
    c: Sequence[Sequence[float]] = UPTAKE,
    m: Sequence[float] | float = 1.0,
    e: Sequence[float] | float = 1.0,
    w: Sequence[float] | float = 1.0,
    r: Sequence[float] | float = 1.0,
    K: Sequence[float] | float = CAPACITIES,
    kappa: Sequence[float] | float = CAPACITIES,
    omega: Sequence[float] | float = 1.0,
    supply: str = 'logistic',
    kind: str = 'substitutable',
    n: float = -10.0,
    v: Sequence[Sequence[float]] | None = None,
) -> System:
    """Return consumer species that compete only through the resources they take up, by MacArthur's niche theory.

    The `'logistic'` supply of substitutable resources is MacArthur's consumer-resource model, `dN_i/dt = e_i N_i
    (sum_b w_b c_ib R_b - m_i)` and `dR_a/dt = r_a R_a (K_a - R_a) / K_a - sum_j c_ja N_j R_a`. The `'chemostat'` supply
    `kappa_a - omega_a R_a` models nutrients that flow in from outside, and essential resources limit growth jointly
    through `(sum_b (c_ib R_b)^n)^(1/n)`, consumed by the stoichiometry `v`. At most as many consumers as resources
    coexist, at the point where their zero net growth isoclines `g_i(R) = m_i` cross. With the defaults, the settings of
    Mehta and Blumenthal (2024, figure 4), both consumers coexist at `R = (5/3, 5/9)`, the point of the region where
    no consumer can invade that is closest to the supply point `K`.

    The competition for one resource of Gros (2015, exercise 8), `dx_i/dt = (f r_i - d) x_i` and `df/dt = a - f sum_i
    r_i x_i`, is the chemostat with one resource, `c = r`, `m = d`, `kappa = a`, and `omega = 0`. The total population
    tends to `a / d`, the resource to `min_i d / r_i`, and only the consumer with the largest `r_i` survives.

    Args:
        consumers:
            The initial population sizes, or `None` for 0.1 each.

        resources:
            The initial resource abundances, or `None` for the capacities `K` or the inflows `kappa`.

        c:
            The uptake rates, one row per consumer and one column per resource.

        m:
            The maintenance energies of the consumers.

        e:
            The efficiencies of the consumers.

        w:
            The energies of the resources.

        r:
            The growth rates of self-renewing resources.

        K:
            The carrying capacities of self-renewing resources.

        kappa:
            The inflows of supplied resources.

        omega:
            The dilution rates of supplied resources.

        supply:
            `'logistic'` for self-renewing resources or `'chemostat'` for supplied ones.

        kind:
            `'substitutable'` or `'essential'` resources.

        n:
            The exponent of the growth on essential resources, below -1.

        v:
            The stoichiometry of essential resources, one row per consumer, or `None` for `c`.

    Returns:
        system:
            The system, with the populations `consumers.N` and the resources `resources.R`.

    Raises:
        ValidationError:
            The uptake rates are not one row per consumer with one column per resource.

    Examples:
        >>> from skcomplex.systems.biology.ecology import consumer_resource
        >>> [group.prepare().size_ for group in consumer_resource().elements]
        [2, 2]
    """
    uptake = np.atleast_2d(np.asarray(c, dtype=float))
    n_consumers, n_resources = uptake.shape
    sizes = np.full(n_consumers, 0.1) if consumers is None else np.asarray(consumers, dtype=float)
    supplied = K if supply == 'logistic' else kappa
    abundances = np.asarray(resources if resources is not None else np.broadcast_to(supplied, n_resources), float)
    if sizes.shape != (n_consumers,) or abundances.shape != (n_resources,):
        error_msg = (
            f'The uptake rates of shape {uptake.shape} need {n_consumers} initial populations and {n_resources} '
            'initial resources.'
        )
        raise ValidationError(error_msg)
    consumer_group = GroupElement(
        'consumers',
        element=ConsumerElement,
        size=n_consumers,
        values={'N': sizes, 'm': m, 'e': e},
    )
    resource_group = GroupElement(
        'resources',
        element=ResourceElement,
        size=n_resources,
        values={'R': np.array(abundances, dtype=float), 'w': w, 'r': r, 'K': K, 'kappa': kappa, 'omega': omega},
    )
    uptake_rates = tuple(map(tuple, uptake.tolist()))
    growth = ConsumerGrowthInteraction(
        'growth',
        consumers='consumers',
        resources='resources',
        c=uptake_rates,
        kind=kind,
        n=n,
    )
    turnover = ResourceTurnoverInteraction(
        'turnover',
        consumers='consumers',
        resources='resources',
        c=uptake_rates,
        kind=kind,
        n=n,
        supply=supply,
        v=v,
    )
    return System(elements=[consumer_group, resource_group], interactions=[growth, turnover])
