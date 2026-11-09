"""Ecology systems: consumers that interact only through the resources they share."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import total
from ....core.components.spaces import BaseSpace
from ....core.components.systems import BaseSystem, System
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

KINDS = ('substitutable', 'essential')

SUPPLIES = ('logistic', 'chemostat')

UPTAKE = ((0.5, 0.3), (0.4, 0.6))

CAPACITIES = (4.8, 2.85)


class ConsumerElement(Element):
    """Consumer species that grows on the energy it takes from resources.

    It needs a maintenance energy to persist.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        N:
            The initial population size.

        e:
            The efficiency that converts the energy taken in into growth.

        m:
            The maintenance energy, the intake at which the population neither grows nor declines.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ConsumerElement
        >>> ConsumerElement('consumers', m=0.5).m
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        e: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.N = N
        self.e = e
        self.m = m


class ResourceElement(Element):
    """Resource that consumers take up, self-renewing or supplied from outside.

    A self-renewing resource grows logistically.

    A self-renewing resource grows as `r R (K - R) / K`, and a supplied one flows in at `kappa` and is washed out at
    the rate `omega`. The interaction that couples the resource to its consumers chooses which supply applies.

    Args:
        label:
            The label of the resource or of the group of resources.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        R:
            The initial abundance.

        w:
            The energy a unit of the resource carries.

        r:
            The growth rate of a self-renewing resource at low abundance.

        K:
            The carrying capacity of a self-renewing resource.

        kappa:
            The inflow of a supplied resource.

        omega:
            The dilution rate of a supplied resource.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ResourceElement
        >>> ResourceElement('resources', K=4.8).K
        4.8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        R: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        w: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        omega: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.R = R
        self.w = w
        self.r = r
        self.K = K
        self.kappa = kappa
        self.omega = omega


class ConsumptionInteraction(Interaction):
    """Consumption of substitutable or essential resources by consumer species.

    Consumption is the only way the consumers interact.

    Consumer `i` takes up resource `a` at the rate `c_ia R_a` per individual. Substitutable resources add their
    energies `w_b c_ib R_b`, and essential resources all limit the growth, `g_i = (sum_b (c_ib R_b)^n)^(1/n)` for
    `n < -1`, which tends to Liebig's law `g_i = min_b c_ib R_b` as `n` goes to minus infinity. Its subclasses give the
    laws of the consumers and of the resources, which read the other group through reductions.

    Args:
        label:
            The label of the interaction.

        consumers:
            The path of the group of consumers.

        resources:
            The path of the group of resources.

        c:
            The uptake rates, one row per consumer and one column per resource.

        kind:
            `'substitutable'` or `'essential'` resources.

        n:
            The exponent of the growth on essential resources, below -1.

    Raises:
        ValidationError:
            The kind is unknown, or the exponent of essential resources is not below -1.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ConsumptionInteraction
        >>> uptake = ((0.5, 0.3), (0.4, 0.6))
        >>> ConsumptionInteraction('consumption', consumers='consumers', resources='resources', c=uptake).kind
        'substitutable'
    """

    def __init__(
        self,
        label: str,
        *,
        consumers: Annotated[str, RoleSpec()],
        resources: Annotated[str, RoleSpec()],
        c: Sequence[Sequence[float]] = ((1.0,),),
        kind: str = 'substitutable',
        n: float = -10.0,
    ) -> None:
        super().__init__(label)
        self.consumers = consumers
        self.resources = resources
        self.c = c
        self.kind = kind
        self.n = n

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Check the kind and the exponent.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The kind is unknown, or the exponent of essential resources is not below -1.
        """
        super()._prepare(context)
        if self.kind not in KINDS:
            error_msg = f'The kind must be one of {", ".join(KINDS)}, not {self.kind!r}.'
            raise ValidationError(error_msg)
        if self.kind == 'essential' and self.n >= -1:
            error_msg = f'The exponent of essential resources must be below -1, not {self.n}.'
            raise ValidationError(error_msg)

    def _essential_growth(self, R: Expression) -> Expression:
        """Return the growth of a consumer on essential resources.

        The growth is `(sum_b (c_ib R_b)^n)^(1/n)`.
        """
        uptake = np.asarray(self.c, dtype=float)
        return total(R**self.n, weights=uptake**self.n) ** (1 / self.n)


class ConsumerGrowthInteraction(ConsumptionInteraction):
    """Growth of consumers on the resources they take up, minus their maintenance.

    On substitutable resources, `dN_i/dt = e_i N_i (sum_b w_b c_ib R_b - m_i)`. On essential resources, `dN_i/dt =
    N_i (g_i - m_i)` with `g_i = (sum_b (c_ib R_b)^n)^(1/n)`.

    Args:
        label:
            The label of the interaction.

        consumers:
            The path of the group of consumers.

        resources:
            The path of the group of resources.

        c:
            The uptake rates, one row per consumer and one column per resource.

        kind:
            `'substitutable'` or `'essential'` resources.

        n:
            The exponent of the growth on essential resources, below -1.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ConsumerGrowthInteraction
        >>> ConsumerGrowthInteraction('growth', consumers='consumers', resources='resources').c
        ((1.0,),)
    """

    law_classes = (RateLaw,)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth of the consumers.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        consumers, resources = s.consumers, s.resources
        if self.kind == 'essential':
            growth = self._essential_growth(resources.R) - consumers.m
        else:
            intake = total(resources.w * resources.R, weights=np.asarray(self.c, dtype=float))
            growth = consumers.e * (intake - consumers.m)
        return [RateLaw(consumers.N, consumers.N * growth)]


class ResourceTurnoverInteraction(ConsumptionInteraction):
    """Supply of resources and their depletion by the consumers.

    The supply is a logistic renewal or a chemostat.

    A `'logistic'` supply renews resource `a` at `r_a R_a (K_a - R_a) / K_a`, and a `'chemostat'` supply at `kappa_a -
    omega_a R_a`. Substitutable resources lose `sum_j c_ja N_j R_a`, and essential resources lose `sum_j v_ja N_j g_j`
    by the fixed stoichiometry `v` of the consumers.

    Args:
        label:
            The label of the interaction.

        consumers:
            The path of the group of consumers.

        resources:
            The path of the group of resources.

        c:
            The uptake rates, one row per consumer and one column per resource.

        kind:
            `'substitutable'` or `'essential'` resources.

        n:
            The exponent of the growth on essential resources, below -1.

        supply:
            `'logistic'` for self-renewing resources or `'chemostat'` for resources supplied from outside.

        v:
            The stoichiometry of essential resources in the consumers, one row per consumer, or `None` for `c`.

    Raises:
        ValidationError:
            The supply or the kind is unknown, or the exponent of essential resources is not below -1.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ResourceTurnoverInteraction
        >>> ResourceTurnoverInteraction('turnover', consumers='consumers', resources='resources').supply
        'logistic'
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        consumers: Annotated[str, RoleSpec()],
        resources: Annotated[str, RoleSpec()],
        c: Sequence[Sequence[float]] = ((1.0,),),
        kind: str = 'substitutable',
        n: float = -10.0,
        supply: str = 'logistic',
        v: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(label, consumers=consumers, resources=resources, c=c, kind=kind, n=n)
        self.supply = supply
        self.v = v

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Check the supply, the kind, and the exponent.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The supply or the kind is unknown, or the exponent of essential resources is not below -1.
        """
        super()._prepare(context)
        if self.supply not in SUPPLIES:
            error_msg = f'The supply must be one of {", ".join(SUPPLIES)}, not {self.supply!r}.'
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the supply and the depletion of the resources.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        consumers, resources = s.consumers, s.resources
        R = resources.R
        if self.supply == 'logistic':
            renewal = resources.r * R * (resources.K - R) / resources.K
        else:
            renewal = resources.kappa - resources.omega * R
        if self.kind == 'essential':
            stoichiometry = np.asarray(self.c if self.v is None else self.v, dtype=float)
            depletion = total(consumers.N * self._essential_growth(R), weights=stoichiometry.T)
        else:
            depletion = R * total(consumers.N, weights=np.asarray(self.c, dtype=float).T)
        return [RateLaw(R, renewal - depletion)]


class ConsumerResourceSystem(BaseSystem):
    """Consumer species that compete only through the resources they take up.

    It follows MacArthur's niche theory.

    The `'logistic'` supply of substitutable resources is MacArthur's consumer-resource model, `dN_i/dt = e_i N_i
    (sum_b w_b c_ib R_b - m_i)` and `dR_a/dt = r_a R_a (K_a - R_a) / K_a - sum_j c_ja N_j R_a`. The `'chemostat'` supply
    `kappa_a - omega_a R_a` models nutrients that flow in from outside, and essential resources limit growth jointly
    through `(sum_b (c_ib R_b)^n)^(1/n)`, consumed by the stoichiometry `v`. At most as many consumers as resources
    coexist, at the point where their zero net growth isoclines `g_i(R) = m_i` cross. With the defaults, the settings of
    Mehta and Blumenthal (2024, figure 4), both consumers coexist at `R = (5/3, 5/9)`, the point of the region where
    no consumer can invade that is closest to the supply point `K`.

    The competition for one resource of Gros (2015, exercise 8), `dx_i/dt = (f r_i - d) x_i` and `df/dt = a - f sum_i
    r_i x_i`, is the chemostat with one resource, `c = r`, `m = d`, `kappa = a`, and `omega = 0`. The total population
    tends to `a / d`, the resource to `min_i d / r_i`, and only the consumer with the largest `r_i` survives. The system
    has the populations `consumers.N` and the resources `resources.R`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The uptake rates are not one row per consumer with one column per resource.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ConsumerResourceSystem
        >>> [group.prepare().size_ for group in ConsumerResourceSystem().prepare().system_.elements]
        [2, 2]
    """

    field = 'biology/ecology'
    topics = ('consumer-resource', 'competitive exclusion', 'niche theory', 'ode')
    references = ('MacArthur 1970', 'Tilman 1982', 'Mehta and Blumenthal 2024', 'Gros 2015')
    until = 100.0
    dt = 0.5

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.consumers = consumers
        self.resources = resources
        self.c = c
        self.m = m
        self.e = e
        self.w = w
        self.r = r
        self.K = K
        self.kappa = kappa
        self.omega = omega
        self.supply = supply
        self.kind = kind
        self.n = n
        self.v = v

    def _build_system(self) -> System:
        """Return the parts of consumer species that compete only through resources.

        They compete only through the resources they take up.
        """
        uptake = np.atleast_2d(np.asarray(self.c, dtype=float))
        n_consumers, n_resources = uptake.shape
        sizes = np.full(n_consumers, 0.1) if self.consumers is None else np.asarray(self.consumers, dtype=float)
        supplied = self.K if self.supply == 'logistic' else self.kappa
        abundances = np.asarray(
            self.resources if self.resources is not None else np.broadcast_to(supplied, n_resources),
            float,
        )
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
            values={'N': sizes, 'm': self.m, 'e': self.e},
        )
        resource_group = GroupElement(
            'resources',
            element=ResourceElement,
            size=n_resources,
            values={
                'R': np.array(abundances, dtype=float),
                'w': self.w,
                'r': self.r,
                'K': self.K,
                'kappa': self.kappa,
                'omega': self.omega,
            },
        )
        uptake_rates = tuple(map(tuple, uptake.tolist()))
        growth = ConsumerGrowthInteraction(
            'growth',
            consumers='consumers',
            resources='resources',
            c=uptake_rates,
            kind=self.kind,
            n=self.n,
        )
        turnover = ResourceTurnoverInteraction(
            'turnover',
            consumers='consumers',
            resources='resources',
            c=uptake_rates,
            kind=self.kind,
            n=self.n,
            supply=self.supply,
            v=self.v,
        )
        return System(self.label, elements=[consumer_group, resource_group], interactions=[growth, turnover])
