"""Size-structured forest, a density of trees over position and height.

The density changes by shading, competition, and dispersal.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, exp
from sage.modules.free_module_element import FreeModuleElement

from ....core.components.elements import FieldElement
from ....core.components.interactions import ConservationInteraction, RateInteraction
from ....core.components.laws.utils import (
    convolve,
    coordinate,
    cumulative_integral,
    displacement,
    reduce_axes,
    where,
)
from ....core.components.spaces import BaseSpace, LatticeSpace, PhaseSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import RK4Solver
from ....core.errors import ValidationError
from ....core.specs import AxisSpec
from ..common.populations import PopulationDensityElement

HEIGHT_AXIS = 1


class SizeGrowthInteraction(ConservationInteraction):
    """Size growth `du/dt + divergence(a u) = source(ns)` of a size density `u` by a growth rate.

    Ecology names it the growth of individuals along a size axis, the transport term of a size-structured population.
    The drift field is `a(ns)`. The flux of the conservation law is the drift times the state, written in conservative
    form, so the total of the state on a periodic field stays constant and, on a spectral field, the dealiasing of
    products applies to it. The drift is one expression for a field of one dimension, or one per dimension, of the
    states and parameters of the field, and may read other states than the one it carries, such as the flow of a fluid
    derived from a streamfunction, or the drift of a probability density in a Fokker-Planck equation.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the field whose state is carried.

        state:
            The name of the carried state of the field.

        drift:
            A function of the namespace that returns the drift, or one drift per dimension.

        source:
            A function of the namespace that returns the source, or `None` for none.

        limiter:
            The slope limiter of the reconstruction of a finite-volume field, `'minmod'`, `'van-leer'`, or `None`.

        riemann:
            The numerical flux of a finite-volume field, `'rusanov'` or `'godunov'`, or `None` for the divergence of
            the flux on a field of another discretization.

        sources:
            The paths of other elements the drift and the source read, as the namespaces `sources`. A field over the
            position lattice of a target on a phase space is read broadcast along the class axes.

    Raises:
        ValidationError:
            The state is not a name, a function is not callable, or the limiter or the numerical flux is unknown.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element, FieldElement
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.biology.ecology import SizeGrowthInteraction
        >>> class CellElement(Element):
        ...     def __init__(self, label, u: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.u = u
        >>> space = LatticeSpace((50,), spacing=0.1, boundary='periodic')
        >>> values = {'u': np.exp(-((np.arange(50) - 25) / 5.0) ** 2)}
        >>> volumes = 'finite-volume'
        >>> field = FieldElement('tracer', element=CellElement, space=space, values=values, discretization=volumes)
        >>> drift = SizeGrowthInteraction('drift', target='tracer', state='u', drift=lambda s: 1.0)
        >>> system = CompositeSystem(elements=[field], interactions=[drift])
        >>> trajectory = simulate(system, until=1.0, dt=0.01)
        >>> total = trajectory.get_values('tracer.u')[-1].sum()
        >>> round(float(total), 6) == round(float(values['u'].sum()), 6)
        True
    """

    def __init__(
        self,
        label: str,
        *,
        target: str,
        state: str = 'u',
        drift: Callable[[SimpleNamespace], object] | None = None,
        source: Callable[[SimpleNamespace], object] | None = None,
        limiter: str | None = 'minmod',
        riemann: str | None = 'rusanov',
        sources: Sequence[str] = (),
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            source=source,
            limiter=limiter,
            riemann=riemann,
            sources=sources,
        )
        self.drift = drift

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an invalid state, drift, or scheme.

        It raises it for a state that is not a name, a drift that is not callable, or an unknown scheme.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The type declares states, the state is not a name, a function is not callable, or the limiter or the
                numerical flux is unknown.
        """
        super()._prepare(context)
        if not callable(self.drift):
            error_msg = f'The drift of {self.label!r} is {self.drift!r}. Give a function of the namespace.'
            raise ValidationError(error_msg)

    def _get_flux(self, namespace: SimpleNamespace) -> object:
        """Return the flux `a u`, one component per component of the drift.

        Args:
            namespace:
                The namespace of the symbols the flux can use, with the namespace of the field.

        Returns:
            flux:
                The flux, one expression or one per dimension.
        """
        state = getattr(self._get_member(namespace, 'target'), self.state)
        drift = self.drift(namespace) if callable(self.drift) else None
        if isinstance(drift, FreeModuleElement | list | tuple):
            return [SR(component) * state for component in drift]
        return SR(drift) * state


class ForestSizeStructureSystem(BaseSystem):
    """The McKendrick-von Foerster density of trees over position and height.

    The position runs across a strip.

    The density `f(x, h)` of trees of height `h` at position `x` across a forest strip grows in height at the rate `I (1
    - h / top)`, where the light `I = exp(-shading int_h^top f(x, h') h'^(2H) dh')` falls through the canopy of taller
    trees by the Beer-Lambert law. Trees die from competition for canopy space at the rate `I h^(2H) f` and for
    resources at the rate `I (1 - h / top) a / h`, where `a = R / resources` and `R = int h^(1+2H) f dh` is the resource
    consumption of the trees at `x`. Saplings enter at the height `sapling` from the seeds `int K(x - x') R(x') dx'` of
    the trees around, dispersed by the exponential kernel `K` of length `dispersal` and integral `fecundity`, and no
    seed comes from beyond the edges of the strip, where the forest was removed. Time and density are rescaled so that
    the growth rate and the canopy mortality are one (Grilletta et al. 2026, arXiv 2608.11918 Eq. 9, 10, 16, and 25).
    The kernel is the exponential of Eq. 35 along the axis across the strip, normalized in one dimension. The system has
    the density `forest.x` over position and height.

    Args:
        positions:
            The number of positions across the strip.

        heights:
            The number of cells of the height axis.

        width:
            The width `L` of the strip.

        dispersal:
            The dispersal length `xi`.

        hurst:
            The Hurst exponent `H` of the crown radius, which scales as `h^H`.

        sapling:
            The height `h0` at which saplings enter.

        top:
            The asymptotic height `h_u`.

        shading:
            The light attenuation `gamma`.

        fecundity:
            The integral `kappa` of the dispersal kernel.

        resources:
            The resource flux `R_a`.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ForestSizeStructureSystem
        >>> ForestSizeStructureSystem(8, 12).prepare().system_.elements[0].space.prepare().shape_
        (8, 12)
    """

    field = 'biology/ecology'
    topics = ('field', 'size structure', 'forest', 'integral operator', 'dispersal', 'pde')
    references = ('Grilletta et al. 2026',)
    until = 200.0
    dt = 5.0
    solver = RK4Solver(dt=0.1)

    def __init__(
        self,
        positions: int = 40,
        heights: int = 100,
        *,
        width: float = 10.0,
        dispersal: float = 1.0,
        hurst: float = 1.0,
        sapling: float = 0.1,
        top: float = 50.0,
        shading: float = 1.0,
        fecundity: float = 0.01,
        resources: float = 125.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.positions = positions
        self.heights = heights
        self.width = width
        self.dispersal = dispersal
        self.hurst = hurst
        self.sapling = sapling
        self.top = top
        self.shading = shading
        self.fecundity = fecundity
        self.resources = resources

    def _build_system(self) -> CompositeSystem:
        """Return the components of the McKendrick-von Foerster density of trees.

        The density is over position across a strip and height.
        """
        step = (self.top - self.sapling) / self.heights
        space = PhaseSpace(
            LatticeSpace((self.positions,), spacing=self.width / self.positions, boundary='fixed'),
            [AxisSpec('h', self.heights, self.sapling, self.top)],
        )
        centres = space.prepare().coordinates_['h']
        start = np.tile(0.1 * (centres / self.sapling) ** -2.0, (self.positions, 1))
        forest = FieldElement(
            'forest',
            element=PopulationDensityElement,
            space=space,
            values={'x': start},
            discretization='finite-volume',
        )
        h = coordinate('h')
        growth = 1 - h / self.top
        kernel = self.fecundity / (2 * self.dispersal) * exp(-abs(displacement(0)) / self.dispersal)

        def light(s: SimpleNamespace) -> Expression:
            return exp(
                -self.shading * cumulative_integral(s.target.x * h ** (2 * self.hurst), HEIGHT_AXIS, reverse=True),
            )

        def demography(s: SimpleNamespace) -> Expression:
            consumption = reduce_axes(s.target.x * h ** (1 + 2 * self.hurst), ('h',))
            recruits = where(h < self.sapling + step, light(s) * growth * convolve(consumption, kernel) / step, 0)
            mortality = light(s) * (h ** (2 * self.hurst) * s.target.x + growth * consumption / (self.resources * h))
            return recruits - mortality * s.target.x

        grow = SizeGrowthInteraction(
            'growth',
            target='forest',
            state='x',
            drift=lambda s: [0, light(s) * growth],
            limiter=None,
        )
        demographics = RateInteraction('demography', target='forest', state='x', rate=demography)
        return CompositeSystem(self.label, elements=[forest], interactions=[grow, demographics])
