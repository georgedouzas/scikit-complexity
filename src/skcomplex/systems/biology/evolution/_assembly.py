"""Evolution systems: assembly theory.

Assembly theory is the discovery of new objects by recursive joining of existing ones.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw, UpdateLaw
from ....core.components.laws.utils import shift_member
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver, SystemView
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role

MODES = ('undirected', 'directed')


class AssemblyLevelElement(Element):
    """Objects of one assembly index.

    They are discovered from the objects one assembly step below them.

    In a group ordered by assembly index, an `AssemblyDiscoveryInteraction` grows the objects of index `a + 1` as
    `dN_{a+1}/dt = k (N_a)^alpha`, Eq. (3) of Sharma et al. (2022), where the first member reads the constant pool
    `N_0` of building blocks. The selectivity `alpha = 1` reuses every object built before, and `alpha < 1` only a
    subset, the signature of selection.

    Args:
        label:
            The label of the level or of the group of levels.

        N:
            The initial number of objects.

        k:
            The rate of discovery.

        alpha:
            The selectivity of reuse.

        size:
            The number of levels, or `None` for one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import AssemblyLevelElement
        >>> AssemblyLevelElement('levels', alpha=0.5).alpha
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        N: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.0,
        k: Annotated[float | Array, RealSet.closed_open(0, oo)] = 1.0,
        alpha: Annotated[float | Array, RealSet.closed(0, 1)] = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.N = N
        self.k = k
        self.alpha = alpha


class AssemblyDiscoveryInteraction(Interaction):
    """Discovery of the objects of every assembly level from the level below, `dN_{a+1}/dt = k (N_a)^alpha`.

    The rate `k` and the selectivity `alpha` are values of the levels, an `AssemblyLevelElement`, and the first level
    is discovered from the pool of building blocks `N0`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of levels, ordered by assembly index.

        N0:
            The number of building blocks below the first level.

    Examples:
        >>> from skcomplex.systems.biology.evolution import AssemblyDiscoveryInteraction
        >>> AssemblyDiscoveryInteraction('discovery', target='levels').N0
        1.0
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role, N0: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.N0 = N0

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the discovery of the objects of every level from the level below.

        Args:
            context:
                The namespace of the group of levels.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            below = shift_member(s.N, -1, boundary='fixed', fill=self.N0)
            self.laws_: list[BaseLaw] = [RateLaw(s.N, s.k * below**s.alpha)]


class PolymerView(AgentView):
    """View of a polymer, with its states as attributes.

    `PolymerElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    length: float
    born: float


class PolymerElement(Element):
    """Linear polymer of an assembly pool, with its length and the step it was made.

    Args:
        label:
            The label of the polymer or of the group of polymers.

        length:
            The number of monomers.

        born:
            The step at which the polymer was made.

        size:
            The number of polymers, or `None` for one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import PolymerElement
        >>> PolymerElement('polymer', length=3.0).length
        3.0
    """

    def __init__(
        self,
        label: str,
        *,
        length: Annotated[float | Array, NN] = 1.0,
        born: Annotated[float | Array, NN] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.length = length
        self.born = born

    @classmethod
    def _get_view(cls) -> type[PolymerView]:
        """Return the view of a polymer, `PolymerView`."""
        return PolymerView


class PolymerJoiningInteraction(Interaction):
    """Joining of two polymers of an assembly pool into a new one.

    The polymers are joined at random or by reuse of the newest polymer.

    At every step, undirected exploration joins two polymers drawn at random from the pool, and directed exploration
    joins the polymer made most recently with one drawn at random. The joined polymer, of the summed length, is added
    to the pool, and the two polymers it joins stay in it.

    Args:
        label:
            The label of the interaction.

        polymers:
            The path of the variable group of polymers.

        mode:
            `'undirected'` or `'directed'`.

    Raises:
        ValidationError:
            The mode is unknown.

    Examples:
        >>> from skcomplex.systems.biology.evolution import PolymerJoiningInteraction
        >>> PolymerJoiningInteraction('joining', polymers='pool', mode='directed').mode
        'directed'
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, polymers: Role, mode: str = 'undirected') -> None:
        super().__init__(label)
        self.polymers = polymers
        self.mode = mode

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the joining, a rule of the whole system.

        Args:
            context:
                The namespace of the system.

        Raises:
            ValidationError:
                The mode is unknown.
        """
        super()._prepare(context)
        if self.mode not in MODES:
            error_msg = f'The mode must be one of {", ".join(MODES)}, not {self.mode!r}.'
            raise ValidationError(error_msg)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context, self.join)]

    def join(self, system: SystemView) -> None:
        """Join two polymers and add the product to the pool.

        Args:
            system:
                The system.
        """
        pool = system.get_agents(self.polymers, PolymerView)
        partner = pool[int(system.rng.integers(len(pool)))]
        if self.mode == 'directed':
            first = max(pool, key=lambda polymer: polymer.born)
        else:
            first = pool[int(system.rng.integers(len(pool)))]
        born = max(polymer.born for polymer in pool) + 1
        system.create(self.polymers, length=first.length + partner.length, born=born)


class AssemblyGrowthSystem(BaseSystem):
    """The numbers of objects at successive assembly indices.

    Each level is discovered from the level below.

    The objects of assembly index `a + 1` grow as `dN_{a+1}/dt = k (N_a)^alpha` from a constant pool of building blocks.
    Without selection, `alpha = 1`, every object built before is reused and `N_a(t) = (k t)^a / a!`, the exponential
    expansion of the assembly possible. Selection, `alpha < 1`, reuses only a subset, so the discovery of new objects
    slows down and the deeper levels fill far later. The system has the numbers `levels.N` of the group `levels`,
    ordered by assembly index from 1.

    Args:
        levels:
            The number of assembly indices above the building blocks.

        k:
            The rate of discovery.

        alpha:
            The selectivity of reuse, 1 for no selection.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import AssemblyGrowthSystem
        >>> AssemblyGrowthSystem(5).prepare().system_.elements[0].size
        5
    """

    field = 'biology/evolution'
    topics = ('assembly theory', 'selection', 'open-ended evolution', 'ode')
    references = ('Sharma et al. 2023',)
    until = 10.0
    dt = 0.1

    def __init__(
        self,
        levels: int = 10,
        *,
        k: float = 1.0,
        alpha: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.levels = levels
        self.k = k
        self.alpha = alpha

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the numbers of objects at successive assembly indices.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        group = AssemblyLevelElement('levels', k=self.k, alpha=self.alpha, size=self.levels)
        discovery = AssemblyDiscoveryInteraction('discovery', target='levels')
        self.system_ = System(self.label, elements=[group], interactions=[discovery])


class PolymerAssemblyPoolSystem(BaseSystem):
    """An assembly pool of linear polymers grown from one monomer.

    The pool grows by joining pairs of its polymers.

    Undirected exploration joins two random polymers of the pool, and directed exploration, a model of selection, joins
    the newest polymer with a random one. Undirected joining fills the pool with many short polymers, while directed
    joining builds ever longer chains, of a higher assembly index, about the base 2 logarithm of the length. The system
    has the lengths `pool.length` of the variable group `pool`.

    Args:
        mode:
            `'undirected'` or `'directed'`.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import PolymerAssemblyPoolSystem
        >>> PolymerAssemblyPoolSystem('directed').prepare().system_.interactions[0].mode
        'directed'
    """

    field = 'biology/evolution'
    topics = ('assembly theory', 'selection', 'polymers', 'agents')
    references = ('Sharma et al. 2023',)
    until = 200

    def __init__(
        self,
        mode: str = 'undirected',
        *,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.mode = mode

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set an assembly pool of linear polymers grown from one monomer.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        pool = PolymerElement('pool', size=1)
        self.system_ = System(
            self.label,
            elements=[pool],
            interactions=[PolymerJoiningInteraction('joining', polymers='pool', mode=self.mode)],
        )
