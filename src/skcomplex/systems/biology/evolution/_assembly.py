"""Evolution systems: assembly theory.

Assembly theory is the discovery of new objects by recursive joining of existing ones.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw, RuleLaw
from ....core.components.laws.utils import shift_member
from ....core.components.spaces import BaseSpace, DiscreteTime
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

MODES = ('undirected', 'directed')


class AssemblyLevelElement(Element):
    """Objects of one assembly index.

    They are discovered from the objects one assembly step below them.

    In a group ordered by assembly index, the objects of index `a + 1` grow as `dN_{a+1}/dt = k (N_a)^alpha`, Eq. (3)
    of Sharma et al. (2022), where the first member reads the constant pool `N_0` of building blocks. The selectivity
    `alpha = 1` reuses every object built before, and `alpha < 1` only a subset, the signature of selection.

    Args:
        label:
            The label of the level or of the group of levels.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        N:
            The initial number of objects.

        k:
            The rate of discovery.

        alpha:
            The selectivity of reuse.

        N0:
            The constant number of building blocks below the first level.

    Examples:
        >>> from skcomplex.systems.biology.evolution import AssemblyLevelElement
        >>> AssemblyLevelElement('levels', alpha=0.5).alpha
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        N0: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.N = N
        self.k = k
        self.alpha = alpha
        self.N0 = N0

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the discovery of the objects of this level from the level below.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        below = shift_member(s.N, -1, boundary='fixed', fill=self.N0)
        return [RateLaw(s.N, s.k * below**s.alpha)]


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

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        length:
            The number of monomers.

        born:
            The step at which the polymer was made.

    Examples:
        >>> from skcomplex.systems.biology.evolution import PolymerElement
        >>> PolymerElement('polymer', length=3.0).length
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        length: Annotated[float, StateSpec(domain=NN)] = 1.0,
        born: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
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
    to the pool, and its parts stay in it.

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

    law_classes = (RuleLaw,)

    def __init__(self, label: str, *, polymers: Annotated[str, RoleSpec()], mode: str = 'undirected') -> None:
        super().__init__(label)
        self.polymers = polymers
        self.mode = mode

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Check the mode.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The mode is unknown.
        """
        super()._prepare(context)
        if self.mode not in MODES:
            error_msg = f'The mode must be one of {", ".join(MODES)}, not {self.mode!r}.'
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the joining rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.join)]

    def join(self, system: SystemView) -> None:
        """Join two polymers and add the product to the pool.

        Args:
            system:
                The system.
        """
        pool = system.agents(self.polymers, PolymerView)
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

    Examples:
        >>> from skcomplex.systems.biology.evolution import AssemblyGrowthSystem
        >>> AssemblyGrowthSystem(5).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.levels = levels
        self.k = k
        self.alpha = alpha

    def _build_system(self) -> System:
        """Return the parts of the numbers of objects at successive assembly indices."""
        group = GroupElement(
            'levels',
            element=AssemblyLevelElement,
            size=self.levels,
            values={'k': self.k, 'alpha': self.alpha},
        )
        return System(self.label, elements=[group])


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

        capacity:
            The largest number of polymers the pool holds.

        label:
            The label of the system, which names it when it is nested in another one.

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
        capacity: int = 2000,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.mode = mode
        self.capacity = capacity

    def _build_system(self) -> System:
        """Return the parts of the system."""
        pool = GroupElement('pool', element=PolymerElement, size=1, variable=True, capacity=self.capacity)
        return System(
            self.label,
            elements=[pool],
            interactions=[PolymerJoiningInteraction('joining', polymers='pool', mode=self.mode)],
            time=DiscreteTime(),
        )
