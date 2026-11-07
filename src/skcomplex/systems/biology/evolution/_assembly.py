"""Evolution systems: assembly theory, the discovery of new objects by recursive joining of existing ones."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.components.elements import GroupElement
from ....core.components.spaces import DiscreteTime
from ....core.components.system import System
from ..._catalog import register_system
from ._assembly_level import AssemblyLevelElement
from ._polymer import PolymerElement
from ._polymer_joining import PolymerJoiningInteraction


@register_system(
    field='biology/evolution',
    tags=('assembly theory', 'selection', 'open-ended evolution', 'ode'),
    references=('Sharma et al. 2023',),
    defaults={'until': 10.0, 'dt': 0.1},
)
def assembly_growth(levels: int = 10, *, k: float = 1.0, alpha: float = 1.0) -> System:
    """Return the numbers of objects at successive assembly indices, each discovered from the level below.

    The objects of assembly index `a + 1` grow as `dN_{a+1}/dt = k (N_a)^alpha` from a constant pool of building
    blocks. Without selection, `alpha = 1`, every object built before is reused and `N_a(t) = (k t)^a / a!`, the
    exponential expansion of the assembly possible. Selection, `alpha < 1`, reuses only a subset, so the discovery of
    new objects slows down and the deeper levels fill far later.

    Args:
        levels:
            The number of assembly indices above the building blocks.

        k:
            The rate of discovery.

        alpha:
            The selectivity of reuse, 1 for no selection.

    Returns:
        system:
            The system, with the numbers `levels.N` of the group `levels`, ordered by assembly index from 1.

    Examples:
        >>> from skcomplex.systems.biology.evolution import assembly_growth
        >>> assembly_growth(5).elements[0].prepare().size_
        5
    """
    group = GroupElement('levels', element=AssemblyLevelElement, size=levels, values={'k': k, 'alpha': alpha})
    return System(elements=[group])


@register_system(
    field='biology/evolution',
    tags=('assembly theory', 'selection', 'polymers', 'agents'),
    references=('Sharma et al. 2023',),
    defaults={'until': 200},
)
def polymer_assembly_pool(mode: str = 'undirected', *, capacity: int = 2000) -> System:
    """Return an assembly pool of linear polymers grown from one monomer by joining pairs of its polymers.

    Undirected exploration joins two random polymers of the pool, and directed exploration, a model of selection,
    joins the newest polymer with a random one. Undirected joining fills the pool with many short polymers, while
    directed joining builds ever longer chains, of a higher assembly index, about the base 2 logarithm of the length.

    Args:
        mode:
            `'undirected'` or `'directed'`.

        capacity:
            The largest number of polymers the pool holds.

    Returns:
        system:
            The system, with the lengths `pool.length` of the variable group `pool`.

    Examples:
        >>> from skcomplex.systems.biology.evolution import polymer_assembly_pool
        >>> polymer_assembly_pool('directed').interactions[0].mode
        'directed'
    """
    pool = GroupElement('pool', element=PolymerElement, size=1, variable=True, capacity=capacity)
    return System(
        elements=[pool],
        interactions=[PolymerJoiningInteraction('joining', polymers='pool', mode=mode)],
        time=DiscreteTime(),
    )
