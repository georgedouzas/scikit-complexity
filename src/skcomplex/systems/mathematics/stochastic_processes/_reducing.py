"""Stochastic process systems: sample-space-reducing processes and fragmentation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.elements import GroupElement
from ....core.components.interactions import LifeCycleInteraction
from ....core.components.system import System
from ..._catalog import register_system
from ._fragment import FragmentElement
from ._reducing_ball import ReducingBallElement
from ._reducing_jump import ReducingJumpInteraction


@register_system(
    field='mathematics/stochastic_processes',
    tags=('sample space reduction', 'agents', 'zipf', 'power law'),
    references=('Corominas-Murtra, Hanel, and Thurner 2015', 'Thurner et al. 2018'),
    defaults={'until': 100},
)
def sample_space_reducing(n_balls: int = 20, *, n_states: int = 10, lam: float = 1.0) -> System:
    """Return sample-space-reducing processes, whose state visits follow Zipf's law `1 / i` for `lam = 1`.

    Args:
        n_balls:
            The number of independent balls, all started at the top state.

        n_states:
            The number of states.

        lam:
            The probability of a reducing jump, the exponent of the visiting distribution.

    Returns:
        system:
            The system, with the group `balls` of state `level`.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import sample_space_reducing
        >>> sample_space_reducing(4).elements[0].prepare().size_
        4
    """
    return System(
        elements=[
            GroupElement(
                'balls',
                element=ReducingBallElement,
                size=n_balls,
                values={'level': float(n_states), 'n_states': float(n_states), 'lam': lam},
                variable=True,
            ),
        ],
        interactions=[ReducingJumpInteraction('jump', target='balls')],
    )


@register_system(
    field='mathematics/stochastic_processes',
    tags=('fragmentation', 'agents', 'sample space reduction'),
    references=('Thurner et al. 2018',),
    defaults={'until': 30},
)
def fragmentation(size: int = 1000, *, rate: float = 0.5) -> System:
    """Return the random fragmentation of an object of integer size, whose fragments keep breaking into smaller ones.

    The total size is conserved, and every fragment ends at size 1.

    Args:
        size:
            The size of the initial object.

        rate:
            The probability that a fragment larger than 1 breaks in a step.

    Returns:
        system:
            The system, with the group `fragments` of state `length`.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import fragmentation
        >>> fragmentation(50).elements[0].prepare().size_
        1
    """
    return System(
        elements=[
            GroupElement(
                'fragments',
                element=FragmentElement,
                size=1,
                values={'length': float(size), 'rate': rate},
                variable=True,
            ),
        ],
        interactions=[
            LifeCycleInteraction(
                'breaking',
                target='fragments',
                birth=FragmentElement.breaks,
                split=('length',),
                keep=FragmentElement.cut,
            ),
        ],
    )
