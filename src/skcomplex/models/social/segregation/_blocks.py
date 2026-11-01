"""Block segregation models: stochastic moves between blocks of a city driven by the utility of their densities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import MapInteraction
from ....core.spaces import DiscreteTime, LatticeSpace
from ....core.system import System
from ....specs import PhaseSpec, PhasesScheduleSpec
from ..._catalog import register_model
from ._block_moves import BlockMovesInteraction
from ._dweller import DwellerElement

PHASE = 'default'


@register_model(
    field='social/segregation',
    tags=('segregation', 'agents', 'grid', 'stochastic'),
    references=('Grauwin et al. 2009', 'Jensen 2023'),
    defaults={'until': 50},
)
def stochastic_schelling(
    width: int = 20,
    height: int = 20,
    *,
    side: int = 5,
    density: float = 0.5,
    beta: float = 3.0,
    altruism: float = 0.5,
    slope: float = -4.0 / 3.0,
    intercept: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the stochastic Schelling model of agents moving between blocks, which segregates above `beta = 2`.

    Args:
        width:
            The number of columns, a multiple of the block side.

        height:
            The number of rows, a multiple of the block side.

        side:
            The side of a square block in cells.

        density:
            The share of cells with an agent.

        beta:
            The inverse temperature of the moves.

        altruism:
            The weight of the change of the global utility.

        slope:
            The slope of the utility in the block density.

        intercept:
            The intercept of the utility.

        random_state:
            The seed or generator of the initial cells.

    Returns:
        system:
            The model, with the group `dwellers`.

    Examples:
        >>> from skcomplex.models.social.segregation import stochastic_schelling
        >>> stochastic_schelling(8, 8, side=4, random_state=0).elements[0].check().prepare().size_
        32
    """
    rng = np.random.default_rng(random_state)
    size = int(width * height * density)
    cells = rng.permutation(width * height)[:size]
    locations = [(int(cell % width), int(cell // width)) for cell in cells]
    blocks = [BlockMovesInteraction._block_of(location, side, width // side) for location in locations]
    dwellers = GroupElement(
        'dwellers',
        element=DwellerElement,
        size=size,
        values={
            'block': np.array(blocks, dtype=float),
            'beta': beta,
            'altruism': altruism,
            'slope': slope,
            'intercept': intercept,
        },
        location=locations,
    )
    moves = BlockMovesInteraction('moves', target='dwellers', width=width, height=height, side=side, phase=PHASE)
    block = MapInteraction(
        'block',
        target='dwellers',
        state='block',
        function=moves.get_block,
        schedule='member',
        form='agent',
        phase=PHASE,
    )
    return System(
        elements=[dwellers],
        interactions=[moves, block],
        space=LatticeSpace((width, height), boundary='fixed', capacity=1),
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec(PHASE, activation='visits', groups=('dwellers',))])),
    )
