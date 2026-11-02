"""Ecology models: population growth, predation, and invasion fronts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import LifeCycleInteraction, MovementInteraction, TransferInteraction
from ....core.laws import Sink, Source
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ..._catalog import register_model
from ._grass import GrassElement
from ._grass_view import GrassView
from ._sheep import SheepElement
from ._sheep_view import SheepView
from ._wolf import WolfElement
from ._wolf_view import WolfView

GROWN_SHARE = 0.5


def _get_gain(food: AgentViewProtocol, animal: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return the energy `gain` an animal draws from its food."""
    return float(animal.gain) if isinstance(animal, SheepView | WolfView) else 0.0


def _is_grown(patch: AgentViewProtocol, sheep: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether the grass of a patch is grown, so a sheep eats it."""
    return isinstance(patch, GrassView) and bool(patch.grown)


def _is_starved(animal: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether an animal has run out of energy, so it dies."""
    return isinstance(animal, SheepView | WolfView) and animal.energy < 0


def _get_reproduction(animal: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return the probability `reproduction` that an animal gives birth in a step."""
    return float(animal.reproduction) if isinstance(animal, SheepView | WolfView) else 0.0


def _get_animal(
    group: str,
    food: str,
    removed: bool,
) -> list[MovementInteraction | TransferInteraction | LifeCycleInteraction]:
    """Return the member interactions of an animal: wander at a unit cost of energy, eat in its cell, and live."""
    cost = TransferInteraction(
        f'{group}_cost',
        source=group,
        target=group,
        state='energy',
        into=Sink(),
        amount=1.0,
        form='agent',
    )
    eating = TransferInteraction(
        f'{group}_eating',
        source=food,
        target=group,
        state=Source(),
        into='energy',
        amount=_get_gain,
        legs=() if removed else (('grown', Sink(), 1.0),),
        guard=None if removed else _is_grown,
        pairing='cell',
        form='agent',
        schedule='member',
        actor='target',
        partners=1,
        remove='source' if removed else None,
    )
    return [
        MovementInteraction(f'{group}_walk', target=group, cost=cost, schedule='member'),
        eating,
        LifeCycleInteraction(
            f'{group}_life',
            target=group,
            death=_is_starved,
            birth=_get_reproduction,
            split=('energy',),
            schedule='member',
        ),
    ]


@register_model(
    field='biology/ecology',
    tags=('predator-prey', 'agents', 'grid'),
    references=('Wilensky 1997',),
    defaults={'until': 100},
)
def wolf_sheep(
    width: int = 20,
    height: int = 20,
    *,
    sheep: int = 100,
    wolves: int = 25,
    regrowth: float = 30.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the wolf and sheep predation model with one grass patch per cell.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        sheep:
            The initial number of sheep.

        wolves:
            The initial number of wolves.

        regrowth:
            The number of steps grass takes to regrow.

        random_state:
            The seed or generator of the initial grass state.

    Returns:
        system:
            The model, with the groups `grass`, `sheep`, and `wolves`.

    Examples:
        >>> from skcomplex.models.biology.ecology import wolf_sheep
        >>> [group.label for group in wolf_sheep(10, 10).elements]
        ['grass', 'sheep', 'wolves']
    """
    rng = np.random.default_rng(random_state)
    cells = [(column, row) for column in range(width) for row in range(height)]
    grown = (rng.random(len(cells)) < GROWN_SHARE).astype(float)
    countdown = np.where(grown > 0, regrowth, rng.uniform(0, regrowth, len(cells)))
    grass = GroupElement(
        'grass',
        element=GrassElement,
        size=len(cells),
        values={'grown': grown, 'countdown': countdown, 'regrowth': regrowth},
        variable=True,
        location=cells,
    )
    return System(
        elements=[
            grass,
            GroupElement('sheep', element=SheepElement, size=sheep, variable=True),
            GroupElement('wolves', element=WolfElement, size=wolves, variable=True),
        ],
        interactions=[*_get_animal('sheep', 'grass', removed=False), *_get_animal('wolves', 'sheep', removed=True)],
        space=LatticeSpace((width, height), boundary='periodic'),
    )
