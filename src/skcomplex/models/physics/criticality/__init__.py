"""Self-organized criticality: sandpiles, forest fires, and evolutionary learning at the edge of ergodicity breaking."""

from ._criticality import (
    Forest,
    MannaCell,
    SandpileCell,
    SpreadingForest,
    forest_fire,
    hexagonal_forest_fire,
    manna_sandpile,
    sandpile,
)
from ._evolutionary_learning import Learner, evolutionary_sk_game

__all__: list[str] = [
    'Forest',
    'Learner',
    'MannaCell',
    'SandpileCell',
    'SpreadingForest',
    'evolutionary_sk_game',
    'forest_fire',
    'hexagonal_forest_fire',
    'manna_sandpile',
    'sandpile',
]
