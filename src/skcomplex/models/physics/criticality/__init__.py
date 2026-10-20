"""Self-organized criticality: deterministic and stochastic sandpiles, and forest fires on lattices and graphs."""

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

__all__: list[str] = [
    'Forest',
    'MannaCell',
    'SandpileCell',
    'SpreadingForest',
    'forest_fire',
    'hexagonal_forest_fire',
    'manna_sandpile',
    'sandpile',
]
