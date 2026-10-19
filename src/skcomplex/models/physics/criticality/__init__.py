"""Self-organized criticality: sandpiles and forest fires."""

from ._criticality import Forest, SandpileCell, forest_fire, sandpile

__all__: list[str] = [
    'Forest',
    'SandpileCell',
    'forest_fire',
    'sandpile',
]
