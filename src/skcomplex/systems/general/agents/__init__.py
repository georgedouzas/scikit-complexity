"""Agent rules that systems of several fields share.

Agents move, give birth and die, choose, imitate, and evolve their strategies.
"""

from ._birth_death import BirthDeathInteraction
from ._choice import ChoiceInteraction
from ._evolution import EvolutionInteraction
from ._imitation import ImitationInteraction
from ._movement import MovementInteraction

__all__: list[str] = [
    'BirthDeathInteraction',
    'ChoiceInteraction',
    'EvolutionInteraction',
    'ImitationInteraction',
    'MovementInteraction',
]
