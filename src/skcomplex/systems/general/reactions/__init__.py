"""Define the reactions and reaction networks that several fields share.

A reaction moves units between states, and a reaction network builds one element per species and one reaction per row of
its table.
"""

from ._reaction import ReactionInteraction
from ._reaction_network import ReactionNetworkSystem

__all__: list[str] = [
    'ReactionInteraction',
    'ReactionNetworkSystem',
]
