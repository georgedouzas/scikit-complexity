"""Resource realizations shared by the economics packages.

They are renewable resources that grow back.
"""

from ._regrowth import RegrowthInteraction

__all__: list[str] = [
    'RegrowthInteraction',
]
