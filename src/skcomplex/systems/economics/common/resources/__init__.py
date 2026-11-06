"""Resource realizations shared by the economics packages: renewable resources that grow back."""

from ._regrowth import RegrowthInteraction

__all__: list[str] = [
    'RegrowthInteraction',
]
