"""Reaction kinetics shared by the biology packages: reactions whose propensity saturates in their substrate."""

from ._michaelis_menten import MichaelisMentenInteraction

__all__: list[str] = [
    'MichaelisMentenInteraction',
]
