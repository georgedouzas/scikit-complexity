"""Processes: stocks of material in engineered flows."""

from ._processes import bathtub, cascade_classifier
from ._systems import food_relief, irrigation, rice_milling, solid_waste

__all__: list[str] = [
    'bathtub',
    'cascade_classifier',
    'food_relief',
    'irrigation',
    'rice_milling',
    'solid_waste',
]
