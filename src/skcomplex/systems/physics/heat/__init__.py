"""Heat transfer systems.

The package covers diffusion of heat on lattices and meshes, and cooling towards the surroundings.
"""

from ._cooling import CoffeeCupElement, CoolingCoffeeSystem
from ._heat import HeatDiffusionSystem, HeatElement, HeatLShapeSystem, harmonic

__all__: list[str] = [
    'CoffeeCupElement',
    'CoolingCoffeeSystem',
    'HeatDiffusionSystem',
    'HeatElement',
    'HeatLShapeSystem',
    'harmonic',
]
