"""Heat transfer: diffusion of heat on lattices and meshes, and cooling towards the surroundings."""

from ._cooling import CoffeeCupElement, cooling_coffee
from ._heat import HeatElement, harmonic, heat_diffusion, heat_l_shape

__all__: list[str] = [
    'CoffeeCupElement',
    'HeatElement',
    'cooling_coffee',
    'harmonic',
    'heat_diffusion',
    'heat_l_shape',
]
