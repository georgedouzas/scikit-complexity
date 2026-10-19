"""Heat transfer: diffusion of heat on lattices and meshes, and cooling towards the surroundings."""

from ._cooling import CoffeeCup, cooling_coffee
from ._heat import Heat, harmonic, heat_diffusion, heat_l_shape

__all__: list[str] = [
    'CoffeeCup',
    'Heat',
    'cooling_coffee',
    'harmonic',
    'heat_diffusion',
    'heat_l_shape',
]
