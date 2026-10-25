"""Heat transfer: diffusion of heat on lattices and meshes, and cooling towards the surroundings."""

from ._cooling import CoffeeCupComponent, cooling_coffee
from ._heat import HeatComponent, harmonic, heat_diffusion, heat_l_shape

__all__: list[str] = [
    'CoffeeCupComponent',
    'HeatComponent',
    'cooling_coffee',
    'harmonic',
    'heat_diffusion',
    'heat_l_shape',
]
