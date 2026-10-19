"""Heat transfer: diffusion of heat on lattices and meshes."""

from ._heat import Heat, harmonic, heat_diffusion, heat_l_shape

__all__: list[str] = [
    'Heat',
    'harmonic',
    'heat_diffusion',
    'heat_l_shape',
]
