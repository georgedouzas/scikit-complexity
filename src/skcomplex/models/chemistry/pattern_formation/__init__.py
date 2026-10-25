"""Pattern formation: amplitude equations and rolls."""

from ._reaction_diffusion import (
    BrusselatorComponent,
    GrayScottComponent,
    SchnakenbergComponent,
    brusselator,
    gray_scott,
    schnakenberg,
)

__all__: list[str] = [
    'BrusselatorComponent',
    'GrayScottComponent',
    'SchnakenbergComponent',
    'brusselator',
    'gray_scott',
    'schnakenberg',
]
