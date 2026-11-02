"""Pattern formation: amplitude equations and rolls."""

from ._brusselator import BrusselatorElement
from ._gray_scott import GrayScottElement
from ._reaction_diffusion import brusselator, gray_scott, schnakenberg
from ._schnakenberg import SchnakenbergElement

__all__: list[str] = [
    'BrusselatorElement',
    'GrayScottElement',
    'SchnakenbergElement',
    'brusselator',
    'gray_scott',
    'schnakenberg',
]
