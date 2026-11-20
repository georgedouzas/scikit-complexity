"""Define interactions, the components that change the states of elements.

An interaction reads the states of the elements its roles bind and returns laws that change them, such as a rate, a map,
a random event, or a transfer between members.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseInteraction
from ._interaction import Interaction
from ._random_partition import RandomPartition
from ._training import TrainingInteraction
from ._wiener_process_spec import WienerProcessSpec

__all__: list[str] = [
    'BaseInteraction',
    'Interaction',
    'RandomPartition',
    'TrainingInteraction',
    'WienerProcessSpec',
]
