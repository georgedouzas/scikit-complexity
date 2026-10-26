"""Systems as components, interactions, space, and time, and the observables that measure them."""

from ._ensembles import replicate
from ._observables import Observable, observe_element, reduce_values
from ._systems import System

__all__: list[str] = [
    'Observable',
    'System',
    'observe_element',
    'reduce_values',
    'replicate',
]
