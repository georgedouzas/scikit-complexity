"""Ready-made models as registered factories, built from the general interface, with the catalog that lists them."""

from ..runtime import MODELS, ModelEntry, catalog, create, describe, register_model, run
from . import ecology, economics, epidemiology, finance, lattices, neuroscience, nonlinear, physics, social

__all__: list[str] = [
    'MODELS',
    'ModelEntry',
    'catalog',
    'create',
    'describe',
    'ecology',
    'economics',
    'epidemiology',
    'finance',
    'lattices',
    'neuroscience',
    'nonlinear',
    'physics',
    'register_model',
    'run',
    'social',
]
