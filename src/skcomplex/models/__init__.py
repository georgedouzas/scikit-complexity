"""Ready-made models of every science, built from the core and organized by field and subfield."""

from ._catalog import MODELS, catalog, create, describe, register_model, run
from ._model_catalog import ModelCatalog
from ._model_entry import ModelEntry
from ._model_listing import ModelListing

__all__: list[str] = [
    'MODELS',
    'ModelCatalog',
    'ModelEntry',
    'ModelListing',
    'catalog',
    'create',
    'describe',
    'register_model',
    'run',
]
