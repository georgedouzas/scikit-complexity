"""Ready-made models of every science, built from the core and organized by field and subfield."""

from ..core.runtime import MODELS, ModelEntry, catalog, create, describe, register_model, run
from . import (
    biology,
    chemistry,
    computer_science,
    earth,
    economics,
    engineering,
    mathematics,
    physics,
    social,
)

__all__: list[str] = [
    'MODELS',
    'ModelEntry',
    'biology',
    'catalog',
    'chemistry',
    'computer_science',
    'create',
    'describe',
    'earth',
    'economics',
    'engineering',
    'mathematics',
    'physics',
    'register_model',
    'run',
    'social',
]
