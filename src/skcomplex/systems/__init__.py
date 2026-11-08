"""Ready-made systems of every science, built from the core and organized by field and subfield."""

from ._catalog import all_systems
from ._extras import import_extra

__all__: list[str] = [
    'all_systems',
    'import_extra',
]
