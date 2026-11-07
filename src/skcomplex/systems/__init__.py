"""Ready-made systems of every science, built from the core and organized by field and subfield."""

from ._catalog import SYSTEMS, all_systems, catalog, create_simulation, create_system, describe, register_system
from ._extras import import_extra
from ._system_catalog import SystemCatalog
from ._system_entry import SystemEntry
from ._system_listing import SystemListing

__all__: list[str] = [
    'SYSTEMS',
    'SystemCatalog',
    'SystemEntry',
    'SystemListing',
    'all_systems',
    'catalog',
    'create_simulation',
    'create_system',
    'describe',
    'import_extra',
    'register_system',
]
