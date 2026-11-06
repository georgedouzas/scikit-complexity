"""Load the synthetic island of the Kauai population system, whose layers are GeoJSON files in metres."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import json
from importlib.resources import files

from shapely import Polygon
from shapely.geometry import shape

PACKAGE = 'skcomplex.systems.social.organizations'
ISLAND_LAYERS = ('boundary', 'census', 'places', 'facilities')


def _read_layer(layer: str) -> str:
    """Return the GeoJSON text of a layer of the synthetic island."""
    return (files(PACKAGE) / f'island_{layer}.geojson').read_text()


def load_island_layers() -> dict[str, list[Polygon]]:
    """Return the polygons of every layer of the synthetic island, in a metric coordinate system.

    Returns:
        layers:
            The polygons of the layers `boundary`, `census`, `places`, and `facilities`, by the name of the layer.

    Examples:
        >>> from skcomplex.systems.social.organizations import load_island_layers
        >>> {layer: len(polygons) for layer, polygons in load_island_layers().items()}
        {'boundary': 1, 'census': 3, 'places': 3, 'facilities': 2}
    """
    return {
        layer: [shape(item['geometry']) for item in json.loads(_read_layer(layer))['features']]
        for layer in ISLAND_LAYERS
    }


def load_island_populations() -> list[int]:
    """Return the number of residents of every region of the census layer of the synthetic island.

    Returns:
        populations:
            The population of every census region, in the order of the regions of the layer.

    Examples:
        >>> from skcomplex.systems.social.organizations import load_island_populations
        >>> load_island_populations()
        [100, 50, 0]
    """
    return [int(item['properties']['population']) for item in json.loads(_read_layer('census'))['features']]
