"""Test the thumbnails of finished observers."""

import warnings

import pytest
from gallery_tools import (
    Budget,
    CaptureSettings,
    GalleryWarning,
    ScreenshotCache,
    cache_key,
    model_sources,
    thumbnail_of,
)
from PIL import Image

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

CAPTION = '# The observer saves every frame, and the primary view is the time series of `n` and `{name}`.\nMODEL = 1\n'


@pytest.mark.parametrize(('name', 'warned'), [('ghost', True), ('n', False)])
def test_thumbnail_of_warns_about_series_caption_names(tmp_path, name, warned):
    """Test that a thumbnail whose primary view lacks a series its caption names warns, and one that has it does not."""
    example = tmp_path / 'plot_lotka_volterra.py'
    example.write_text(CAPTION.format(name=name))
    observer = RerunObserver('lotka_volterra', save_path=tmp_path / 'lotka_volterra.rrd', blueprint='default')
    Simulation(create('lotka_volterra'), dt=0.5, observers=[observer]).run(until=2)
    settings, cache = CaptureSettings(), ScreenshotCache(tmp_path / 'cache')
    Image.new('RGB', (12, 8), 'red').save(tmp_path / 'thumbnail.png')
    cache.put(cache_key(example, model_sources('lotka_volterra'), 'native', settings), tmp_path / 'thumbnail.png', {})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        thumbnail_of(
            example,
            'lotka_volterra',
            observer,
            cache,
            backend='native',
            settings=settings,
            budget=Budget(),
        )
    messages = [str(warning.message) for warning in caught if issubclass(warning.category, GalleryWarning)]
    assert messages == (
        [f'plot_lotka_volterra: the primary view shows no series `{name}`, which its caption names.'] if warned else []
    )
