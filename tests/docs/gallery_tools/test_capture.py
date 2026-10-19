"""Test the capture of screenshots with both backends, which needs a graphics adapter and, for the web, Chromium."""

import doctest
import importlib
import os
import sys
from pathlib import Path

import gallery_tools
import numpy as np
import pytest
import rerun.blueprint as rrb
from gallery_tools import (
    CaptureSettings,
    GalleryError,
    capture_thumbnail,
    find_problems,
    thumbnail_blueprint,
    view_boxes,
)
from PIL import Image

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create

ROOT = Path(__file__).parent.parent.parent.parent
CAPTURE = sys.platform == 'darwin' or os.environ.get('SKCOMPLEX_GALLERY_CAPTURE') == '1'
MODULES = ['_build', '_cache', '_capture', '_checks', '_examples', '_runner', '_scraper', '_screenshots', '_settings']


def _web_available():
    """Return whether Playwright and its Chromium are installed."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
    except ImportError:
        return False
    with sync_playwright() as playwright:
        return Path(playwright.chromium.executable_path).exists()


BACKENDS = [
    pytest.param('native', marks=pytest.mark.skipif(not CAPTURE, reason='needs a graphics adapter')),
    pytest.param('web', marks=pytest.mark.skipif(not (CAPTURE and _web_available()), reason='needs Chromium')),
]


@pytest.fixture(scope='module')
def recording(tmp_path_factory):
    """Return a Gray-Scott recording and its blueprint, with the `v` field as the primary view."""
    path = tmp_path_factory.mktemp('recording') / 'gray_scott.rrd'
    system = create('gray_scott', shape=(64, 64), random_state=0)
    observer = RerunObserver('gray_scott', save_path=path, blueprint=default_blueprint(system, primary='cells/v'))
    Simulation(system, dt=20.0, observers=[observer]).run(until=2000)
    return path, observer.blueprint_


@pytest.mark.parametrize('name', MODULES)
def test_gallery_tools_docstring_examples_pass(name, monkeypatch):
    """Test that the docstring examples of a module of the gallery tooling pass."""
    monkeypatch.chdir(ROOT)
    module = importlib.import_module(f'gallery_tools.{name}')
    assert doctest.testmod(module, optionflags=doctest.ELLIPSIS).failed == 0


def test_thumbnail_blueprint_keeps_primary_view_alone():
    """Test that the thumbnail shows the first view without its title bar and hides the panels."""
    blueprint = rrb.Blueprint(rrb.Horizontal(rrb.Spatial2DView(origin='/v'), rrb.Spatial2DView(origin='/u')))
    thumbnail = thumbnail_blueprint(blueprint)
    view = thumbnail.root_container.contents[0]
    assert (str(view.origin), view.titlebar) == ('/v', False)
    assert thumbnail.time_panel.state == 'hidden'


def test_capture_thumbnail_rejects_unknown_backend(tmp_path, recording):
    """Test that an unknown backend is named."""
    path, blueprint = recording
    with pytest.raises(GalleryError, match='vnc'):
        capture_thumbnail(path, blueprint, 'gray_scott', tmp_path, backend='vnc')


@pytest.mark.parametrize('backend', BACKENDS)
def test_capture_thumbnail_shows_field_at_last_frame(tmp_path, recording, backend):
    """Test that the thumbnail passes the checks, with the spots of the last frame and no notification."""
    path, blueprint = recording
    settings = CaptureSettings()
    thumbnail = capture_thumbnail(path, blueprint, 'gray_scott', tmp_path, backend=backend, settings=settings)
    assert thumbnail.stat().st_size <= settings.max_png_bytes
    part = thumbnail_blueprint(blueprint).root_container
    with Image.open(thumbnail) as image:
        assert image.format == 'PNG'
        assert find_problems(image, settings.thumbnail_size, view_boxes(part, settings.thumbnail_size)) == []


@pytest.mark.parametrize('backend', BACKENDS)
def test_capture_thumbnail_is_reproducible(tmp_path, recording, backend):
    """Test that two captures of one recording differ by less than 2% of the pixel range."""
    path, blueprint = recording
    images = []
    for attempt in ('first', 'second'):
        thumbnail = capture_thumbnail(path, blueprint, 'gray_scott', tmp_path / attempt, backend=backend)
        with Image.open(thumbnail) as image:
            images.append(np.asarray(image.convert('RGB'), dtype=float))
    assert np.mean(np.abs(images[0] - images[1])) / 255 < 0.02


def test_gallery_tools_exports_its_surface():
    """Test that the package re-exports every name of its surface."""
    assert all(hasattr(gallery_tools, name) for name in gallery_tools.__all__)
