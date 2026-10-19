"""Test the cache key, the screenshot cache, and the selection of the examples a build runs."""

import dataclasses
import shutil
from pathlib import Path

import gallery_tools
import pytest
from gallery_tools import (
    CaptureSettings,
    GalleryBuild,
    ScreenshotCache,
    cache_key,
    find_examples,
    model_sources,
)
from PIL import Image

DOCS = Path(__file__).parent.parent.parent.parent / 'docs'
EXAMPLE = DOCS / 'examples' / 'biology' / 'ecology' / 'plot_lotka_volterra.py'


@pytest.fixture
def example(tmp_path: Path) -> Path:
    """Return a copy of the Lotka-Volterra example."""
    path = tmp_path / EXAMPLE.name
    shutil.copyfile(EXAMPLE, path)
    return path


@pytest.fixture
def thumbnail(tmp_path: Path) -> Path:
    """Return a small thumbnail."""
    Image.new('RGB', (12, 8), 'red').save(tmp_path / 'thumbnail.png')
    return tmp_path / 'thumbnail.png'


def test_cache_key_depends_on_example_backend_and_settings(example):
    """Test that the key changes with the example source, the backend, and the capture settings."""
    settings, sources = CaptureSettings(), model_sources('lotka_volterra')
    key = cache_key(example, sources, 'native', settings)
    assert key == cache_key(example, sources, 'native', settings)
    assert key != cache_key(example, sources, 'web', settings)
    assert key != cache_key(example, sources, 'native', dataclasses.replace(settings, wait=1.0))
    example.write_text(example.read_text() + '\n')
    assert key != cache_key(example, sources, 'native', settings)


def test_cache_key_depends_on_model_sources(example, tmp_path):
    """Test that a change of the sources of a subfield changes the key of its examples only."""
    ecology, epidemiology = tmp_path / 'ecology.py', tmp_path / 'epidemiology.py'
    ecology.write_text('x = 1\n')
    epidemiology.write_text('y = 1\n')
    keys = [cache_key(example, [source], 'native', CaptureSettings()) for source in (ecology, epidemiology)]
    ecology.write_text('x = 2\n')
    assert cache_key(example, [ecology], 'native', CaptureSettings()) != keys[0]
    assert cache_key(example, [epidemiology], 'native', CaptureSettings()) == keys[1]


def test_screenshot_cache_stores_and_returns_thumbnails(tmp_path, thumbnail):
    """Test that a stored thumbnail is returned under its key and a missing one is not."""
    cache = ScreenshotCache(tmp_path / 'cache')
    assert cache.get('key') is None
    stored = cache.put('key', thumbnail, {'model': 'lotka_volterra'})
    assert cache.get('key') == stored
    assert stored.read_bytes() == thumbnail.read_bytes()


def test_gallery_build_runs_curated_examples_by_default():
    """Test that a default build runs the curated examples and a full build runs all of them."""
    examples, gallery = DOCS / 'examples', DOCS / 'generated' / 'gallery'
    curated = GalleryBuild(examples, gallery).selected()
    assert {example.model for example in curated} >= {'lotka_volterra', 'gray_scott', 'sir'}
    assert all(not example.scaffolded for example in curated)
    assert len(GalleryBuild(examples, gallery, run_all=True).selected()) == len(find_examples(examples))
    assert 'plot_lorenz\\.py' in GalleryBuild(examples, gallery).filename_pattern()


def test_gallery_build_marks_stale_examples(tmp_path, example, thumbnail):
    """Test that an example whose page shows other screenshots loses its checksum, and an up to date one keeps it."""
    examples, gallery = tmp_path / 'examples', tmp_path / 'gallery'
    script = examples / 'biology' / 'ecology' / example.name
    script.parent.mkdir(parents=True)
    shutil.copyfile(example, script)
    build = GalleryBuild(examples, gallery)
    checksum = gallery / 'biology' / 'ecology' / f'{example.name}.md5'
    checksum.parent.mkdir(parents=True)
    checksum.write_text('md5')
    key = cache_key(script, model_sources('lotka_volterra'), build.backend, build.settings)
    build.cache.put(key, thumbnail, {})
    assert [stale.model for stale in build.mark_stale()] == ['lotka_volterra']
    checksum.write_text('md5')
    build.stamp(find_examples(examples)[0]).write_text(key)
    assert build.mark_stale() == []
    assert checksum.exists()
    script.write_text(script.read_text() + '\n')
    assert [stale.model for stale in build.mark_stale()] == ['lotka_volterra']
    assert not checksum.exists()


def test_gallery_build_reads_backend_from_environment():
    """Test that the environment selects the backend and rejects an unknown one."""
    build = GalleryBuild.from_environment(Path('e'), Path('g'), {'SKCOMPLEX_GALLERY_BACKEND': 'web'})
    assert build.backend == 'web'
    with pytest.raises(gallery_tools.GalleryError, match='SKCOMPLEX_GALLERY_BACKEND'):
        GalleryBuild.from_environment(Path('e'), Path('g'), {'SKCOMPLEX_GALLERY_BACKEND': 'vnc'})


def test_gallery_build_names_failed_pages(tmp_path, example):
    """Test that a run example whose page shows an error is named."""
    examples, gallery = tmp_path / 'examples', tmp_path / 'gallery'
    script = examples / 'biology' / 'ecology' / example.name
    script.parent.mkdir(parents=True)
    shutil.copyfile(example, script)
    page = gallery / 'biology' / 'ecology' / f'{script.stem}.md'
    page.parent.mkdir(parents=True)
    page.write_text('```{.pytb .mkd-glr-script-err-disp}\nTraceback\n```\n')
    assert [failed.model for failed in GalleryBuild(examples, gallery).failed_pages()] == ['lotka_volterra']
