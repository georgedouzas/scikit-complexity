"""Test the Markdown the Rerun scraper returns for an example page, its budget, and its recordings."""

import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from gallery_tools import (
    Budget,
    CaptureSettings,
    GalleryError,
    RerunScraper,
    ScreenshotCache,
    cache_key,
    images_markdown,
    model_sources,
)
from PIL import Image

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

markdown = pytest.importorskip('markdown')
pytest.importorskip('mkdocs_gallery')

DOCS = Path(__file__).parent.parent.parent.parent / 'docs'
EXAMPLE = DOCS / 'examples' / 'biology' / 'ecology' / 'plot_lotka_volterra.py'


def _script(tmp_path: Path, name: str, variables: dict[str, Any], example: Path = EXAMPLE) -> SimpleNamespace:
    """Return a gallery script of the Lotka-Volterra example that writes its page to a folder of its own."""
    generated = tmp_path / name
    (generated / 'images').mkdir(parents=True)
    return SimpleNamespace(
        src_py_file=example,
        dwnld_py_file=generated / EXAMPLE.name,
        run_vars=SimpleNamespace(
            example_globals=variables,
            image_path_iterator=iter([generated / 'images' / f'mkd_glr_{EXAMPLE.stem}_001.png']),
        ),
        gallery=SimpleNamespace(generated_dir=generated, all_info=SimpleNamespace(mkdocs_docs_dir=tmp_path)),
    )


def test_rerun_scraper_enforces_whole_script_budget(tmp_path):
    """Test that a script running beyond the budget fails at its next block, even after its recording was taken."""
    scraper = RerunScraper('native', tmp_path / 'cache', budget=Budget(max_seconds=0.05))
    scraper.begin({}, EXAMPLE)
    time.sleep(0.1)
    with pytest.raises(GalleryError, match='plot_lotka_volterra exceeds the budget: a run of'):
        scraper(None, _script(tmp_path, 'page', {'MODEL': 'lotka_volterra'}))


def test_rerun_scraper_keeps_recordings_of_concurrent_runs(tmp_path, monkeypatch):
    """Test that two builds running one example in its folder each publish their own recording."""
    example = tmp_path / 'examples' / EXAMPLE.name
    example.parent.mkdir()
    shutil.copyfile(EXAMPLE, example)
    monkeypatch.chdir(example.parent)
    settings, cache = CaptureSettings(), tmp_path / 'cache'
    thumbnail = tmp_path / 'thumbnail.png'
    Image.new('RGB', settings.thumbnail_size, 'red').save(thumbnail)
    ScreenshotCache(cache).put(cache_key(EXAMPLE, model_sources('lotka_volterra'), 'native', settings), thumbnail, {})
    scrapers, scripts = [RerunScraper('native', cache, settings) for _ in range(2)], []
    for number, scraper in enumerate(scrapers):
        scraper.begin({}, example)
        observer = RerunObserver('lotka_volterra', save_path='lotka_volterra.rrd', blueprint='default')
        Simulation(create('lotka_volterra'), dt=0.5, observers=[observer]).run(until=2)
        variables = {'MODEL': 'lotka_volterra', 'observer': observer}
        scripts.append(_script(tmp_path, f'build{number}', variables, example))
    for scraper, script in zip(scrapers, scripts, strict=True):
        assert '.rrd' in scraper(None, script)
        assert (script.gallery.generated_dir / 'images' / f'{EXAMPLE.stem}.rrd').stat().st_size > 0
    assert not (example.parent / 'lotka_volterra.rrd').exists()


def test_images_markdown_renders_every_image(tmp_path: Path):
    """Test that two screenshots render as two images, without the unformatted list of `mkdocs-gallery`."""
    generated = tmp_path / 'generated' / 'gallery' / 'physics'
    paths = [generated / 'images' / f'mkd_glr_plot_ideal_gas_00{number}.png' for number in (1, 2)]
    script = SimpleNamespace(
        gallery=SimpleNamespace(generated_dir=generated, all_info=SimpleNamespace(mkdocs_docs_dir=tmp_path)),
    )
    text = images_markdown(paths, script)
    html = markdown.markdown(text, extensions=['attr_list'])
    assert '%s' not in text
    assert html.count('<img') == 2
    assert 'src="./images/mkd_glr_plot_ideal_gas_001.png"' in html
    assert html.index('_001.png') < html.index('_002.png')


def test_rerun_scraper_returns_nothing_without_recording():
    """Test that a code block that saved no recording adds nothing to the page."""
    script = SimpleNamespace(run_vars=SimpleNamespace(example_globals={'MODEL': 'lorenz'}))
    assert RerunScraper('native', Path('cache'))(None, script) == ''
