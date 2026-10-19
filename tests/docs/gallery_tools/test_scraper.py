"""Test the Markdown the Rerun scraper returns for an example page."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from gallery_tools import RerunScraper, images_markdown

markdown = pytest.importorskip('markdown')
pytest.importorskip('mkdocs_gallery')


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
