"""Test the hooks that check the gallery after `mkdocs-gallery` generated it."""

import shutil
from pathlib import Path

import pytest
from gallery_tools import GalleryBuild

pytest.importorskip('mkdocs_gallery')

import gallery_hooks
from mkdocs.exceptions import PluginError

EXAMPLE = Path(__file__).parent.parent.parent / 'docs' / 'examples' / 'biology' / 'ecology' / 'plot_lotka_volterra.py'


def test_on_pre_build_removes_checksum_of_failed_example(tmp_path, monkeypatch):
    """Test that an example whose page shows an error loses its checksum, so the next build runs it again."""
    examples, gallery = tmp_path / 'examples', tmp_path / 'gallery'
    script = examples / 'biology' / 'ecology' / EXAMPLE.name
    script.parent.mkdir(parents=True)
    shutil.copyfile(EXAMPLE, script)
    generated = gallery / 'biology' / 'ecology'
    generated.mkdir(parents=True)
    (generated / f'{script.stem}.md').write_text('```{.pytb .mkd-glr-script-err-disp}\nTraceback\n```\n')
    checksum = generated / f'{script.name}.md5'
    checksum.write_text('md5')
    monkeypatch.setattr(gallery_hooks, 'build_from_environment', lambda _: GalleryBuild(examples, gallery))
    with pytest.raises(PluginError, match='plot_lotka_volterra'):
        gallery_hooks.on_pre_build(None)
    assert not checksum.exists()
