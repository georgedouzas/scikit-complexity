"""Test the run of gallery examples and the summary of their runs."""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from gallery_tools import Budget, GalleryBuild, RunReport, find_examples, run_example, summarize

DOCS = Path(__file__).parents[3] / 'docs'


def test_summarize_prints_thumbnail_paths():
    """Test that the summary names the thumbnail of every example that has one, so it can be looked at."""
    reports = [
        RunReport('plot_lorenz.py', 1.5, 251, 9000, thumbnail='/cache/abc/thumbnail.png'),
        RunReport('plot_sir.py', error='GalleryError: blank'),
    ]
    summary = summarize(reports)
    assert 'Thumbnails:\n  plot_lorenz.py: /cache/abc/thumbnail.png' in summary
    assert 'plot_sir.py: None' not in summary


def test_run_example_draws_on_agg_whatever_backend_pyplot_already_uses(tmp_path):
    """Test that an example's plt.show opens no window although pyplot was imported with another backend."""
    mpl.use('pdf', force=True)
    example = next(item for item in find_examples(DOCS / 'examples') if item.model == 'lotka_volterra')
    build = GalleryBuild(examples=DOCS / 'examples', gallery=tmp_path / 'gallery')
    folder = Path.cwd()
    try:
        report = run_example(example, build, Budget())
    finally:
        os.chdir(folder)
    assert report.error is None
    assert mpl.get_backend().lower() == 'agg'
    assert plt.show() is None
