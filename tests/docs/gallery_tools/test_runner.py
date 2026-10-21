"""Test the run of gallery examples and the summary of their runs."""

from gallery_tools import RunReport, summarize


def test_summarize_prints_thumbnail_paths():
    """Test that the summary names the thumbnail of every example that has one, so it can be looked at."""
    reports = [
        RunReport('plot_lorenz.py', 1.5, 251, 9000, thumbnail='/cache/abc/thumbnail.png'),
        RunReport('plot_sir.py', error='GalleryError: blank'),
    ]
    summary = summarize(reports)
    assert 'Thumbnails:\n  plot_lorenz.py: /cache/abc/thumbnail.png' in summary
    assert 'plot_sir.py: None' not in summary
