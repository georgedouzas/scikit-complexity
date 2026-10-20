"""Test the plots of signature analyses for the analysis figures of the gallery."""

import matplotlib as mpl

mpl.use('Agg')

from gallery_tools import plot_signatures

from skcomplex.core.analysis import signature_analysis
from skcomplex.models import run


def test_plot_signatures_draws_the_signature_of_a_run():
    """Test that the figure of a run without a named analysis shows its signature with labelled axes and title."""
    signatures = signature_analysis(run('lotka_volterra', settings={'until': 50.0}).trajectory_)
    figure = plot_signatures(signatures)
    (axes,) = figure.axes
    assert axes.get_xlabel() == 'frequency'
    assert axes.get_title().startswith('Power spectrum')
    assert axes.get_xscale() == 'log'
    assert len(axes.lines) == 2
