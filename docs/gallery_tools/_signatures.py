"""Plot the signature analyses of a run.

They form the analysis figure of a gallery example.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from skcomplex.core.analysis import SignaturesResult
from skcomplex.core.specs import SignatureSpec

PANEL_WIDTH = 5.0
PANEL_HEIGHT = 4.0
STATED_VALUES = 3


def _caption(signature: SignatureSpec) -> str:
    """Return the title of a signature followed by the first stated values."""
    stated = [f'{name} = {value:.3g}' for name, value in list(signature.values.items())[:STATED_VALUES]]
    return '\n'.join([signature.title, ', '.join(stated)]) if stated else signature.title


def plot_signature(signature: SignatureSpec, axes: Axes) -> Axes:
    """Draw one signature on axes.

    It draws its curves, its scales, its labels, and its title with values.

    Args:
        signature:
            One signature of `SignaturesResult.signatures`.

        axes:
            The Matplotlib axes.

    Returns:
        axes:
            The axes drawn on.
    """
    for label, (abscissas, ordinates) in signature.curves.items():
        if signature.points:
            axes.plot(abscissas, ordinates, '.', markersize=2, label=label)
        else:
            axes.plot(abscissas, ordinates, label=label)
    axes.set_xscale(signature.xscale)
    axes.set_yscale(signature.yscale)
    axes.set_xlabel(signature.xlabel)
    axes.set_ylabel(signature.ylabel)
    axes.set_title(_caption(signature), fontsize='medium')
    if len(signature.curves) > 1:
        axes.legend(fontsize='small')
    return axes


def plot_signatures(result: SignaturesResult) -> Figure:
    """Return a figure with one panel per signature of a run.

    The panels are side by side.

    Args:
        result:
            The signatures of a run, as `signatures` returns them.

    Returns:
        figure:
            The figure.
    """
    signatures = result.signatures
    count = max(len(signatures), 1)
    figure, panels = plt.subplots(1, count, figsize=(PANEL_WIDTH * count, PANEL_HEIGHT), squeeze=False)
    for signature, axes in zip(signatures, panels[0], strict=False):
        plot_signature(signature, axes)
    figure.tight_layout()
    return figure
