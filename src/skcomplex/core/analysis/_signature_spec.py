"""Define the signature spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field

from ..base import BaseSpec
from ..enums import DataKind
from ..types import Array


@dataclass(frozen=True)
class SignatureSpec(BaseSpec):
    """A spec of the default analysis of one kind of data, ready to plot.

    It holds the title, the labels and scales of the axes, the curves, and the values worth stating of one figure,
    such as the power spectrum of the prey and the predators of a Lotka-Volterra run on log axes. It gives the first
    look at the series of an ecosystem, the jumps of a market, or the field of a chemical pattern.

    Read more in the user guide, under Analyse.

    Args:
        kind:
            The kind of data analysed.

        title:
            The title of the figure.

        xlabel:
            The label of the horizontal axis.

        ylabel:
            The label of the vertical axis.

        curves:
            A mapping from label to the abscissas and ordinates of every curve.

        values:
            The measured values worth stating, by name.

        xscale:
            `'linear'` or `'log'`.

        yscale:
            `'linear'` or `'log'`.

        points:
            Whether the curves are drawn as points rather than lines.

    See Also:
        signatures: Returns one spec per kind of data of a trajectory.

    Examples:
        >>> from skcomplex.core.analysis import signatures
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> trajectory = LotkaVolterraSystem().prepare().simulate(until=20.0, random_state=0)
        >>> signatures(trajectory).signatures[0].xlabel
        'frequency'
    """

    kind: DataKind
    title: str
    xlabel: str
    ylabel: str
    curves: dict[str, tuple[Array, Array]]
    values: dict[str, float] = field(default_factory=dict)
    xscale: str = 'linear'
    yscale: str = 'linear'
    points: bool = False
