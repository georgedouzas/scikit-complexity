"""Define the signature spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..types import Array
from ._base_spec import BaseSpec

if TYPE_CHECKING:
    from ..types import DataKind


@dataclass(frozen=True)
class SignatureSpec(BaseSpec):
    """Default analysis of one kind of data, ready to plot.

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
