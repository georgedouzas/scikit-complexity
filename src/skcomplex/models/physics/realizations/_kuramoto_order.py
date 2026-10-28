"""Name Kuramoto's order parameter of phase oscillators, their synchrony and mean phase."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....analysis import AlignmentAnalysis


class KuramotoOrderAnalysis(AlignmentAnalysis):
    """Kuramoto's order parameter `r exp(i psi) = <exp(i theta)>` of oscillators at every sample.

    It is the alignment of kind `'mean-phase'`: the synchrony `r` in `order_`, 1 when every oscillator has the same
    phase and about `1 / sqrt(n)` for incoherent ones, and the mean phase `psi` in `phase_`.

    Args:
        variables:
            The phases of a trajectory to read, one series per oscillator, such as `['oscillators.theta']`.

        group:
            The path of an agent group of a trajectory whose headings are the phases, or `None`.

    Examples:
        >>> from skcomplex.models.physics.realizations import KuramotoOrderAnalysis
        >>> KuramotoOrderAnalysis().check().prepare([[0.5, 0.5, 0.5]]).run().order_.round(6).tolist()
        [1.0]
    """

    def __init__(self, variables: Sequence[str] | None = None, *, group: str | None = None) -> None:
        self.kind = 'mean-phase'
        self.variables = variables
        self.group = group
        self.vectors = False
