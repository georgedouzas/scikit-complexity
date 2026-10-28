"""Name the avalanches of an activity series or of the relaxations of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....analysis import EventAnalysis


class AvalancheAnalysis(EventAnalysis):
    """Avalanches of an activity series or of the relaxations of a lattice, with their sizes, durations, and shapes.

    It is the event analysis of kind `'excursions'`. An avalanche of an activity series is a run of steps whose
    activity exceeds `threshold`, and its profile is the activity above the threshold. The relaxation records of a
    trajectory, such as the avalanches of a sandpile, are avalanches already.

    Args:
        variables:
            The one variable of a trajectory to read, or `None` for the only scalar series.

        threshold:
            The activity an avalanche exceeds at every step, subtracted from its profile.

    Examples:
        >>> from skcomplex.models.physics.realizations import AvalancheAnalysis
        >>> found = AvalancheAnalysis().check().prepare([0, 1, 2, 1, 0, 3, 0]).run()
        >>> found.sizes_.tolist(), found.durations_.tolist()
        ([4.0, 3.0], [3.0, 1.0])
    """

    def __init__(self, variables: Sequence[str] | None = None, *, threshold: float = 0.0) -> None:
        self.kind = 'excursions'
        self.variables = variables
        self.threshold = threshold
