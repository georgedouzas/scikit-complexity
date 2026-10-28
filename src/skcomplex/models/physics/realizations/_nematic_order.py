"""Name the nematic order of active particles, the alignment of their headings regardless of sign."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....analysis import AlignmentAnalysis


class NematicOrderAnalysis(AlignmentAnalysis):
    """Nematic order of active particles at every frame, `|<exp(2 i theta)>|` of their headings.

    It is the alignment of kind `'axial'`, which ignores the sign of a heading, the order parameter of active nematics
    such as the Ginelli model.

    Args:
        variables:
            The field or the series of a trajectory to read, or `None` for its agents.

        group:
            The path of the agent group of a trajectory, or `None` for the first one.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.physics.realizations import NematicOrderAnalysis
        >>> NematicOrderAnalysis().check().prepare([0.0, np.pi]).run().order_.round(6).tolist()
        [1.0]
    """

    def __init__(self, variables: Sequence[str] | None = None, *, group: str | None = None) -> None:
        super().__init__('axial', variables, group=group)
