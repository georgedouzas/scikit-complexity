"""Measures of a whole network, held as parameters that the observables of the model report."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import ParameterSpec, VisualSpec


class NetworkMeasuresElement(Element):
    """Measures of a whole network, held as parameters that the observables of the model report.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        path_length:
            The mean shortest path length of the largest component.

        assortativity:
            The degree assortativity.

        modularity:
            The modularity of the communities found.

        agreement:
            The agreement of the communities found with a reference partition.

        triangle_zscore:
            The z-score of the number of triangles against degree-preserving edge switches.

    Examples:
        >>> from skcomplex.models.mathematics.networks import NetworkMeasuresElement
        >>> NetworkMeasuresElement('measures', modularity=0.4).modularity
        0.4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        path_length: Annotated[float, ParameterSpec()] = 0.0,
        assortativity: Annotated[float, ParameterSpec()] = 0.0,
        modularity: Annotated[float, ParameterSpec()] = 0.0,
        agreement: Annotated[float, ParameterSpec()] = 0.0,
        triangle_zscore: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.path_length = path_length
        self.assortativity = assortativity
        self.modularity = modularity
        self.agreement = agreement
        self.triangle_zscore = triangle_zscore
