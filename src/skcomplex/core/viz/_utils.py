"""Hold the drawing conventions and helpers that the visuals share."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx
import numpy as np
import numpy.typing as npt
import rerun as rr

UPWARD = np.array([1.0, -1.0])
LAYOUT_SEED = 0
NODE_SPACING = 40.0
POINT_SHARE = 0.02
CROWD_SIZE = 10
CROWD_EXPONENT = 0.25
GRID_RADIUS = 0.4
MODULUS = 'modulus'
PHASE = 'phase'
PARTS = ('real', 'imag')


def _radius_of(extent: float, count: int) -> float:
    """Return a point radius that is a share of the extent of a view, smaller for crowds so they do not overlap."""
    return POINT_SHARE * (extent if extent > 0 else 1.0) / max(count / CROWD_SIZE, 1.0) ** CROWD_EXPONENT


def _network_layout(graph: nx.Graph) -> dict[str, npt.NDArray[np.float64]]:
    """Return fixed positions of the nodes of a network, with neighbours about a spacing apart.

    The positions are the `pos` attributes of the nodes when every node has one, and a seeded spring layout otherwise.
    """
    scale = NODE_SPACING * float(np.sqrt(max(graph.number_of_nodes(), 1))) / 2
    given = nx.get_node_attributes(graph, 'pos')
    if given and len(given) == graph.number_of_nodes():
        layout = nx.rescale_layout_dict({node: np.asarray(position, dtype=float) for node, position in given.items()})
    else:
        layout = nx.spring_layout(graph, seed=LAYOUT_SEED)
    return {str(node): np.asarray(position, dtype=float) * scale for node, position in layout.items()}


def log_scalars(recording: rr.RecordingStream, path: str, value: object) -> None:
    """Log the components of a value as scalar series, the real and imaginary parts of a complex value apart.

    Args:
        recording:
            The Rerun recording.

        path:
            The entity path of the series.

        value:
            A number or an array of numbers, real or complex.

    Examples:
        >>> import rerun as rr
        >>> from skcomplex.core.viz import log_scalars
        >>> log_scalars(rr.RecordingStream('example'), 'value', 1.0 + 2.0j)
    """
    values = np.ravel(np.asarray(value))
    if not np.iscomplexobj(values):
        recording.log(path, rr.Scalars(values))
        return
    for part, components in zip(PARTS, (values.real, values.imag), strict=True):
        recording.log(f'{path}/{part}', rr.Scalars(components))
