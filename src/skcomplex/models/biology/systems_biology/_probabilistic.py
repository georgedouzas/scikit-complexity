"""Systems biology models: probabilistic Boolean networks, which draw the function of every node at every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

from ....core.system import System
from ..._catalog import register_model
from ._probabilistic_function import ProbabilisticFunctionInteraction
from ._probabilistic_node import ProbabilisticNodeElement
from .types import Candidates

SHMULEVICH_INPUTS = ('x1', 'x2', 'x3')


SHMULEVICH_NETWORK: dict[str, tuple[tuple[tuple[int, ...], float], ...]] = {
    'x1': (((0, 1, 1, 1, 0, 1, 1, 1), 0.6), ((0, 0, 0, 1, 0, 1, 1, 1), 0.4)),
    'x2': (((0, 1, 1, 0, 0, 1, 0, 1), 1.0),),
    'x3': (((0, 0, 0, 1, 0, 1, 1, 1), 0.5), ((0, 0, 0, 0, 0, 0, 0, 1), 0.5)),
}


@register_model(
    field='biology/systems_biology',
    tags=('boolean network', 'gene regulation', 'discrete map', 'stochastic'),
    references=('Shmulevich et al. 2002', 'Raman 2021'),
    defaults={'until': 100, 'dt': 1},
)
def probabilistic_boolean_network(
    network: Mapping[str, Candidates] | None = None,
    *,
    inputs: Sequence[str] = SHMULEVICH_INPUTS,
    active: Sequence[str] = ('x1',),
) -> System:
    """Return a probabilistic Boolean network, whose nodes draw one of their candidate functions at every step.

    Every node has candidate truth tables of the inputs, each with a probability, and at every step every node draws
    one of its tables independently and takes its value, all nodes together. The default network follows the three-gene
    example of Shmulevich et al., in which `x1` and `x3` have two candidate functions and `x2` one. From one initial
    state the network reaches different attractors in different runs.

    Args:
        network:
            A mapping from node name to its candidate tables with their probabilities, or `None` for the three-gene
            example.

        inputs:
            The nodes every table reads, in the order of its bits, from the most significant.

        active:
            The nodes that are on at the start.

    Returns:
        system:
            The model, with the values `<node>.x` and the indices of the drawn functions `<node>.function`.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import probabilistic_boolean_network
        >>> probabilistic_boolean_network().interactions[0].probabilities
        (0.6, 0.4)
    """
    chosen = SHMULEVICH_NETWORK if network is None else network
    nodes = [ProbabilisticNodeElement(name, x=int(name in active)) for name in chosen]
    functions = [
        ProbabilisticFunctionInteraction(
            f'functions_{name}',
            target=name,
            inputs=list(inputs),
            tables=tuple(tuple(table) for table, _ in candidates),
            probabilities=tuple(probability for _, probability in candidates),
        )
        for name, candidates in chosen.items()
    ]
    return System(elements=nodes, interactions=functions)
