"""Innovation models: technologies that improve by redesigning their components or combining logic circuits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw
from ....core.system import System
from ....specs import ObservableSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ._invention import InventionInteraction
from ._part import PartElement
from ._redesign import RedesignInteraction
from ._technology_view import TechnologyView

NAND = (2, 0b0111)


FALSE = (0, 0)


TRUE = (0, 1)


def _design_structure(n_components: int, d: int, rng: np.random.Generator) -> Array:
    """Return a design structure matrix where a change of every component changes it and `d - 1` other components."""
    matrix = np.eye(n_components)
    for column in range(n_components):
        others = rng.choice([row for row in range(n_components) if row != column], d - 1, replace=False)
        matrix[others, column] = 1.0
    return matrix


class TechnologyElement(Element):
    """Logic circuit in the pool of technologies, described by the truth table of the function it computes.

    Bit `r` of the table is the output on the inputs whose bits form `r`, with the first input as the lowest bit.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        arity:
            The number of inputs.

        table:
            The truth table.

        cost:
            The number of NAND gates in the circuit.

    Examples:
        >>> from skcomplex.models.economics.innovation import TechnologyElement
        >>> GroupElement(
        ...     'technologies', element=TechnologyElement, size=3, values={'arity': [2, 0, 0], 'table': [7, 0, 1]}
        ... ).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        arity: Annotated[float, StateSpec(domain=NN)] = 2,
        table: Annotated[float, StateSpec(domain=NN)] = NAND[1],
        cost: Annotated[float, StateSpec(domain=NN)] = 1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.arity = arity
        self.table = table
        self.cost = cost

    @classmethod
    def _get_view(cls) -> type[TechnologyView]:
        """Return the view of the agents of the element."""
        return TechnologyView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since inventions combine several technologies at once.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


@register_model(
    field='economics/innovation',
    tags=('innovation', 'technology', 'learning curve', 'agents'),
    references=('Thurner et al. 2018', 'McNerney et al. 2011'),
    defaults={'until': 200},
    observables=lambda system: [ObservableSpec('total_cost', group='components', reduce='sum', variable='cost')],
)
def nk_technology(
    n_components: int = 10,
    d: int = 2,
    *,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return McNerney's model of technological improvement, whose cost falls as a power of the cumulative production.

    Every step is one unit of production and one attempt to redesign a random component together with the components
    that depend on it. Redesigns are kept only when they cut the total cost, which falls like Wright's law, more slowly
    for designs whose components depend on more others.

    Args:
        n_components:
            The number of components.

        d:
            The number of components that a redesign of one component changes, itself included.

        random_state:
            The seed or generator of the design structure matrix and the initial costs.

    Returns:
        system:
            The model, with the group `components` and the total cost `total_cost`.

    Examples:
        >>> from skcomplex.models.economics.innovation import nk_technology
        >>> nk_technology(5, 1).interactions[0].dependencies[0]
        [1.0, 0.0, 0.0, 0.0, 0.0]
    """
    rng = np.random.default_rng(random_state)
    dependencies = _design_structure(n_components, d, rng)
    components = GroupElement(
        'components',
        element=PartElement,
        size=n_components,
        values={'cost': rng.random(n_components)},
    )
    return System(
        elements=[components],
        interactions=[RedesignInteraction('redesign', components='components', dependencies=dependencies.tolist())],
    )


@register_model(
    field='economics/innovation',
    tags=('innovation', 'technology', 'combinatorial evolution', 'agents'),
    references=('Thurner et al. 2018', 'Arthur and Polak 2006'),
    defaults={'until': 200},
)
def arthur_polak(*, max_components: int = 12, max_inputs: int = 3) -> System:
    """Return Arthur and Polak's evolution of technology, which builds logic functions from NAND gates.

    The pool starts with the NAND gate and the constants 0 and 1. Every step wires a random circuit from the pool and
    keeps it when it computes a wanted function, negation, two-input and, or, exclusive or, nor, equivalence, and
    implication, or three-input and, or, parity, and majority, so simple inventions become the building blocks of
    complex ones.

    Args:
        max_components:
            The largest number of technologies in a circuit.

        max_inputs:
            The largest number of inputs of a circuit.

    Returns:
        system:
            The model, with the pool `technologies`, whose size is the reporter `technologies.count`.

    Examples:
        >>> from skcomplex.models.economics.innovation import arthur_polak
        >>> arthur_polak().elements[0].check().prepare().size_
        3
    """
    primitives = (NAND, FALSE, TRUE)
    technologies = GroupElement(
        'technologies',
        element=TechnologyElement,
        size=len(primitives),
        values={
            'arity': [float(arity) for arity, _ in primitives],
            'table': [float(table) for _, table in primitives],
            'cost': [1.0, 0.0, 0.0],
        },
        variable=True,
    )
    invention = InventionInteraction(
        'invention',
        technologies='technologies',
        max_components=float(max_components),
        max_inputs=float(max_inputs),
    )
    return System(elements=[technologies], interactions=[invention])
