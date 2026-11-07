"""Systems biology systems: multi-valued logical networks and Petri nets of regulatory and metabolic processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from itertools import product
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, SR, Expression

from ....core.components.elements import CompartmentElement, Element
from ....core.components.interactions import Interaction, ReactionInteraction
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.components.laws.utils import where
from ....core.components.system import System
from ....core.specs import RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system

type Regulations = Mapping[str, tuple[Sequence[str], Sequence[int], Mapping[tuple[bool, ...], int]]]

type Transitions = Sequence[tuple[str, Mapping[str, int], Mapping[str, int], float]]

NEGATIVE_CIRCUIT: Regulations = {
    'x': (('y',), (1,), {(False,): 2, (True,): 0}),
    'y': (('x',), (2,), {(False,): 0, (True,): 1}),
}

POSITIVE_CIRCUIT: Regulations = {
    'x': (('y',), (1,), {(False,): 2, (True,): 0}),
    'y': (('x',), (1,), {(False,): 1, (True,): 0}),
}

GENE_EXPRESSION_PLACES = {'gene_off': 1, 'gene_on': 0, 'mRNA': 0, 'protein': 0}

GENE_EXPRESSION_TRANSITIONS: Transitions = (
    ('activation', {'gene_off': 1}, {'gene_on': 1}, 0.5),
    ('repression', {'gene_on': 1, 'protein': 1}, {'gene_off': 1, 'protein': 1}, 0.05),
    ('transcription', {'gene_on': 1}, {'gene_on': 1, 'mRNA': 1}, 10.0),
    ('translation', {'mRNA': 1}, {'mRNA': 1, 'protein': 1}, 2.0),
    ('mRNA_decay', {'mRNA': 1}, {}, 1.0),
    ('protein_decay', {'protein': 1}, {}, 0.1),
)


class LogicalVariableElement(Element):
    """Variable of a generalised logical network, a discrete level from zero to its largest level.

    The levels stand for qualitative concentrations, such as absent, basal, medium, or high, cut at the thresholds at
    which the variable acts on its targets.

    Args:
        label:
            The label of the variable.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        level:
            The initial level.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import LogicalVariableElement
        >>> LogicalVariableElement('x', level=2.0).level
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        level: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level


class LogicalRegulationInteraction(Interaction):
    """Regulation of a logical variable, which steps one level at a time towards the focal level its regulators set.

    In the formalism of Thomas, every regulator acts when its level reaches its threshold. The logical parameters map
    every combination of acting regulators, a tuple of truth values in the order of the regulators, to the focal level
    of the target. The target rises or falls by one level towards its focal level at the unit rate, so the transitions
    of all variables are asynchronous random events.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the regulated variable.

        regulators:
            The paths of the regulating variables.

        thresholds:
            The level at which every regulator acts, in the order of the regulators.

        parameters:
            The focal level of the target for every combination of acting regulators.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import LogicalRegulationInteraction
        >>> regulation = LogicalRegulationInteraction(
        ...     'x_regulation', target='x', regulators=['y'], thresholds=(1,), parameters={(False,): 2, (True,): 0}
        ... )
        >>> regulation.parameters[(True,)]
        0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        regulators: Annotated[Sequence[str], RoleSpec(variadic=True)] = (),
        thresholds: Sequence[int] = (),
        parameters: Mapping[tuple[bool, ...], int] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.regulators = regulators
        self.thresholds = thresholds
        self.parameters = parameters

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the steps of the target up and down towards its focal level.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        level = s.target.level
        focal = self._focal(s.regulators)
        return [
            JumpLaw(where(focal > level, 1, 0), {level: 1}),
            JumpLaw(where(focal < level, 1, 0), {level: -1}),
        ]

    def _focal(self, regulators: Sequence[SimpleNamespace]) -> Expression:
        """Return the focal level of the target, the logical parameter of the regulators that act."""
        parameters = self.parameters or {}
        focal: Expression = SR(0)
        for acting in product((False, True), repeat=len(regulators)):
            indicator: Expression = SR(1)
            for regulator, threshold, active in zip(regulators, self.thresholds, acting, strict=True):
                indicator *= (
                    where(regulator.level >= threshold, 1, 0) if active else where(regulator.level < threshold, 1, 0)
                )
            focal += parameters.get(acting, 0) * indicator
        return focal


@register_system(
    field='biology/systems_biology',
    tags=('logical network', 'multi-valued', 'asynchronous', 'thomas', 'jumps'),
    references=('Thomas 1991', 'Raman 2021'),
    defaults={'until': 50.0, 'dt': 0.25},
)
def generalized_logical_network(
    regulations: Regulations | None = None,
    levels: Mapping[str, int] | None = None,
) -> System:
    """Return a generalised logical network of Thomas, whose variables step asynchronously towards focal levels.

    Every variable takes discrete levels, and each of its regulators acts once its level reaches a threshold. The
    logical parameters give the focal level of the variable for every combination of acting regulators, and the
    variable rises or falls one level at a time towards it, each step a random event, so the dynamics explores the
    asynchronous state transition graph. The default is the negative circuit of a three-level `x` that activates `y` at
    its second threshold and a Boolean `y` that represses `x`, whose state graph is a cycle, a sustained oscillation.
    A positive circuit, two mutual repressors, has two stable states instead.

    Args:
        regulations:
            For every variable, its regulators, their thresholds, and the logical parameters of its focal level, or
            `None` for the negative circuit.

        levels:
            The initial level of every variable, or `None` for 0.

    Returns:
        system:
            The system, with the levels `<variable>.level`.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import generalized_logical_network
        >>> [element.label for element in generalized_logical_network().elements]
        ['x', 'y']
    """
    network = NEGATIVE_CIRCUIT if regulations is None else regulations
    start = levels or {}
    return System(
        elements=[LogicalVariableElement(name, level=float(start.get(name, 0))) for name in network],
        interactions=[
            LogicalRegulationInteraction(
                f'{name}_regulation',
                target=name,
                regulators=list(regulators),
                thresholds=tuple(thresholds),
                parameters=dict(parameters),
            )
            for name, (regulators, thresholds, parameters) in network.items()
        ],
    )


@register_system(
    field='biology/systems_biology',
    tags=('petri net', 'stochastic simulation', 'gene expression', 'invariants', 'jumps'),
    references=('Petri 1962', 'Reddy et al. 1993', 'Raman 2021'),
    defaults={'until': 100.0, 'dt': 0.5},
)
def petri_net(
    places: Mapping[str, int] | None = None,
    transitions: Transitions | None = None,
    *,
    stochastic: bool = True,
) -> System:
    """Return a Petri net, places that hold tokens and transitions that consume and produce them when they fire.

    Every transition is a pair of input and output arcs with weights and a rate. In a stochastic Petri net it fires at
    random times with the mass-action propensity of its input places, consuming the tokens of its input arcs and
    producing those of its output arcs, and otherwise it is a continuous Petri net of rate equations. Every place
    invariant, a weighted sum of tokens that no transition changes, is conserved. The default net is the expression
    of a gene that switches on, is transcribed and translated, and is switched off by its own protein, with the
    invariant `gene_on + gene_off = 1`.

    Args:
        places:
            The initial tokens of every place, or `None` for the gene expression net.

        transitions:
            For every transition, its label, input arcs, output arcs, and rate, or `None` for the gene expression net.

        stochastic:
            Whether the transitions fire at random times.

    Returns:
        system:
            The system, with the tokens `<place>.n` of every place.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import petri_net
        >>> [element.label for element in petri_net().elements]
        ['gene_off', 'gene_on', 'mRNA', 'protein']
    """
    marking = GENE_EXPRESSION_PLACES if places is None else places
    arcs = GENE_EXPRESSION_TRANSITIONS if transitions is None else transitions
    mode = 'jump' if stochastic else 'rate'
    return System(
        elements=[CompartmentElement(name, n=float(tokens)) for name, tokens in marking.items()],
        interactions=[
            ReactionInteraction(
                label,
                reactants={(place, 'n'): weight for place, weight in inputs.items()} or None,
                products={(place, 'n'): weight for place, weight in outputs.items()} or None,
                k=rate,
                mode=mode,
            )
            for label, inputs, outputs, rate in arcs
        ],
    )
