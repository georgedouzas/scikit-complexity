"""Define metabolic network elements, whose fluxes one linear program sets from their balances and bounds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import ast
import inspect
import re
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import SR, Expression

from .....core.elements import Element
from .....core.laws import LinearProgramLaw
from .....core.laws.utils import both, either
from .....core.spaces import BaseSpace
from .....errors import ValidationError
from .....specs import LinearObjectiveSpec, ParameterSpec, StateSpec, VisualSpec
from ._reaction_spec import ReactionSpec

SWITCH_THRESHOLD = 0.5


class MetabolicNetworkElement(Element):
    """Metabolic network whose fluxes one linear program sets, with a balance per metabolite.

    A subclass, which `metabolic_network_type` builds, declares one algebraic state `v_<reaction>` per reaction, one
    adaptive switch `g_<gene>` per gene, and the parameters `lower_<reaction>` and `upper_<reaction>` of its bounds. The
    rule of a reaction gates both bounds, so a reaction whose rule is false carries no flux.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for the bars of its program.

        objective:
            The coefficient of every reaction in the objective, or `None` for the objective of the type, or nothing for
            a network whose objective another law holds.

        parsimonious:
            Whether to minimise the total absolute flux after the objective.

        infeasible:
            The infeasibility policy of the program.

    Attributes:
        reactions:
            The reactions of the type, by name.

        default_objective:
            The objective of the type.

    Examples:
        >>> from skcomplex.models.biology.common.metabolism import ReactionSpec, metabolic_network_type
        >>> reactions = {'uptake': ReactionSpec({'A': 1.0}), 'use': ReactionSpec({'A': -1.0}, upper=4.0)}
        >>> network = metabolic_network_type('TwoStepsElement', reactions, {'use': 1.0})('cell')
        >>> network.upper_use
        4.0
    """

    reactions: ClassVar[Mapping[str, ReactionSpec]] = {}
    default_objective: ClassVar[Mapping[str, float]] = {}

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        objective: Mapping[str, float] | None = None,
        parsimonious: bool = False,
        infeasible: str = 'raise',
    ) -> None:
        super().__init__(label, visual=visual)
        self.objective = objective
        self.parsimonious = parsimonious
        self.infeasible = infeasible

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, the bars of the fluxes.

        Args:
            context:
                The space the element inherits from its system, which a network does not read.
        """
        self.default_visual_ = VisualSpec('linear_program')

    @staticmethod
    def _identifier(name: str) -> str:
        """Return a name as a part of an identifier."""
        return re.sub(r'\W', '_', name)

    @classmethod
    def flux_name(cls, reaction: str) -> str:
        """Return the name of the state of the flux of a reaction.

        Args:
            reaction:
                The name of the reaction.

        Returns:
            name:
                The name of the state.
        """
        return f'v_{cls._identifier(reaction)}'

    @classmethod
    def gene_name(cls, gene: str) -> str:
        """Return the name of the switch of a gene.

        Args:
            gene:
                The name of the gene.

        Returns:
            name:
                The name of the switch.
        """
        return f'g_{cls._identifier(gene)}'

    @classmethod
    def genes_of(cls) -> tuple[str, ...]:
        """Return the names of the genes that the rules of the reactions read.

        Returns:
            genes:
                The names, in the order the rules name them.
        """
        names = [name for reaction in cls.reactions.values() for name in _rule_genes(reaction.rule)]
        return tuple(dict.fromkeys(names))

    def _gate(self, namespace: SimpleNamespace, rule: str) -> Expression | int:
        """Return the condition of a rule on the switches of its genes, or 1 for a reaction without genes."""
        if not rule.strip():
            return 1
        return both(self._condition(namespace, ast.parse(rule, mode='eval').body))

    def _condition(self, namespace: SimpleNamespace, node: ast.expr) -> Expression:
        """Return the condition of a node of a rule."""
        if isinstance(node, ast.BoolOp):
            combine = both if isinstance(node.op, ast.And) else either
            return combine(*(self._condition(namespace, item) for item in node.values))
        if isinstance(node, ast.Name):
            return getattr(namespace, self.gene_name(node.id)) > SWITCH_THRESHOLD
        error_msg = f'A gene rule of {self.label} holds {ast.dump(node)}. Use gene names, `and`, `or`, and parentheses.'
        raise ValidationError(error_msg)

    def _balances(self, namespace: SimpleNamespace) -> list[Expression]:
        """Return the balance of every metabolite."""
        terms: dict[str, list[Expression]] = {}
        for reaction, spec in self.reactions.items():
            flux = getattr(namespace, self.flux_name(reaction))
            for metabolite, coefficient in spec.coefficients.items():
                terms.setdefault(metabolite, []).append(float(coefficient) * flux)
        return [sum(items[1:], items[0]) == 0 for items in terms.values()]

    def _bounds(self, namespace: SimpleNamespace) -> list[Expression]:
        """Return the lower and upper bound of every reaction, gated by its rule."""
        relations = []
        for reaction, spec in self.reactions.items():
            flux = getattr(namespace, self.flux_name(reaction))
            gate = self._gate(namespace, spec.rule)
            identifier = self._identifier(reaction)
            relations.append(flux >= getattr(namespace, f'lower_{identifier}') * gate)
            relations.append(flux <= getattr(namespace, f'upper_{identifier}') * gate)
        return relations

    def _objectives(self, namespace: SimpleNamespace) -> list[LinearObjectiveSpec]:
        """Return the objective of the network and, for a parsimonious network, its total absolute flux."""
        weights = self.default_objective if self.objective is None else self.objective
        if not weights:
            return []
        terms = [float(weight) * getattr(namespace, self.flux_name(name)) for name, weight in weights.items()]
        objectives = [LinearObjectiveSpec(sum(terms[1:], terms[0]))]
        if self.parsimonious:
            fluxes = [abs(getattr(namespace, self.flux_name(name))) for name in self.reactions]
            objectives.append(LinearObjectiveSpec(sum(fluxes[1:], fluxes[0]), sense='min'))
        return objectives

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[LinearProgramLaw]:
        """Return the linear program of the network.

        Args:
            namespace:
                The namespace of the symbols of the element.

        Returns:
            laws:
                The linear program law.
        """
        variables = [SR(getattr(namespace, self.flux_name(name))) for name in self.reactions]
        constraints = [*self._balances(namespace), *self._bounds(namespace)]
        objectives = self._objectives(namespace)
        return [LinearProgramLaw(variables, constraints, objectives, infeasible=self.infeasible)]


def _rule_genes(rule: str) -> list[str]:
    """Return the gene names in a rule."""
    if not rule.strip():
        return []
    return [node.id for node in ast.walk(ast.parse(rule, mode='eval')) if isinstance(node, ast.Name)]


def metabolic_network_type(
    name: str,
    reactions: Mapping[str, ReactionSpec],
    objective: Mapping[str, float] | None = None,
) -> type[MetabolicNetworkElement]:
    """Return a subclass of `MetabolicNetworkElement` that declares the states and parameters of a set of reactions.

    Args:
        name:
            The name of the class.

        reactions:
            The reactions by name.

        objective:
            The coefficient of every reaction in the objective, or `None` for none.

    Returns:
        element_type:
            The subclass.

    Examples:
        >>> from skcomplex.models.biology.common.metabolism import ReactionSpec, metabolic_network_type
        >>> element_type = metabolic_network_type('OneStepElement', {'r': ReactionSpec({'A': -1.0}, rule='g')})
        >>> element_type.flux_name('r'), element_type.genes_of()
        ('v_r', ('g',))
    """
    switch = Annotated[float, ParameterSpec(adaptive=True, bounds=(0.0, 1.0))]
    flux = Annotated[float, StateSpec(algebraic=True)]
    bound = Annotated[float, ParameterSpec()]
    values: dict[str, float] = {}
    annotations: dict[str, object] = {}
    for reaction, spec in reactions.items():
        identifier = MetabolicNetworkElement._identifier(reaction)
        values[f'v_{identifier}'] = 0.0
        annotations[f'v_{identifier}'] = flux
        values[f'lower_{identifier}'] = float(spec.lower)
        annotations[f'lower_{identifier}'] = bound
        values[f'upper_{identifier}'] = float(spec.upper)
        annotations[f'upper_{identifier}'] = bound
    genes = dict.fromkeys(gene for spec in reactions.values() for gene in _rule_genes(spec.rule))
    for gene in genes:
        values[MetabolicNetworkElement.gene_name(gene)] = 1.0
        annotations[MetabolicNetworkElement.gene_name(gene)] = switch

    def initialize(
        self: MetabolicNetworkElement,
        label: str,
        visual: VisualSpec | None = None,
        *,
        objective: Mapping[str, float] | None = None,
        parsimonious: bool = False,
        infeasible: str = 'raise',
        **given: float,
    ) -> None:
        MetabolicNetworkElement.__init__(
            self,
            label,
            visual,
            objective=objective,
            parsimonious=parsimonious,
            infeasible=infeasible,
        )
        for variable, value in {**values, **given}.items():
            setattr(self, variable, value)

    settings = [
        inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('label', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('visual', inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
        inspect.Parameter('objective', inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter('parsimonious', inspect.Parameter.KEYWORD_ONLY, default=False),
        inspect.Parameter('infeasible', inspect.Parameter.KEYWORD_ONLY, default='raise'),
    ]
    declared = [
        inspect.Parameter(variable, inspect.Parameter.KEYWORD_ONLY, default=values[variable], annotation=annotation)
        for variable, annotation in annotations.items()
    ]
    vars(initialize)['__signature__'] = inspect.Signature([*settings, *declared])
    members = {
        '__init__': initialize,
        'reactions': dict(reactions),
        'default_objective': dict(objective or {}),
        '__doc__': f'Metabolic network {name}, with its reactions declared as states and parameters.',
    }
    return type(name, (MetabolicNetworkElement,), members)
