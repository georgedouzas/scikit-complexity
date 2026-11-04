"""Define metabolic network elements, whose fluxes one linear program sets from their balances and bounds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import ast
import functools
import re
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import cast

from sage.all__sagemath_symbolics import SR, Expression

from .....core.elements import DeclaredElement
from .....core.laws import LinearProgramLaw
from .....core.laws.utils import both, either
from .....core.spaces import BaseSpace
from .....errors import ValidationError
from .....specs import DeclaredVariableSpec, LinearObjectiveSpec, ParameterSpec, StateSpec, VisualSpec
from ._reaction_spec import ReactionSpec

SWITCH_THRESHOLD = 0.5
NO_REACTIONS: Mapping[str, ReactionSpec] = {}
NO_OBJECTIVE: Mapping[str, float] = {}


@functools.cache
def _rule_genes(rule: str) -> tuple[str, ...]:
    """Return the gene names in a rule, read once per rule, since compiling reads the variables of a network often."""
    if not rule.strip():
        return ()
    return tuple(node.id for node in ast.walk(ast.parse(rule, mode='eval')) if isinstance(node, ast.Name))


class MetabolicNetworkElement(DeclaredElement):
    """Metabolic network whose fluxes one linear program sets, with a balance per metabolite.

    The element declares one algebraic state `v_<reaction>` per reaction, one adaptive switch `g_<gene>` per gene, and
    the parameters `lower_<reaction>` and `upper_<reaction>` of its bounds. The rule of a reaction gates both bounds, so
    a reaction whose rule is false carries no flux.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for the bars of its program.

        objective:
            The coefficient of every reaction in the objective, or `None` for `default_objective`, or nothing for a
            network whose objective another law holds.

        parsimonious:
            Whether to minimise the total absolute flux after the objective.

        infeasible:
            The infeasibility policy of the program.

        name:
            The name of the network, or `None` for none.

        reactions:
            The reactions by name.

        default_objective:
            The objective of the network when `objective` is `None`.

        variables:
            The declared states and parameters that replace the ones of the reactions, by name, or `None` for none.

    Examples:
        >>> from skcomplex.models.biology.common.metabolism import ReactionSpec, metabolic_network_type
        >>> reactions = {'uptake': ReactionSpec({'A': 1.0}), 'use': ReactionSpec({'A': -1.0}, upper=4.0)}
        >>> network = metabolic_network_type('TwoSteps', reactions, {'use': 1.0})('cell')
        >>> network.get_value('upper_use')
        4.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        objective: Mapping[str, float] | None = None,
        parsimonious: bool = False,
        infeasible: str = 'raise',
        name: str | None = None,
        reactions: Mapping[str, ReactionSpec] = NO_REACTIONS,
        default_objective: Mapping[str, float] = NO_OBJECTIVE,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
    ) -> None:
        super().__init__(label, visual=visual, variables=variables)
        self.objective = objective
        self.parsimonious = parsimonious
        self.infeasible = infeasible
        self.name = name
        self.reactions = reactions
        self.default_objective = default_objective

    def _declare_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the states and parameters of the reactions and of their genes, with the variables that replace them."""
        flux, bound = StateSpec(algebraic=True), ParameterSpec()
        declared: dict[str, DeclaredVariableSpec] = {}
        for reaction, spec in self.reactions.items():
            identifier = self._identifier(reaction)
            declared[f'v_{identifier}'] = DeclaredVariableSpec(0.0, flux)
            declared[f'lower_{identifier}'] = DeclaredVariableSpec(float(spec.lower), bound)
            declared[f'upper_{identifier}'] = DeclaredVariableSpec(float(spec.upper), bound)
        switch = ParameterSpec(adaptive=True, bounds=(0.0, 1.0))
        for gene in self.genes_of():
            declared[self.gene_name(gene)] = DeclaredVariableSpec(1.0, switch)
        return {**declared, **(self.variables or {})}

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

    def genes_of(self) -> tuple[str, ...]:
        """Return the names of the genes that the rules of the reactions read.

        Returns:
            genes:
                The names, in the order the rules name them.
        """
        names = [name for reaction in self.reactions.values() for name in _rule_genes(reaction.rule)]
        return tuple(dict.fromkeys(names))

    def get_number(self, name: str) -> float:
        """Return the value of a state or parameter of the network as a number.

        Args:
            name:
                The name of the variable, such as `upper_<reaction>`.

        Returns:
            number:
                The value.
        """
        return float(cast(float, self.get_value(name)))

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


def metabolic_network_type(
    name: str,
    reactions: Mapping[str, ReactionSpec],
    objective: Mapping[str, float] | None = None,
) -> Callable[..., MetabolicNetworkElement]:
    """Return a function that creates a `MetabolicNetworkElement` of a set of reactions.

    Args:
        name:
            The name of the network.

        reactions:
            The reactions by name.

        objective:
            The coefficient of every reaction in the objective, or `None` for none.

    Returns:
        create_network:
            The function, which takes the label, the visual, the other arguments of the element, and the values of its
            states and parameters by name.

    Examples:
        >>> from skcomplex.models.biology.common.metabolism import ReactionSpec, metabolic_network_type
        >>> create_network = metabolic_network_type('OneStep', {'r': ReactionSpec({'A': -1.0}, rule='g')})
        >>> network = create_network('cell', g_g=0.0)
        >>> network.flux_name('r'), network.genes_of(), network.get_value('g_g')
        ('v_r', ('g',), 0.0)
    """

    def create_network(label: str, visual: VisualSpec | None = None, **settings: object) -> MetabolicNetworkElement:
        network = MetabolicNetworkElement(
            label,
            visual,
            name=name,
            reactions=dict(reactions),
            default_objective=dict(objective or {}),
        )
        return network.set_params(**settings)

    return create_network
