"""Define the flux balance of a metabolic network.

One linear program sets the fluxes of the network from its balances and bounds.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import ast
import functools
import re
from collections.abc import Mapping
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from .....core.components.elements import Element, VisualSpec
from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, ConstraintLaw, ObjectiveLaw
from .....core.components.laws.utils import both, either
from .....core.errors import ValidationError
from .....core.types import Role
from ._reaction_spec import ReactionSpec

SWITCH_THRESHOLD = 0.5
NO_REACTIONS: Mapping[str, ReactionSpec] = {}
NO_OBJECTIVE: Mapping[str, float] = {}


class MetabolicCellElement(Element):
    """Cell whose values hold the fluxes, the bounds, and the gene switches of a metabolic network.

    It is drawn as the linear program of its network, and `FluxBalanceInteraction.get_values` gives its values.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import FluxBalanceInteraction, MetabolicCellElement
        >>> from skcomplex.systems.biology.common.metabolism import ReactionSpec
        >>> network = FluxBalanceInteraction('fba', reactions={'r': ReactionSpec({'A': -1.0})})
        >>> MetabolicCellElement('cell', values=network.get_values()).values['upper_r']
        1000.0
    """

    visual = VisualSpec('linear_program')


class FluxBalanceInteraction(Interaction):
    """Flux balance of a metabolic network, which sets its fluxes to the best ones its balances and bounds allow.

    A network is a set of reactions, each with the coefficients of its metabolites, its bounds, and the rule of the
    genes that make its enzymes. The interaction sets the flux `v_<reaction>` of its cell, one per reaction, to the
    optimum of a linear program: every metabolite balances, `sum_r S_mr v_r = 0`, every flux stays between the values
    `lower_<reaction>` and `upper_<reaction>` of the cell, and the weighted sum of the fluxes of `objective` is
    greatest (Raman 2021, Eq. 8.9). A reaction whose rule is false, read from the switches `g_<gene>` of the cell,
    carries no flux. The program is solved at every time of a run, so the network also drives a culture that changes
    in time, and `get_values` gives the values the cell starts with.

    Args:
        label:
            The label of the interaction.

        cell:
            The path of the cell whose values hold the fluxes, the bounds, and the switches.

        reactions:
            The reactions by name.

        objective:
            The coefficient of every reaction in the objective, or nothing for a network whose fluxes another objective
            chooses, such as the growth of a community.

        parsimonious:
            Whether to minimise the total absolute flux after the objective.

        infeasible:
            What a run does when the balances and bounds cannot all hold: `'raise'`, `'zero'` to set the fluxes to
            zero, or `'hold'` to keep their last values.

        name:
            The name of the network, or `None` for none.

        biomass:
            The path of the biomass that `threshold` reads, or `None` for none.

        threshold:
            The biomass above which the program is solved, or `None` for always. The fluxes are zero elsewhere, as at
            the empty boxes of a plate.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.biology.common.metabolism import FluxBalanceInteraction, ReactionSpec
        >>> reactions = {'uptake': ReactionSpec({'A': 1.0}, upper=4.0), 'use': ReactionSpec({'A': -1.0})}
        >>> fba = FluxBalanceInteraction('fba', cell='cell', reactions=reactions, objective={'use': 1.0})
        >>> cell = Element('cell', values=fba.get_values())
        >>> trajectory = System(elements=[cell], interactions=[fba]).prepare().simulate(until=0.0)
        >>> float(trajectory.get_values('cell.v_use')[0])
        4.0
    """

    def __init__(
        self,
        label: str = 'fba',
        *,
        cell: Role = 'cell',
        reactions: Mapping[str, ReactionSpec] = NO_REACTIONS,
        objective: Mapping[str, float] = NO_OBJECTIVE,
        parsimonious: bool = False,
        infeasible: str = 'raise',
        name: str | None = None,
        biomass: Role | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__(label)
        self.cell = cell
        self.reactions = reactions
        self.objective = objective
        self.parsimonious = parsimonious
        self.infeasible = infeasible
        self.name = name
        self.biomass = biomass
        self.threshold = threshold

    @staticmethod
    @functools.cache
    def _get_rule_genes(rule: str) -> tuple[str, ...]:
        """Return the gene names in a rule, read once per rule.

        Compiling reads the rules of a network often.
        """
        if not rule.strip():
            return ()
        return tuple(node.id for node in ast.walk(ast.parse(rule, mode='eval')) if isinstance(node, ast.Name))

    @staticmethod
    def _get_identifier(name: str) -> str:
        """Return a name as a part of an identifier."""
        return re.sub(r'\W', '_', name)

    @classmethod
    def get_flux_name(cls, reaction: str) -> str:
        """Return the name of the value of the cell that holds the flux of a reaction.

        Args:
            reaction:
                The name of the reaction.

        Returns:
            name:
                The name of the value, such as `v_PGI` for the reaction `PGI`.
        """
        return f'v_{cls._get_identifier(reaction)}'

    @classmethod
    def get_gene_name(cls, gene: str) -> str:
        """Return the name of the value of the cell that holds the switch of a gene.

        Args:
            gene:
                The name of the gene.

        Returns:
            name:
                The name of the value, such as `g_b4025` for the gene `b4025`.
        """
        return f'g_{cls._get_identifier(gene)}'

    def get_genes_of(self) -> tuple[str, ...]:
        """Return the names of the genes that the rules of the reactions read.

        Returns:
            genes:
                The names, in the order the rules name them.
        """
        names = [name for reaction in self.reactions.values() for name in self._get_rule_genes(reaction.rule)]
        return tuple(dict.fromkeys(names))

    def get_values(self) -> dict[str, float]:
        """Return the values the cell of the network starts with.

        They are, for every reaction in order, its flux at zero and its bounds, and then the switch of every gene, on
        at 1.

        Returns:
            values:
                The values by name, such as `{'v_r': 0.0, 'lower_r': 0.0, 'upper_r': 1000.0, 'g_g': 1.0}`.

        Examples:
            >>> from skcomplex.systems.biology.common.metabolism import FluxBalanceInteraction, ReactionSpec
            >>> reactions = {'r': ReactionSpec({'A': -1.0}, upper=4.0, rule='g')}
            >>> FluxBalanceInteraction('fba', reactions=reactions).get_values()
            {'v_r': 0.0, 'lower_r': 0.0, 'upper_r': 4.0, 'g_g': 1.0}
        """
        values = {}
        for reaction, spec in self.reactions.items():
            identifier = self._get_identifier(reaction)
            values[f'v_{identifier}'] = 0.0
            values[f'lower_{identifier}'] = float(spec.lower)
            values[f'upper_{identifier}'] = float(spec.upper)
        return values | dict.fromkeys(map(self.get_gene_name, self.get_genes_of()), 1.0)

    def _get_gate(self, cell: SimpleNamespace, rule: str) -> Expression | int:
        """Return the condition of a rule on the switches of its genes.

        A reaction without genes has the condition 1.
        """
        if not rule.strip():
            return 1
        return both(self._get_condition(cell, ast.parse(rule, mode='eval').body))

    def _get_condition(self, cell: SimpleNamespace, node: ast.expr) -> Expression:
        """Return the condition of a node of a rule."""
        if isinstance(node, ast.BoolOp):
            combine = both if isinstance(node.op, ast.And) else either
            return combine(*(self._get_condition(cell, item) for item in node.values))
        if isinstance(node, ast.Name):
            return getattr(cell, self.get_gene_name(node.id)) > SWITCH_THRESHOLD
        error_msg = f'A gene rule of {self.label} holds {ast.dump(node)}. Use gene names, `and`, `or`, and parentheses.'
        raise ValidationError(error_msg)

    def _get_balances(self, cell: SimpleNamespace) -> list[ConstraintLaw]:
        """Return the balance of every metabolite."""
        terms: dict[str, list[Expression]] = {}
        for reaction, spec in self.reactions.items():
            flux = getattr(cell, self.get_flux_name(reaction))
            for metabolite, coefficient in spec.coefficients.items():
                terms.setdefault(metabolite, []).append(float(coefficient) * flux)
        return [ConstraintLaw(sum(items[1:], items[0]) == 0) for items in terms.values()]

    def _get_bounds(self, cell: SimpleNamespace) -> list[ConstraintLaw]:
        """Return the lower and upper bound of every reaction, gated by its rule."""
        bounds = []
        for reaction, spec in self.reactions.items():
            flux = getattr(cell, self.get_flux_name(reaction))
            gate = self._get_gate(cell, spec.rule)
            identifier = self._get_identifier(reaction)
            bounds.append(ConstraintLaw(flux >= getattr(cell, f'lower_{identifier}') * gate))
            bounds.append(ConstraintLaw(flux <= getattr(cell, f'upper_{identifier}') * gate))
        return bounds

    def _get_objective(self, cell: SimpleNamespace, biomass: SimpleNamespace | None) -> ObjectiveLaw:
        """Return the objective of the network over its fluxes.

        It is the greatest weighted sum of `objective`, then the smallest total absolute flux for a parsimonious
        network, or any point that the balances and bounds allow for a network without an objective. With a threshold,
        it is sought only where the biomass lies above it.
        """
        active = None if self.threshold is None or biomass is None else biomass.x > float(self.threshold)
        fluxes = [getattr(cell, self.get_flux_name(name)) for name in self.reactions]
        values = []
        if self.objective:
            terms = [float(weight) * getattr(cell, self.get_flux_name(name)) for name, weight in self.objective.items()]
            values.append(-sum(terms[1:], terms[0]))
            if self.parsimonious:
                absolutes = [abs(flux) for flux in fluxes]
                values.append(sum(absolutes[1:], absolutes[0]))
        return ObjectiveLaw(values, fluxes, active=active, infeasible=self.infeasible)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the objective of the network with its balances and bounds.

        Args:
            context:
                The namespace of the cell and the biomass.
        """
        super()._prepare(context)
        laws: list[BaseLaw] = [*self._get_balances(context.cell), *self._get_bounds(context.cell)]
        self.laws_ = [*laws, self._get_objective(context.cell, None if self.biomass is None else context.biomass)]
