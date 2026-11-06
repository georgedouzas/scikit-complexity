"""Analyse metabolic networks: flux variability, flux scans, COBRApy methods, and strain designs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from .....core.analysis import BaseAnalysis
from .....core.analysis.types import Source
from .....core.components.spaces import StaticTime
from .....core.components.system import System
from .....core.engine.compile import PassCompiler
from .....core.engine.problems import LinearProgramKernel, ProblemSpec
from .....core.errors import AnalysisError
from .....core.specs import RunStateSpec
from .....core.types import Array, DataKind
from ...._extras import import_extra
from ._cobra_io import load_extra, to_cobra
from ._culture import ObjectiveFloorInteraction
from ._metabolic_network import MetabolicNetworkElement

COBRA_KINDS = ('variability', 'parsimonious', 'deletions', 'moma', 'room', 'synthetic-lethals')
DESIGN_KINDS = ('optknock', 'robustknock', 'optcouple', 'mcs')
DESIGN_NAMES = {'optknock': 'OPTKNOCK', 'robustknock': 'ROBUSTKNOCK', 'optcouple': 'OPTCOUPLE', 'mcs': 'SUPPRESS'}
LETHAL_GROWTH = 1e-6
PAIR_SIZE = 2


class BaseNetworkAnalysis(BaseAnalysis):
    """Base of the analyses of a system that holds a metabolic network.

    Args:
        network:
            The label of the network, or `None` for the first network of the system.

    Attributes:
        network_:
            The metabolic network.
    """

    network: str | None

    def __init__(self, *, network: str | None = None) -> None:
        self.network = network

    def _get_data_kind(self) -> DataKind:
        """Return the kind of data the analysis reads, the network of a system."""
        return DataKind.SERIES

    def _networks_of(self, system: System) -> list[MetabolicNetworkElement]:
        """Return the metabolic networks of a system and its nested systems."""
        found = [item for item in system.elements if isinstance(item, MetabolicNetworkElement)]
        for nested in system.systems:
            found += self._networks_of(nested)
        return found

    def _prepare(self, context: Source) -> None:
        """Read the network of a system.

        Args:
            context:
                The system.

        Raises:
            AnalysisError:
                The source is not a system with the network.
        """
        networks = self._networks_of(context) if isinstance(context, System) else []
        chosen = [item for item in networks if self.network is None or item.label == self.network]
        if not chosen:
            error_msg = f'{type(self).__name__} reads a system that holds a metabolic network element.'
            raise AnalysisError(error_msg)
        self.network_ = chosen[0]
        self.system_ = context

    def _compiled_network(self, *, parsimonious: bool = False, floor: float | None = None) -> None:
        """Compile the network under static time, with its objective kept above a floor if one is given."""
        element = type(self.network_)(
            **{**self.network_.get_params(deep=False), 'parsimonious': parsimonious, 'infeasible': 'raise'},
        )
        weights = self.network_.default_objective if self.network_.objective is None else self.network_.objective
        kept = (
            []
            if floor is None
            else [ObjectiveFloorInteraction('kept', cell=element.label, weights=weights, floor=floor)]
        )
        problem = (
            PassCompiler().prepare(System(elements=[element], interactions=kept, time=StaticTime())).run().problem_
        )
        if not isinstance(problem, ProblemSpec):
            error_msg = f'The network compiles to a {type(problem).__name__}, not a linear program.'
            raise AnalysisError(error_msg)
        self._problem = problem

    def _parameter_values(self, bounds: Mapping[str, tuple[float, float]]) -> Array:
        """Return the parameters with the bounds of some reactions replaced."""
        problem = self._problem
        values = problem.initial_parameters.copy()
        names = problem.parameters.names
        label = self.network_.label
        for reaction, (lower, upper) in bounds.items():
            identifier = self.network_._identifier(reaction)
            values[names.index(f'{label}.lower_{identifier}')] = lower
            values[names.index(f'{label}.upper_{identifier}')] = upper
        return values

    def _solved(
        self,
        weights: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        *,
        sense: float = -1.0,
    ) -> tuple[float, Array]:
        """Return the optimum of the first objective with its weights and bounds replaced, and the fluxes.

        Args:
            weights:
                The coefficient of every reaction in the objective, or `None` for the objective of the network.

            bounds:
                The bounds to replace, by reaction.

            sense:
                `-1` to maximise and `1` to minimise.

        Returns:
            optimum:
                The value of the objective.

            fluxes:
                The flux of every reaction.
        """
        problem = self._problem
        kernel = problem.programs[0]
        program = kernel.program
        if weights is not None:
            costs = program.costs.copy()
            costs[0] = 0.0
            for reaction, weight in weights.items():
                column = list(self.network_.reactions).index(reaction)
                costs[0, column] = sense * float(weight)
            kernel = LinearProgramKernel(replace(program, costs=costs, signs=(sense, *program.signs[1:])))._validated()
            kernel.prepare()
        state = RunStateSpec(0.0, problem.initial_state, self._parameter_values(bounds or {}))
        solved = kernel.run(state).values_
        fluxes = np.asarray([solved[item[0]] for item in program.variables], dtype=float)
        return float(solved[program.objectives[0][0]]), fluxes


class FluxVariabilityAnalysis(BaseNetworkAnalysis):
    """Range of every flux at an optimum, from the solve of the core, which runs without the extra.

    Every flux is maximised and minimised with the objective kept at a fraction of its optimum, `c' v >= f` (Raman 2021,
    Eq. 8.11d). A network of a genome's size is faster with `CobraAnalysis(kind='variability')`.

    Args:
        fraction:
            The fraction of the optimum to keep.

        network:
            The label of the network, or `None` for the first network of the system.

    Attributes:
        optimum_:
            The optimum of the objective.

        ranges_:
            The columns `minimum` and `maximum` of every reaction.

    Examples:
        >>> from skcomplex.core.components.spaces import StaticTime
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.common.metabolism import (
        ...     FluxVariabilityAnalysis,
        ...     ReactionSpec,
        ...     metabolic_network_type,
        ... )
        >>> reactions = {
        ...     'in': ReactionSpec({'A': 1.0}, upper=4.0),
        ...     'one': ReactionSpec({'A': -1.0}),
        ...     'two': ReactionSpec({'A': -1.0}),
        ... }
        >>> network = metabolic_network_type('TwoWaysElement', reactions, {'in': 1.0})('net')
        >>> system = System(elements=[network], time=StaticTime())
        >>> FluxVariabilityAnalysis().prepare(system).run().ranges_.loc['one'].tolist()
        [0.0, 4.0]
    """

    def __init__(self, fraction: float = 1.0, *, network: str | None = None) -> None:
        super().__init__(network=network)
        self.fraction = fraction

    def _run(self, inputs: object = None) -> None:
        """Set the optimum and the range of every flux.

        Args:
            inputs:
                Not read.
        """
        self._compiled_network()
        self.optimum_, _ = self._solved()
        self._compiled_network(floor=self.fraction * self.optimum_)
        rows = {}
        for reaction in self.network_.reactions:
            low = self._solved({reaction: 1.0}, sense=1.0)[0]
            high = self._solved({reaction: 1.0}, sense=-1.0)[0]
            rows[reaction] = (low, high)
        self.ranges_ = pd.DataFrame.from_dict(rows, orient='index', columns=['minimum', 'maximum'])


class ProductionScanAnalysis(BaseNetworkAnalysis):
    """Scan of fluxes with enforced objective flux (FSEOF), from the solve of the core, which runs without the extra.

    The target flux is enforced at increasing levels from its value at the optimum to its maximum, and every level
    maximises the objective, with the total absolute flux minimised after it. The analysis reports the reactions whose
    absolute flux rises with every level.

    Args:
        target:
            The reaction whose flux the scan enforces.

        steps:
            The number of levels.

        network:
            The label of the network, or `None` for the first network of the system.

    Attributes:
        levels_:
            The enforced flux of every level.

        objectives_:
            The optimum of the objective at every level.

        fluxes_:
            The fluxes by reaction, one row per reaction and one column per level.

        reactions_:
            The reactions whose absolute flux rises with every level.

    Examples:
        >>> from skcomplex.core.components.spaces import StaticTime
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.common.metabolism import (
        ...     ProductionScanAnalysis,
        ...     ReactionSpec,
        ...     metabolic_network_type,
        ... )
        >>> reactions = {
        ...     'in': ReactionSpec({'A': 1.0}, upper=4.0),
        ...     'grow': ReactionSpec({'A': -1.0}),
        ...     'make': ReactionSpec({'A': -1.0, 'P': 1.0}),
        ...     'out': ReactionSpec({'P': -1.0}),
        ... }
        >>> network = metabolic_network_type('ScanElement', reactions, {'grow': 1.0})('net')
        >>> system = System(elements=[network], time=StaticTime())
        >>> ProductionScanAnalysis('out', steps=2).prepare(system).run().reactions_
        ['make', 'out']
    """

    def __init__(self, target: str = '', *, steps: int = 10, network: str | None = None) -> None:
        super().__init__(network=network)
        self.target = target
        self.steps = steps

    def _run(self, inputs: object = None) -> None:
        """Set the levels, the objective and the fluxes of every level, and the reactions whose flux rises.

        Args:
            inputs:
                Not read.

        Raises:
            AnalysisError:
                The network holds no target reaction.
        """
        reactions = list(self.network_.reactions)
        if self.target not in reactions:
            error_msg = f'The network holds no reaction {self.target!r}, the target of the scan.'
            raise AnalysisError(error_msg)
        self._compiled_network(parsimonious=True)
        start = self._solved()[1][reactions.index(self.target)]
        maximum = self._solved({self.target: 1.0})[0]
        self.levels_ = start + (maximum - start) * np.arange(int(self.steps) + 1) / int(self.steps)
        solved = [self._solved(None, {self.target: (float(level), float(level))}) for level in self.levels_[:-1]]
        self.objectives_ = np.asarray([item[0] for item in solved], dtype=float)
        amounts = np.stack([item[1] for item in solved], axis=1)
        self.fluxes_ = pd.DataFrame(amounts, index=reactions)
        rising = np.all(np.diff(np.abs(amounts), axis=1) > LETHAL_GROWTH, axis=1)
        self.reactions_ = [name for name, rises in zip(reactions, rising, strict=True) if rises]


class CobraAnalysis(BaseNetworkAnalysis):
    """Analysis of a metabolic network with COBRApy, which needs the extra `metabolism`.

    - `'variability'`: the range of every flux at a fraction of the optimum.
    - `'parsimonious'`: the fluxes of the smallest total flux at the optimum.
    - `'deletions'`: the growth after single or double deletions of genes or reactions.
    - `'moma'` and `'room'`: the fluxes of a knock-out adjusted against the wild-type solution.
    - `'synthetic-lethals'`: the pairs of genes or reactions lethal together, but not alone.

    Args:
        kind:
            The analysis, one of the kinds above.

        fraction:
            The fraction of the optimum for `'variability'`.

        target:
            `'gene'` or `'reaction'`, what `'deletions'` and `'synthetic-lethals'` delete.

        order:
            `1` or `2`, the size of the deletions.

        knockouts:
            The genes to knock out for `'moma'` and `'room'`.

        linear:
            Whether to solve MoMA and ROOM in their linear forms, rather than the quadratic form of MoMA and the
            mixed-integer form of ROOM, with `delta = 0.03` and `epsilon = 0.001` (Raman 2021, Eqs. 9.1 and 9.5).

        network:
            The label of the network, or `None` for the first network of the system.

    Attributes:
        system_:
            The `cobra.Model` of the network.

        ranges_:
            The columns `minimum` and `maximum` of every reaction, for `'variability'`.

        fluxes_:
            The flux of every reaction, for `'parsimonious'`, `'moma'`, and `'room'`.

        total_:
            The total absolute flux, for `'parsimonious'`.

        growth_:
            The growth of every deletion, by the name of the gene or reaction or by a pair of names, for `'deletions'`,
            and the adjusted growth for `'moma'` and `'room'`.

        pairs_:
            The lethal pairs, for `'synthetic-lethals'`.

    Raises:
        AnalysisError:
            The kind is unknown, raised by `prepare`.

        MissingExtraError:
            The extra is not installed, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.components.spaces import StaticTime
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.common.metabolism import CobraAnalysis, load_network
        >>> system = System(elements=[load_network('textbook', 'cell')], time=StaticTime())
        >>> analysis = CobraAnalysis('parsimonious').prepare(system).run()
        >>> round(float(analysis.total_), 3)
        518.422
    """

    def __init__(
        self,
        kind: str = 'variability',
        *,
        fraction: float = 1.0,
        target: str = 'gene',
        order: int = 1,
        knockouts: Sequence[str] = (),
        linear: bool = False,
        network: str | None = None,
    ) -> None:
        super().__init__(network=network)
        self.kind = kind
        self.fraction = fraction
        self.target = target
        self.order = order
        self.knockouts = knockouts
        self.linear = linear

    def _check(self) -> None:
        """Raise an error for an unknown kind or a missing extra.

        Raises:
            AnalysisError:
                The kind is unknown.

            MissingExtraError:
                The extra is not installed.
        """
        self._check_choice('kind', self.kind, COBRA_KINDS)
        self._check_choice('target', self.target, ('gene', 'reaction'))
        load_extra(f'{type(self).__name__}(kind={self.kind!r})')

    def _prepare(self, context: Source) -> None:
        """Read the network of a system and convert it to a COBRApy model.

        Args:
            context:
                The system.
        """
        super()._prepare(context)
        self.system_ = to_cobra(self.network_, needed_by=type(self).__name__)

    def _deletions(self, order: int) -> pd.Series:
        """Return the growth of every deletion of some size, by the name or the pair of names."""
        cobra = load_extra(type(self).__name__)
        functions = {
            ('gene', 1): cobra.flux_analysis.single_gene_deletion,
            ('gene', PAIR_SIZE): cobra.flux_analysis.double_gene_deletion,
            ('reaction', 1): cobra.flux_analysis.single_reaction_deletion,
            ('reaction', PAIR_SIZE): cobra.flux_analysis.double_reaction_deletion,
        }
        table = functions[self.target, order](self.system_, processes=1)
        keys = [next(iter(ids)) if order == 1 else tuple(sorted(ids)) for ids in table['ids']]
        return pd.Series(table['growth'].fillna(0.0).to_numpy(dtype=float), index=keys)

    def _run_variability(self) -> None:
        """Set the range of every flux."""
        cobra = load_extra(type(self).__name__)
        table = cobra.flux_analysis.flux_variability_analysis(
            self.system_,
            fraction_of_optimum=self.fraction,
            processes=1,
        )
        self.ranges_ = pd.DataFrame({'minimum': table['minimum'], 'maximum': table['maximum']})

    def _run_parsimonious(self) -> None:
        """Set the fluxes of the smallest total flux."""
        cobra = load_extra(type(self).__name__)
        solution = cobra.flux_analysis.pfba(self.system_)
        self.fluxes_ = solution.fluxes
        self.total_ = float(solution.fluxes.abs().sum())

    def _run_adjustment(self) -> None:
        """Set the fluxes and the growth of a knock-out adjusted against the wild-type solution."""
        cobra = load_extra(type(self).__name__)
        system = self.system_
        if not isinstance(system, cobra.Model):
            error_msg = f'{type(self).__name__} adjusts a `cobra.Model`, not a {type(system).__name__}.'
            raise AnalysisError(error_msg)
        reference = system.optimize()
        function = cobra.flux_analysis.moma if self.kind == 'moma' else cobra.flux_analysis.room
        objective = next(reaction for reaction in system.reactions if reaction.objective_coefficient != 0)
        with system:
            for gene in self.knockouts:
                system.genes.get_by_id(gene).knock_out()
            solution = function(system, solution=reference, linear=self.linear)
        self.fluxes_ = solution.fluxes
        self.growth_ = float(solution.fluxes[objective.id])

    def _run_lethals(self) -> None:
        """Set the pairs lethal together but not alone."""
        alone = self._deletions(1)
        essential = set(alone.index[alone < LETHAL_GROWTH])
        together = self._deletions(PAIR_SIZE)
        self.pairs_ = [
            pair
            for pair, growth in together.items()
            if isinstance(pair, tuple)
            and len(pair) == PAIR_SIZE
            and growth < LETHAL_GROWTH
            and not essential.intersection(pair)
        ]

    def _run(self, inputs: object = None) -> None:
        """Run the kind of the analysis.

        Args:
            inputs:
                Not read.
        """
        if self.kind == 'variability':
            self._run_variability()
        elif self.kind == 'parsimonious':
            self._run_parsimonious()
        elif self.kind == 'deletions':
            self.growth_ = self._deletions(int(self.order))
        elif self.kind in {'moma', 'room'}:
            self._run_adjustment()
        else:
            self._run_lethals()


class StrainDesignAnalysis(BaseNetworkAnalysis):
    """Strain designs of a metabolic network with StrainDesign, which needs the extra `metabolism`.

    - `'optknock'`, `'robustknock'`, and `'optcouple'`: the knock-outs that couple a product to growth.
    - `'mcs'`: the minimal cut sets that suppress the flux states of a constraint.

    Args:
        kind:
            The design, one of the kinds above.

        product:
            The reaction of the product, the outer objective.

        growth:
            The reaction of growth, the inner objective, or `None` for the objective of the network.

        constraints:
            The constraints of the design in the syntax of StrainDesign, such as `'Biomass >= 0.1'`.

        interventions:
            The largest number of interventions.

        solutions:
            The largest number of designs.

        network:
            The label of the network, or `None` for the first network of the system.

    Attributes:
        designs_:
            Every design as a set of reaction names.

    Raises:
        AnalysisError:
            The kind is unknown, raised by `prepare`.

        MissingExtraError:
            The extra is not installed, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import StrainDesignAnalysis
        >>> StrainDesignAnalysis('optknock', product='EX_succ_e').interventions
        3
    """

    def __init__(
        self,
        kind: str = 'optknock',
        *,
        product: str = '',
        growth: str | None = None,
        constraints: str = '',
        interventions: int = 3,
        solutions: int = 1,
        network: str | None = None,
    ) -> None:
        super().__init__(network=network)
        self.kind = kind
        self.product = product
        self.growth = growth
        self.constraints = constraints
        self.interventions = interventions
        self.solutions = solutions

    def _check(self) -> None:
        """Raise an error for an unknown kind or a missing extra.

        Raises:
            AnalysisError:
                The kind is unknown.

            MissingExtraError:
                The extra is not installed.
        """
        self._check_choice('kind', self.kind, DESIGN_KINDS)
        load_extra(f'{type(self).__name__}(kind={self.kind!r})')

    def _prepare(self, context: Source) -> None:
        """Read the network of a system and convert it to a COBRApy model.

        Args:
            context:
                The system.
        """
        super()._prepare(context)
        self.system_ = to_cobra(self.network_, needed_by=type(self).__name__)

    def _run(self, inputs: object = None) -> None:
        """Compute the designs.

        Args:
            inputs:
                Not read.

        Raises:
            MissingExtraError:
                StrainDesign is not installed.
        """
        straindesign = import_extra('straindesign', type(self).__name__, 'metabolism')
        weights = self.network_.default_objective if self.network_.objective is None else self.network_.objective
        growth = self.growth or next(iter(weights))
        kind = getattr(straindesign.names, DESIGN_NAMES[self.kind])
        options: dict[str, object] = {'constraints': self.constraints}
        if self.kind == 'optcouple':
            options |= {'inner_objective': growth, 'prod_id': self.product}
        elif self.kind != 'mcs':
            options |= {'inner_objective': growth, 'outer_objective': self.product}
        module = straindesign.SDModule(self.system_, kind, **options)
        found = straindesign.compute_strain_designs(
            self.system_,
            sd_modules=module,
            max_solutions=int(self.solutions),
            max_cost=int(self.interventions),
            solution_approach=straindesign.names.BEST,
            solver='scip',
        )
        self.designs_ = [set(design) for design in found.reaction_sd]
