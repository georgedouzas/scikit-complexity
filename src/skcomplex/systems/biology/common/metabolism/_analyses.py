"""Analyse metabolic networks.

The analyses are flux variability, flux scans, COBRApy methods, and strain designs.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.base import clone

from .....core.analysis.types import Source
from .....core.components.elements import BaseElement, Element
from .....core.engine.problems import ProblemSpec, RunStateSpec
from .....core.errors import AnalysisError
from .....core.systems import BaseSystem, System
from .....core.types import Array
from ...._extras import import_extra
from ._cobra_io import load_extra, to_cobra
from ._culture import ObjectiveFloorInteraction
from ._flux_balance import FluxBalanceInteraction

COBRA_KINDS = ('variability', 'parsimonious', 'deletions', 'moma', 'room', 'synthetic-lethals')
DESIGN_KINDS = ('optknock', 'robustknock', 'optcouple', 'mcs')
DESIGN_NAMES = {'optknock': 'OPTKNOCK', 'robustknock': 'ROBUSTKNOCK', 'optcouple': 'OPTCOUPLE', 'mcs': 'SUPPRESS'}
LETHAL_GROWTH = 1e-6
PAIR_SIZE = 2


@dataclass(frozen=True)
class FluxVariabilityResult:
    """Range of every flux of a metabolic network at a fraction of its optimum.

    Args:
        network:
            The flux balance of the metabolic network.

        system:
            The system that holds the network.

        optimum:
            The optimum of the objective.

        ranges:
            The columns `minimum` and `maximum` of every reaction.
    """

    network: FluxBalanceInteraction
    system: System
    optimum: float
    ranges: pd.DataFrame


def flux_variability(source: Source, *, fraction: float = 1.0, network: str | None = None) -> FluxVariabilityResult:
    """Return the range of every flux at an optimum.

    The ranges come from the solve of the core, which runs without the extra.

    Every flux is maximised and minimised with the objective kept at a fraction of its optimum, `c' v >= f` (Raman 2021,
    Eq. 8.11d). A network of a genome's size is faster with `cobra_measures(system, kind='variability')`.

    Args:
        source:
            The system that holds the metabolic network.

        fraction:
            The fraction of the optimum to keep.

        network:
            The label of the network, or `None` for the first network of the system.

    Returns:
        result:
            The network, the optimum, and the range of every flux.

    Raises:
        AnalysisError:
            The source is not a system with the network.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.biology.common.metabolism import (
        ...     FluxBalanceInteraction,
        ...     ReactionSpec,
        ...     flux_variability,
        ... )
        >>> reactions = {
        ...     'in': ReactionSpec({'A': 1.0}, upper=4.0),
        ...     'one': ReactionSpec({'A': -1.0}),
        ...     'two': ReactionSpec({'A': -1.0}),
        ... }
        >>> network = FluxBalanceInteraction('fba', cell='net', reactions=reactions, objective={'in': 1.0})
        >>> system = System(elements=[Element('net', values=network.get_values())], interactions=[network])
        >>> flux_variability(system).ranges.loc['one'].tolist()
        [0.0, 4.0]
    """
    system, interaction, cell = _read_network(source, network, 'flux_variability')
    problem = _compiled_network(interaction, cell)
    optimum, _ = _solved(problem, interaction, cell)
    problem = _compiled_network(interaction, cell, floor=fraction * optimum)
    rows = {}
    for reaction in interaction.reactions:
        low = _solved(problem, interaction, cell, {reaction: 1.0}, sense=1.0)[0]
        high = _solved(problem, interaction, cell, {reaction: 1.0}, sense=-1.0)[0]
        rows[reaction] = (low, high)
    ranges = pd.DataFrame.from_dict(rows, orient='index', columns=['minimum', 'maximum'])
    return FluxVariabilityResult(interaction, system, optimum, ranges)


@dataclass(frozen=True)
class ProductionScanResult:
    """Fluxes of a metabolic network at increasing levels of an enforced target flux.

    Args:
        network:
            The flux balance of the metabolic network.

        system:
            The system that holds the network.

        levels:
            The enforced flux of every level.

        objectives:
            The optimum of the objective at every level.

        fluxes:
            The fluxes by reaction, one row per reaction and one column per level.

        reactions:
            The reactions whose absolute flux rises with every level.
    """

    network: FluxBalanceInteraction
    system: System
    levels: Array
    objectives: Array
    fluxes: pd.DataFrame
    reactions: list[str]


def production_scan(
    source: Source,
    *,
    target: str = '',
    steps: int = 10,
    network: str | None = None,
) -> ProductionScanResult:
    """Return the scan of fluxes with enforced objective flux (FSEOF).

    The scan comes from the solve of the core, without the extra.

    The target flux is enforced at increasing levels from its value at the optimum to its maximum, and every level
    maximises the objective, with the total absolute flux minimised after it. The scan reports the reactions whose
    absolute flux rises with every level.

    Args:
        source:
            The system that holds the metabolic network.

        target:
            The reaction whose flux the scan enforces.

        steps:
            The number of levels.

        network:
            The label of the network, or `None` for the first network of the system.

    Returns:
        result:
            The levels, the objective and the fluxes of every level, and the reactions whose flux rises.

    Raises:
        AnalysisError:
            The source is not a system with the network, or the network holds no target reaction.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.biology.common.metabolism import (
        ...     FluxBalanceInteraction,
        ...     ReactionSpec,
        ...     production_scan,
        ... )
        >>> reactions = {
        ...     'in': ReactionSpec({'A': 1.0}, upper=4.0),
        ...     'grow': ReactionSpec({'A': -1.0}),
        ...     'make': ReactionSpec({'A': -1.0, 'P': 1.0}),
        ...     'out': ReactionSpec({'P': -1.0}),
        ... }
        >>> network = FluxBalanceInteraction('fba', cell='net', reactions=reactions, objective={'grow': 1.0})
        >>> system = System(elements=[Element('net', values=network.get_values())], interactions=[network])
        >>> production_scan(system, target='out', steps=2).reactions
        ['make', 'out']
    """
    system, interaction, cell = _read_network(source, network, 'production_scan')
    reactions = list(interaction.reactions)
    if target not in reactions:
        error_msg = f'The network holds no reaction {target!r}, the target of the scan.'
        raise AnalysisError(error_msg)
    problem = _compiled_network(interaction, cell, parsimonious=True)
    start = _solved(problem, interaction, cell)[1][reactions.index(target)]
    maximum = _solved(problem, interaction, cell, {target: 1.0})[0]
    levels = start + (maximum - start) * np.arange(int(steps) + 1) / int(steps)
    solved = [
        _solved(problem, interaction, cell, None, {target: (float(level), float(level))}) for level in levels[:-1]
    ]
    objectives = np.asarray([item[0] for item in solved], dtype=float)
    amounts = np.stack([item[1] for item in solved], axis=1)
    fluxes = pd.DataFrame(amounts, index=reactions)
    rising = np.all(np.diff(np.abs(amounts), axis=1) > LETHAL_GROWTH, axis=1)
    found = [name for name, rises in zip(reactions, rising, strict=True) if rises]
    return ProductionScanResult(interaction, system, levels, objectives, fluxes, found)


@dataclass(frozen=True)
class CobraMeasuresResult:
    """Result of a COBRApy method on a metabolic network.

    Args:
        network:
            The flux balance of the metabolic network.

        system:
            The `cobra.Model` of the network.

        ranges:
            The columns `minimum` and `maximum` of every reaction, for `'variability'`, or `None`.

        fluxes:
            The flux of every reaction, for `'parsimonious'`, `'moma'`, and `'room'`, or `None`.

        total:
            The total absolute flux, for `'parsimonious'`, or `None`.

        growth:
            The growth of every deletion, by the name of the gene or reaction or by a pair of names, for `'deletions'`,
            and the adjusted growth for `'moma'` and `'room'`, or `None`.

        pairs:
            The lethal pairs, for `'synthetic-lethals'`, or `None`.
    """

    network: FluxBalanceInteraction
    system: object
    ranges: pd.DataFrame | None = None
    fluxes: pd.Series | None = None
    total: float | None = None
    growth: pd.Series | float | None = None
    pairs: list[tuple[str, ...]] | None = None


def cobra_measures(
    source: Source,
    *,
    kind: str = 'variability',
    fraction: float = 1.0,
    target: str = 'gene',
    order: int = 1,
    knockouts: Sequence[str] = (),
    linear: bool = False,
    network: str | None = None,
) -> CobraMeasuresResult:
    """Return the analysis of a metabolic network with COBRApy.

    It needs the extra `metabolism`.

    - `'variability'`: the range of every flux at a fraction of the optimum.
    - `'parsimonious'`: the fluxes of the smallest total flux at the optimum.
    - `'deletions'`: the growth after single or double deletions of genes or reactions.
    - `'moma'` and `'room'`: the fluxes of a knock-out adjusted against the wild-type solution.
    - `'synthetic-lethals'`: the pairs of genes or reactions lethal together, but not alone.

    Args:
        source:
            The system that holds the metabolic network.

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

    Returns:
        result:
            The network, its `cobra.Model`, and the values of the kind.

    Raises:
        AnalysisError:
            The kind or the target is unknown, or the source is not a system with the network.

        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.biology.common.metabolism import cobra_measures, load_network
        >>> network = load_network('textbook', 'cell')
        >>> system = System(elements=[Element('cell', values=network.get_values())], interactions=[network])
        >>> round(float(cobra_measures(system, kind='parsimonious').total), 3)
        518.422
    """
    _check_choice('cobra_measures', 'kind', kind, COBRA_KINDS)
    _check_choice('cobra_measures', 'target', target, ('gene', 'reaction'))
    load_extra(f'cobra_measures(kind={kind!r})')
    _, element, cell = _read_network(source, network, 'cobra_measures')
    model = to_cobra(element, cell, needed_by='cobra_measures')
    cobra = load_extra('cobra_measures')
    if kind == 'variability':
        table = cobra.flux_analysis.flux_variability_analysis(model, fraction_of_optimum=fraction, processes=1)
        ranges = pd.DataFrame({'minimum': table['minimum'], 'maximum': table['maximum']})
        return CobraMeasuresResult(element, model, ranges=ranges)
    if kind == 'parsimonious':
        solution = cobra.flux_analysis.pfba(model)
        return CobraMeasuresResult(
            element,
            model,
            fluxes=solution.fluxes,
            total=float(solution.fluxes.abs().sum()),
        )
    if kind == 'deletions':
        return CobraMeasuresResult(element, model, growth=_deletions(model, target, int(order)))
    if kind in {'moma', 'room'}:
        fluxes, growth = _adjusted(model, kind, knockouts, linear=linear)
        return CobraMeasuresResult(element, model, fluxes=fluxes, growth=growth)
    alone = _deletions(model, target, 1)
    essential = set(alone.index[alone < LETHAL_GROWTH])
    together = _deletions(model, target, PAIR_SIZE)
    pairs = [
        pair
        for pair, growth in together.items()
        if isinstance(pair, tuple)
        and len(pair) == PAIR_SIZE
        and growth < LETHAL_GROWTH
        and not essential.intersection(pair)
    ]
    return CobraMeasuresResult(element, model, pairs=pairs)


@dataclass(frozen=True)
class StrainDesignResult:
    """Strain designs of a metabolic network.

    Args:
        network:
            The flux balance of the metabolic network.

        system:
            The `cobra.Model` of the network.

        designs:
            Every design as a set of reaction names.
    """

    network: FluxBalanceInteraction
    system: object
    designs: list[set[str]]


def strain_design(
    source: Source,
    *,
    kind: str = 'optknock',
    product: str = '',
    growth: str | None = None,
    constraints: str = '',
    interventions: int = 3,
    solutions: int = 1,
    network: str | None = None,
) -> StrainDesignResult:
    """Return the strain designs of a metabolic network with StrainDesign.

    It needs the extra `metabolism`.

    - `'optknock'`, `'robustknock'`, and `'optcouple'`: the knock-outs that couple a product to growth.
    - `'mcs'`: the minimal cut sets that suppress the flux states of a constraint.

    Args:
        source:
            The system that holds the metabolic network.

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

    Returns:
        result:
            The network, its `cobra.Model`, and the designs.

    Raises:
        AnalysisError:
            The kind is unknown, or the source is not a system with the network.

        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.core.errors import AnalysisError
        >>> from skcomplex.systems.biology.common.metabolism import strain_design
        >>> try:
        ...     strain_design(None, kind='minimal', product='EX_succ_e')
        ... except AnalysisError as error:
        ...     print(str(error).split('. ')[0])
        strain_design has kind='minimal'
    """
    _check_choice('strain_design', 'kind', kind, DESIGN_KINDS)
    load_extra(f'strain_design(kind={kind!r})')
    _, element, cell = _read_network(source, network, 'strain_design')
    model = to_cobra(element, cell, needed_by='strain_design')
    straindesign = import_extra('straindesign', 'strain_design', 'metabolism')
    inner = growth or next(iter(element.objective))
    module_kind = getattr(straindesign.names, DESIGN_NAMES[kind])
    options: dict[str, object] = {'constraints': constraints}
    if kind == 'optcouple':
        options |= {'inner_objective': inner, 'prod_id': product}
    elif kind != 'mcs':
        options |= {'inner_objective': inner, 'outer_objective': product}
    module = straindesign.SDModule(model, module_kind, **options)
    found = straindesign.compute_strain_designs(
        model,
        sd_modules=module,
        max_solutions=int(solutions),
        max_cost=int(interventions),
        solution_approach=straindesign.names.BEST,
        solver='scip',
    )
    return StrainDesignResult(element, model, [set(design) for design in found.reaction_sd])


def _check_choice(analysis: str, name: str, value: object, choices: tuple[object, ...]) -> None:
    """Raise an error for a setting of an analysis that is not one of its choices."""
    if value not in choices:
        error_msg = f'{analysis} has {name}={value!r}. Use one of {", ".join(map(str, choices))}.'
        raise AnalysisError(error_msg)


def _networks_of(system: System) -> list[tuple[FluxBalanceInteraction, Element]]:
    """Return the flux balances of a system and its nested systems, each with its cell."""
    cells = {item.label: item for item in system.elements if isinstance(item, BaseElement)}
    found = [
        (item, cells[item.cell])
        for item in system.interactions
        if isinstance(item, FluxBalanceInteraction) and isinstance(cells.get(item.cell), Element)
    ]
    for nested in system.systems:
        if isinstance(nested, System):
            found += _networks_of(nested)
    return found


def _read_network(
    source: Source,
    network: str | None,
    analysis: str,
) -> tuple[System, FluxBalanceInteraction, Element]:
    """Return a system and the flux balance of its metabolic network of a label, or its first one, with its cell.

    Args:
        source:
            The system.

        network:
            The label of the cell of the network, or `None` for the first network of the system.

        analysis:
            The name of the analysis, for the error message.

    Returns:
        system:
            The system.

        network:
            The flux balance of the metabolic network.

        cell:
            The cell whose values hold the fluxes, the bounds, and the switches.

    Raises:
        AnalysisError:
            The source is not a system with the network.
    """
    if isinstance(source, BaseSystem) and not isinstance(source, System):
        source = source.prepare().system_
    if not isinstance(source, System):
        error_msg = f'{analysis} reads a system that holds the flux balance of a network, not {type(source).__name__}.'
        raise AnalysisError(error_msg)
    chosen = [item for item in _networks_of(source) if network is None or item[1].label == network]
    if not chosen:
        error_msg = f'{analysis} reads a system that holds the flux balance of a network.'
        raise AnalysisError(error_msg)
    return source, *chosen[0]


def _compiled_network(
    network: FluxBalanceInteraction,
    cell: Element,
    *,
    parsimonious: bool = False,
    floor: float | None = None,
) -> ProblemSpec:
    """Return the network compiled alone.

    Its objective is kept above a floor if one is given.
    """
    interaction = clone(network).set_params(parsimonious=parsimonious, infeasible='raise')
    kept = (
        []
        if floor is None
        else [ObjectiveFloorInteraction('kept', cell=cell.label, weights=network.objective, floor=floor)]
    )
    problem = System(elements=[cell], interactions=[interaction, *kept]).prepare().simulate(until=0.0).problem
    if not isinstance(problem, ProblemSpec):
        error_msg = f'The network compiles to a {type(problem).__name__}, not a linear program.'
        raise AnalysisError(error_msg)
    return problem


def _parameter_values(
    problem: ProblemSpec,
    network: FluxBalanceInteraction,
    cell: Element,
    bounds: Mapping[str, tuple[float, float]],
) -> Array:
    """Return the parameters with the bounds of some reactions replaced."""
    values = problem.initial_parameters.copy()
    names = problem.parameters.names
    for reaction, (lower, upper) in bounds.items():
        identifier = network._get_identifier(reaction)
        values[names.index(f'{cell.label}.lower_{identifier}')] = lower
        values[names.index(f'{cell.label}.upper_{identifier}')] = upper
    return values


def _solved(
    problem: ProblemSpec,
    network: FluxBalanceInteraction,
    cell: Element,
    weights: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    sense: float = -1.0,
) -> tuple[float, Array]:
    """Return the optimum of the first objective and the fluxes.

    The optimum is solved with the weights and bounds of the objective replaced.

    Args:
        problem:
            The compiled problem of the network.

        network:
            The flux balance of the metabolic network.

        cell:
            The cell whose values hold the fluxes, the bounds, and the switches.

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
    kernel = problem.programs[0]
    program = kernel.program
    costs = program.costs.copy()
    costs[0] = 0.0
    for reaction, weight in (network.objective if weights is None else weights).items():
        column = list(network.reactions).index(reaction)
        costs[0, column] = sense * float(weight)
    kernel = clone(kernel).set_params(program=replace(program, costs=costs, signs=(sense, *program.signs[1:])))
    kernel.prepare()
    state = RunStateSpec(0.0, problem.initial_state, _parameter_values(problem, network, cell, bounds or {}))
    solved = kernel.run(state).values_
    fluxes = np.asarray([solved[item[0]] for item in program.variables], dtype=float)
    return float(solved[program.objectives[0][0]]), fluxes


def _deletions(model: object, target: str, order: int) -> pd.Series:
    """Return the growth of every deletion of some size of a `cobra.Model`.

    The growth is keyed by the name or the pair of names.
    """
    cobra = load_extra('cobra_measures')
    functions = {
        ('gene', 1): cobra.flux_analysis.single_gene_deletion,
        ('gene', PAIR_SIZE): cobra.flux_analysis.double_gene_deletion,
        ('reaction', 1): cobra.flux_analysis.single_reaction_deletion,
        ('reaction', PAIR_SIZE): cobra.flux_analysis.double_reaction_deletion,
    }
    table = functions[target, order](model, processes=1)
    keys = [next(iter(ids)) if order == 1 else tuple(sorted(ids)) for ids in table['ids']]
    return pd.Series(table['growth'].fillna(0.0).to_numpy(dtype=float), index=keys)


def _adjusted(model: object, kind: str, knockouts: Sequence[str], *, linear: bool) -> tuple[pd.Series, float]:
    """Return the fluxes and the growth of a knock-out.

    They are adjusted against the wild-type solution.
    """
    cobra = load_extra('cobra_measures')
    if not isinstance(model, cobra.Model):
        error_msg = f'cobra_measures adjusts a `cobra.Model`, not a {type(model).__name__}.'
        raise AnalysisError(error_msg)
    reference = model.optimize()
    function = cobra.flux_analysis.moma if kind == 'moma' else cobra.flux_analysis.room
    objective = next(reaction for reaction in model.reactions if reaction.objective_coefficient != 0)
    with model:
        for gene in knockouts:
            model.genes.get_by_id(gene).knock_out()
        solution = function(model, solution=reference, linear=linear)
    return solution.fluxes, float(solution.fluxes[objective.id])
