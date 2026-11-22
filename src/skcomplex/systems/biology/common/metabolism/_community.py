"""Analyse communities of metabolic networks.

The analyses are SteadyCom on the solve of the core, and the trade-off of MICOM.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sage.all__sagemath_symbolics import Expression
from sklearn.base import clone

from .....core.analysis.types import DataSource
from .....core.components.elements import Element
from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, ConstraintLaw, ObjectiveLaw
from .....core.engine.problems import ProblemSpec, RunStateSpec
from .....core.errors import AnalysisError
from .....core.systems import BaseSystem, System
from ...._extras import import_extra
from ._analyses import _check_choice, _networks_of
from ._cobra_io import load_extra, to_cobra
from ._culture import CommunityInteraction
from ._flux_balance import FluxBalanceInteraction

COMMUNITY_KINDS = ('steadycom', 'tradeoff')
RATE_TOLERANCE = 1e-7
FULL_ABUNDANCE = 1.0 - 1e-7
ELEMENT_LABEL = 'community'

type Member = tuple[FluxBalanceInteraction, Element]


@dataclass(frozen=True)
class CommunityGrowthResult:
    """Growth of a community of metabolic networks that share a medium.

    Args:
        growth:
            The common growth rate, or the community growth rate of the trade-off.

        abundances:
            The abundance of every member, by its label, for `'steadycom'`, or `None`.

        members:
            The growth of every member, by its label, for `'tradeoff'`, or `None`.
    """

    growth: float
    abundances: dict[str, float] | None = None
    members: dict[str, float] | None = None


def community_growth(
    source: DataSource,
    *,
    kind: str = 'steadycom',
    fraction: float = 1.0,
    abundances: Sequence[float] | None = None,
    medium: Mapping[str, float] | None = None,
) -> CommunityGrowthResult:
    """Return the growth of a community of metabolic networks that share a medium.

    - `'steadycom'`: the largest common growth rate of the members, after Chan et al. (2017), by a bisection on the rate
      with the solve of the core, without the extra. The fluxes of a member scale with its abundance, the abundances
      sum to at most 1, and a shared exchange of the medium stays above the bound that `medium` gives.
    - `'tradeoff'`: the cooperative trade-off of MICOM, which needs the extra `metabolism`.

    Args:
        source:
            The system whose metabolic networks are the members.

        kind:
            The analysis, one of the kinds above.

        fraction:
            The fraction of the community growth rate that the trade-off keeps.

        abundances:
            The abundance of every member for `'tradeoff'`, or `None` for equal ones.

        medium:
            The lower bound of every shared exchange of the whole community for `'steadycom'`, by reaction, or `None`
            for no shared bound.

    Returns:
        result:
            The growth rate, and the abundances or the growth of every member.

    Raises:
        AnalysisError:
            The kind is unknown, or the source is not a system with a metabolic network.

        MissingExtraError:
            The kind `'tradeoff'` needs the extra.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.biology.common.metabolism import (
        ...     FluxBalanceInteraction,
        ...     ReactionSpec,
        ...     community_growth,
        ... )
        >>> reactions = {
        ...     'EX_a': ReactionSpec({'a': 1.0}, upper=2.0),
        ...     'grow': ReactionSpec({'a': -1.0}),
        ... }
        >>> networks = [
        ...     FluxBalanceInteraction(f'fba_{name}', cell=name, reactions=reactions, objective={'grow': 1.0})
        ...     for name in ('one', 'two')
        ... ]
        >>> cells = [Element(network.cell, values=network.get_values()) for network in networks]
        >>> result = community_growth(System(elements=cells, interactions=networks))
        >>> round(result.growth, 6), round(sum(result.abundances.values()), 6)
        (2.0, 1.0)
    """
    _check_choice('community_growth', 'kind', kind, COMMUNITY_KINDS)
    if kind == 'tradeoff':
        load_extra(f'community_growth(kind={kind!r})')
    if isinstance(source, BaseSystem) and not isinstance(source, System):
        source = source.prepare().system_
    members = _get_members(source) if isinstance(source, System) else []
    if not members:
        error_msg = 'community_growth reads a system that holds the flux balance of a metabolic network.'
        raise AnalysisError(error_msg)
    if kind == 'steadycom':
        growth, found = _steadycom(members, medium)
        return CommunityGrowthResult(growth, abundances=found)
    growth, rates = _tradeoff(members, fraction, abundances)
    return CommunityGrowthResult(growth, members=rates)


def _get_members(system: System) -> list[Member]:
    """Return the flux balances of a system with their cells, each with the objective of its growth.

    A member without an objective of its own, whose growth a joint program of the community holds, grows by the
    reaction of growth of that program.
    """
    growth = {
        member: interaction.growth
        for interaction in system.interactions
        if isinstance(interaction, CommunityInteraction)
        for member in interaction.members
    }
    return [
        (
            (
                network
                if network.objective or cell.label not in growth
                else clone(network).set_params(objective={growth[cell.label]: 1.0})
            ),
            cell,
        )
        for network, cell in _networks_of(system)
    ]


def _get_value(member: Member, name: str) -> float:
    """Return a value of the cell of a member, or the value its network starts with."""
    network, cell = member
    return float((network.get_values() | dict(cell.get_params()['values'] or {}))[name])


def _member_laws(s: SimpleNamespace, position: int, member: Member) -> list[ConstraintLaw]:
    """Return the balances of a member and its growth at the common rate.

    The bounds of the member are scaled by its abundance.
    """
    network = member[0]
    terms: dict[str, list[Expression]] = {}
    relations = []
    abundance = getattr(s, f'x_{position}')
    for column, (name, spec) in enumerate(network.reactions.items()):
        identifier = network._get_identifier(name)
        flux = getattr(s, f'v_{position}_{column}')
        relations.append(flux >= _get_value(member, f'lower_{identifier}') * abundance)
        relations.append(flux <= _get_value(member, f'upper_{identifier}') * abundance)
        for metabolite, coefficient in spec.coefficients.items():
            terms.setdefault(metabolite, []).append(float(coefficient) * flux)
    relations += [sum(items[1:], items[0]) == 0 for items in terms.values()]
    growth = [
        float(weight) * getattr(s, f'v_{position}_{list(network.reactions).index(name)}')
        for name, weight in network.objective.items()
    ]
    relations.append(sum(growth[1:], growth[0]) - s.rate * abundance == 0)
    return [ConstraintLaw(relation) for relation in relations]


def _community_laws(
    members: list[Member],
    medium: Mapping[str, float] | None,
    s: SimpleNamespace,
) -> list[BaseLaw]:
    """Return the program of the abundances of the members.

    The program holds at the common rate of a namespace.
    """
    s = s.target
    laws: list[BaseLaw] = [law for position, member in enumerate(members) for law in _member_laws(s, position, member)]
    abundances = [getattr(s, f'x_{position}') for position in range(len(members))]
    total = sum(abundances[1:], abundances[0])
    laws += [ConstraintLaw(item >= 0) for item in abundances]
    laws.append(ConstraintLaw(total <= 1))
    for name, lower in (medium or {}).items():
        shared = [
            getattr(s, f'v_{position}_{list(member[0].reactions).index(name)}')
            for position, member in enumerate(members)
            if name in member[0].reactions
        ]
        laws.append(ConstraintLaw(sum(shared[1:], shared[0]) >= float(lower)))
    variables = [
        *abundances,
        *(
            getattr(s, f'v_{position}_{column}')
            for position, member in enumerate(members)
            for column in range(len(member[0].reactions))
        ),
    ]
    return [*laws, ObjectiveLaw([-total], variables)]


def _community_problem(members: list[Member], medium: Mapping[str, float] | None) -> ProblemSpec:
    """Return the compiled problem of the community.

    Its program sets the abundances and the scaled fluxes at the common rate `rate`.
    """
    names = [f'x_{position}' for position in range(len(members))]
    names += [
        f'v_{position}_{column}'
        for position, member in enumerate(members)
        for column in range(len(member[0].reactions))
    ]
    community = Element(ELEMENT_LABEL, values={**dict.fromkeys(names, 0.0), 'rate': 0.0})
    program = Interaction(
        'steadycom',
        roles={'target': ELEMENT_LABEL},
        laws=partial(_community_laws, members, medium),
    )
    problem = System(elements=[community], interactions=[program]).prepare().simulate(until=0.0).problem
    if not isinstance(problem, ProblemSpec):
        error_msg = f'The community compiles to a {type(problem).__name__}, not a linear program.'
        raise AnalysisError(error_msg)
    return problem


def _abundances_at(
    members: list[Member],
    problem: ProblemSpec,
    rate: float,
) -> dict[str, float] | None:
    """Return the abundances at a common rate.

    It returns `None` when they cannot reach a sum of 1.
    """
    parameters = problem.initial_parameters.copy()
    parameters[problem.parameters.names.index(f'{ELEMENT_LABEL}.rate')] = rate
    solved = problem.programs[0].run(RunStateSpec(rate, problem.initial_state, parameters)).values_
    found = {
        member[1].label: float(np.real(np.ravel(problem.states.read(solved, f'{ELEMENT_LABEL}.x_{position}'))[0]))
        for position, member in enumerate(members)
    }
    return found if sum(found.values()) >= FULL_ABUNDANCE else None


def _largest_rate(members: list[Member]) -> float:
    """Return a rate no member can reach.

    The rate comes from the upper bounds of the reactions of the growth of every member.
    """
    rates = []
    for member in members:
        network = member[0]
        rates.append(
            sum(
                abs(float(weight)) * _get_value(member, f'upper_{network._get_identifier(name)}')
                for name, weight in network.objective.items()
            ),
        )
    return max(rates)


def _steadycom(
    members: list[Member],
    medium: Mapping[str, float] | None,
) -> tuple[float, dict[str, float]]:
    """Return the largest common growth rate and the abundances.

    They come from a bisection on the rate.
    """
    problem = _community_problem(members, medium)
    low, high = 0.0, _largest_rate(members)
    best = _abundances_at(members, problem, low) or {member[1].label: 0.0 for member in members}
    while high - low > RATE_TOLERANCE * max(1.0, high):
        middle = 0.5 * (low + high)
        found = _abundances_at(members, problem, middle)
        if found is None:
            high = middle
        else:
            low, best = middle, found
    return low, best


def _tradeoff(
    members: list[Member],
    fraction: float,
    abundances: Sequence[float] | None,
) -> tuple[float, dict[str, float]]:
    """Return the community growth rate and the growth of every member.

    They are the ones of the cooperative trade-off.
    """
    cobra = load_extra('community_growth')
    micom = import_extra('micom', 'community_growth', 'metabolism')
    shares = abundances or [1.0 / len(members)] * len(members)
    labels = [cell.label for _, cell in members]
    with tempfile.TemporaryDirectory() as folder:
        files = []
        for network, cell in members:
            path = Path(folder) / f'{cell.label}.json'
            cobra.io.save_json_model(to_cobra(network, cell), str(path))
            files.append(str(path))
        taxonomy = pd.DataFrame({'id': labels, 'file': files, 'abundance': list(shares)})
        community = micom.Community(taxonomy, solver='hybrid', progress=False)
    solution = community.cooperative_tradeoff(fraction=fraction, fluxes=False, pfba=False)
    rates = solution.members.growth_rate
    return float(solution.growth_rate), {label: float(rates[label]) for label in labels}
