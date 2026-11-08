"""Analyse communities of metabolic networks: SteadyCom on the solve of the core, and the trade-off of MICOM."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sage.all__sagemath_symbolics import Expression

from .....core.analysis.types import Source
from .....core.components.elements import DeclaredElement
from .....core.components.laws import LinearProgramLaw
from .....core.components.spaces import StaticTime
from .....core.components.system import BaseSystem, System
from .....core.engine.problems import ProblemSpec
from .....core.engine.runtime import simulate
from .....core.errors import AnalysisError
from .....core.specs import (
    DeclaredVariableSpec,
    LinearObjectiveSpec,
    ParameterSpec,
    RunStateSpec,
    StateSpec,
    VisualSpec,
)
from ...._extras import import_extra
from ._analyses import _check_choice, _networks_of
from ._cobra_io import load_extra, to_cobra
from ._metabolic_network import MetabolicNetworkElement

COMMUNITY_KINDS = ('steadycom', 'tradeoff')
RATE_TOLERANCE = 1e-7
FULL_ABUNDANCE = 1.0 - 1e-7
ELEMENT_LABEL = 'community'


class _SteadyComElement(DeclaredElement):
    """Element whose program sets the abundances and the scaled fluxes of a community."""

    law_classes = (LinearProgramLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
        law: Callable[[SimpleNamespace], LinearProgramLaw] | None = None,
    ) -> None:
        super().__init__(label, visual=visual, variables=variables)
        self.law = law

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[LinearProgramLaw]:
        """Return the program of the community for a namespace of symbols, or none without a function."""
        return [] if self.law is None else [self.law(namespace)]


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
    source: Source,
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
        >>> from skcomplex.core.components.spaces import StaticTime
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.common.metabolism import (
        ...     ReactionSpec,
        ...     community_growth,
        ...     metabolic_network_type,
        ... )
        >>> reactions = {
        ...     'EX_a': ReactionSpec({'a': 1.0}, upper=2.0),
        ...     'grow': ReactionSpec({'a': -1.0}),
        ... }
        >>> element_type = metabolic_network_type('MemberElement', reactions, {'grow': 1.0})
        >>> system = System(elements=[element_type('one'), element_type('two')], time=StaticTime())
        >>> result = community_growth(system)
        >>> round(result.growth, 6), round(sum(result.abundances.values()), 6)
        (2.0, 1.0)
    """
    _check_choice('community_growth', 'kind', kind, COMMUNITY_KINDS)
    if kind == 'tradeoff':
        load_extra(f'community_growth(kind={kind!r})')
    if isinstance(source, BaseSystem) and not isinstance(source, System):
        source = source.prepare().system_
    members = _networks_of(source) if isinstance(source, System) else []
    if not members:
        error_msg = 'community_growth reads a system that holds a metabolic network element.'
        raise AnalysisError(error_msg)
    if kind == 'steadycom':
        growth, found = _steadycom(members, medium)
        return CommunityGrowthResult(growth, abundances=found)
    growth, rates = _tradeoff(members, fraction, abundances)
    return CommunityGrowthResult(growth, members=rates)


def _member_relations(s: SimpleNamespace, position: int, member: MetabolicNetworkElement) -> list[Expression]:
    """Return the balances of a member, its bounds scaled by its abundance, and the growth at the common rate."""
    terms: dict[str, list[Expression]] = {}
    relations = []
    abundance = getattr(s, f'x_{position}')
    for column, (name, spec) in enumerate(member.reactions.items()):
        identifier = member._identifier(name)
        flux = getattr(s, f'v_{position}_{column}')
        relations.append(flux >= member.get_number(f'lower_{identifier}') * abundance)
        relations.append(flux <= member.get_number(f'upper_{identifier}') * abundance)
        for metabolite, coefficient in spec.coefficients.items():
            terms.setdefault(metabolite, []).append(float(coefficient) * flux)
    relations += [sum(items[1:], items[0]) == 0 for items in terms.values()]
    weights = member.objective or member.default_objective
    growth = [
        float(weight) * getattr(s, f'v_{position}_{list(member.reactions).index(name)}')
        for name, weight in weights.items()
    ]
    relations.append(sum(growth[1:], growth[0]) - s.rate * abundance == 0)
    return relations


def _community_law(
    members: list[MetabolicNetworkElement],
    medium: Mapping[str, float] | None,
    s: SimpleNamespace,
) -> LinearProgramLaw:
    """Return the program of the abundances of the members at the common rate of a namespace."""
    relations = [
        relation for position, member in enumerate(members) for relation in _member_relations(s, position, member)
    ]
    abundances = [getattr(s, f'x_{position}') for position in range(len(members))]
    total = sum(abundances[1:], abundances[0])
    relations += [item >= 0 for item in abundances]
    relations.append(total <= 1)
    for name, lower in (medium or {}).items():
        shared = [
            getattr(s, f'v_{position}_{list(member.reactions).index(name)}')
            for position, member in enumerate(members)
            if name in member.reactions
        ]
        relations.append(sum(shared[1:], shared[0]) >= float(lower))
    variables = [
        *abundances,
        *(
            getattr(s, f'v_{position}_{column}')
            for position, member in enumerate(members)
            for column in range(len(member.reactions))
        ),
    ]
    return LinearProgramLaw(variables, relations, [LinearObjectiveSpec(total)])


def _community_problem(members: list[MetabolicNetworkElement], medium: Mapping[str, float] | None) -> ProblemSpec:
    """Return the compiled problem of an element whose program sets the abundances and the scaled fluxes."""
    names = [f'x_{position}' for position in range(len(members))]
    names += [
        f'v_{position}_{column}' for position, member in enumerate(members) for column in range(len(member.reactions))
    ]
    variables = {name: DeclaredVariableSpec(0.0, StateSpec(algebraic=True)) for name in names}
    variables['rate'] = DeclaredVariableSpec(0.0, ParameterSpec())
    law = partial(_community_law, members, medium)
    element = _SteadyComElement(ELEMENT_LABEL, variables=variables, law=law)
    problem = simulate(System(elements=[element], time=StaticTime()), until=0).problem
    if not isinstance(problem, ProblemSpec):
        error_msg = f'The community compiles to a {type(problem).__name__}, not a linear program.'
        raise AnalysisError(error_msg)
    return problem


def _abundances_at(
    members: list[MetabolicNetworkElement],
    problem: ProblemSpec,
    rate: float,
) -> dict[str, float] | None:
    """Return the abundances at a common rate, or `None` when they cannot reach a sum of 1."""
    parameters = problem.initial_parameters.copy()
    parameters[problem.parameters.names.index(f'{ELEMENT_LABEL}.rate')] = rate
    solved = problem.programs[0].run(RunStateSpec(rate, problem.initial_state, parameters)).values_
    found = {
        member.label: float(np.real(np.ravel(problem.states.read(solved, f'{ELEMENT_LABEL}.x_{position}'))[0]))
        for position, member in enumerate(members)
    }
    return found if sum(found.values()) >= FULL_ABUNDANCE else None


def _largest_rate(members: list[MetabolicNetworkElement]) -> float:
    """Return a rate no member can reach, from the upper bounds of the reactions of its growth."""
    rates = []
    for member in members:
        weights = member.objective or member.default_objective
        rates.append(
            sum(
                abs(float(weight)) * member.get_number(f'upper_{member._identifier(name)}')
                for name, weight in weights.items()
            ),
        )
    return max(rates)


def _steadycom(
    members: list[MetabolicNetworkElement],
    medium: Mapping[str, float] | None,
) -> tuple[float, dict[str, float]]:
    """Return the largest common growth rate and the abundances, by a bisection on the rate."""
    problem = _community_problem(members, medium)
    low, high = 0.0, _largest_rate(members)
    best = _abundances_at(members, problem, low) or {member.label: 0.0 for member in members}
    while high - low > RATE_TOLERANCE * max(1.0, high):
        middle = 0.5 * (low + high)
        found = _abundances_at(members, problem, middle)
        if found is None:
            high = middle
        else:
            low, best = middle, found
    return low, best


def _tradeoff(
    members: list[MetabolicNetworkElement],
    fraction: float,
    abundances: Sequence[float] | None,
) -> tuple[float, dict[str, float]]:
    """Return the community growth rate and the growth of every member of the cooperative trade-off."""
    cobra = load_extra('community_growth')
    micom = import_extra('micom', 'community_growth', 'metabolism')
    shares = abundances or [1.0 / len(members)] * len(members)
    with tempfile.TemporaryDirectory() as folder:
        files = []
        for member in members:
            path = Path(folder) / f'{member.label}.json'
            cobra.io.save_json_model(to_cobra(member), str(path))
            files.append(str(path))
        taxonomy = pd.DataFrame(
            {'id': [member.label for member in members], 'file': files, 'abundance': list(shares)},
        )
        community = micom.Community(taxonomy, solver='hybrid', progress=False)
    solution = community.cooperative_tradeoff(fraction=fraction, fluxes=False, pfba=False)
    rates = solution.members.growth_rate
    return float(solution.growth_rate), {member.label: float(rates[member.label]) for member in members}
