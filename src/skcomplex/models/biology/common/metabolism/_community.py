"""Analyse communities of metabolic networks: SteadyCom on the solve of the core, and the trade-off of MICOM."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pandas as pd
from sage.all__sagemath_symbolics import Expression

from .....analysis.types import Source
from .....core.elements import Element
from .....core.laws import LinearProgramLaw
from .....core.spaces import StaticTime
from .....core.system import System
from .....engine.compile import compile_system
from .....engine.problems import LinearProgramKernel, ProblemSpec
from .....errors import AnalysisError
from .....specs import LinearObjectiveSpec, ParameterSpec, RunStateSpec, StateSpec
from ...._extras import import_extra
from ._analyses import BaseNetworkAnalysis
from ._cobra_io import load_extra, to_cobra
from ._metabolic_network import MetabolicNetworkElement

COMMUNITY_KINDS = ('steadycom', 'tradeoff')
RATE_TOLERANCE = 1e-7
FULL_ABUNDANCE = 1.0 - 1e-7
ELEMENT_LABEL = 'community'


class CommunityAnalysis(BaseNetworkAnalysis):
    """Analysis of a community of metabolic networks that share a medium.

    - `'steadycom'`: the largest common growth rate of the members, after Chan et al. (2017), by a bisection on the rate
      with the solve of the core, without the extra. The fluxes of a member scale with its abundance, the abundances
      sum to at most 1, and a shared exchange of the medium stays above the bound that `medium` gives.
    - `'tradeoff'`: the cooperative trade-off of MICOM, which needs the extra `metabolism`.

    Args:
        kind:
            The analysis, one of the kinds above.

        fraction:
            The fraction of the community growth rate that the trade-off keeps.

        abundances:
            The abundance of every member for `'tradeoff'`, or `None` for equal ones.

        medium:
            The lower bound of every shared exchange of the whole community for `'steadycom'`, by reaction, or `None`
            for no shared bound.

    Attributes:
        growth_:
            The common growth rate, or the community growth rate of the trade-off.

        abundances_:
            The abundance of every member, by its label, for `'steadycom'`.

        members_:
            The growth of every member, by its label, for `'tradeoff'`.

    Raises:
        AnalysisError:
            The kind is unknown, raised by `check`.

        MissingExtraError:
            The kind `'tradeoff'` needs the extra, raised by `check`.

    Examples:
        >>> from skcomplex.core.spaces import StaticTime
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.common.metabolism import (
        ...     CommunityAnalysis,
        ...     ReactionSpec,
        ...     metabolic_network_type,
        ... )
        >>> reactions = {
        ...     'EX_a': ReactionSpec({'a': 1.0}, upper=2.0),
        ...     'grow': ReactionSpec({'a': -1.0}),
        ... }
        >>> element_type = metabolic_network_type('MemberElement', reactions, {'grow': 1.0})
        >>> system = System(elements=[element_type('one'), element_type('two')], time=StaticTime())
        >>> analysis = CommunityAnalysis().check().prepare(system).run()
        >>> round(analysis.growth_, 6), round(sum(analysis.abundances_.values()), 6)
        (2.0, 1.0)
    """

    def __init__(
        self,
        kind: str = 'steadycom',
        *,
        fraction: float = 1.0,
        abundances: Sequence[float] | None = None,
        medium: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.fraction = fraction
        self.abundances = abundances
        self.medium = medium

    def _check(self) -> None:
        """Raise an error for an unknown kind, or a missing extra for the trade-off.

        Raises:
            AnalysisError:
                The kind is unknown.

            MissingExtraError:
                The trade-off needs the extra.
        """
        self._check_choice('kind', self.kind, COMMUNITY_KINDS)
        if self.kind == 'tradeoff':
            load_extra(f'{type(self).__name__}(kind={self.kind!r})')

    def _prepare(self, context: Source) -> None:
        """Read the members of a system.

        Args:
            context:
                The system.
        """
        super()._prepare(context)
        if isinstance(context, System):
            self._members = self._networks_of(context)

    def _member_relations(self, s: SimpleNamespace, position: int, member: MetabolicNetworkElement) -> list[Expression]:
        """Return the balances of a member, its bounds scaled by its abundance, and the growth at the common rate."""
        terms: dict[str, list[Expression]] = {}
        relations = []
        abundance = getattr(s, f'x_{position}')
        for column, (name, spec) in enumerate(member.reactions.items()):
            identifier = member._identifier(name)
            flux = getattr(s, f'v_{position}_{column}')
            relations.append(flux >= float(getattr(member, f'lower_{identifier}')) * abundance)
            relations.append(flux <= float(getattr(member, f'upper_{identifier}')) * abundance)
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

    def _community_law(self, s: SimpleNamespace) -> LinearProgramLaw:
        """Return the program of the abundances of the members at the common rate of a namespace."""
        members = self._members
        relations = [
            relation
            for position, member in enumerate(members)
            for relation in self._member_relations(s, position, member)
        ]
        abundances = [getattr(s, f'x_{position}') for position in range(len(members))]
        total = sum(abundances[1:], abundances[0])
        relations += [item >= 0 for item in abundances]
        relations.append(total <= 1)
        for name, lower in (self.medium or {}).items():
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

    def _community_problem(self) -> ProblemSpec:
        """Return the compiled problem of an element whose program sets the abundances and the scaled fluxes."""
        flux = Annotated[float, StateSpec(algebraic=True)]
        names = [f'x_{position}' for position in range(len(self._members))]
        names += [
            f'v_{position}_{column}'
            for position, member in enumerate(self._members)
            for column in range(len(member.reactions))
        ]
        declared = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=0.0, annotation=flux) for name in names
        ]
        rate = inspect.Parameter(
            'rate',
            inspect.Parameter.KEYWORD_ONLY,
            default=0.0,
            annotation=Annotated[float, ParameterSpec()],
        )

        def initialize(self: Element, label: str, **values: float) -> None:
            Element.__init__(self, label)
            for name in [*names, 'rate']:
                setattr(self, name, values.get(name, 0.0))

        first = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in ('self', 'label')]
        vars(initialize)['__signature__'] = inspect.Signature([*first, *declared, rate])
        law = self._community_law
        element_type = type(
            'SteadyComElement',
            (Element,),
            {'__init__': initialize, '_get_laws': lambda _, namespace: [law(namespace)]},
        )
        problem = compile_system(System(elements=[element_type(ELEMENT_LABEL)], time=StaticTime()))
        if not isinstance(problem, ProblemSpec):
            error_msg = f'The community compiles to a {type(problem).__name__}, not a linear program.'
            raise AnalysisError(error_msg)
        return problem

    def _abundances_at(self, problem: ProblemSpec, kernel: LinearProgramKernel, rate: float) -> dict[str, float] | None:
        """Return the abundances at a common rate, or `None` when they cannot reach a sum of 1."""
        parameters = problem.initial_parameters.copy()
        parameters[problem.parameters.names.index(f'{ELEMENT_LABEL}.rate')] = rate
        solved = kernel.run(RunStateSpec(rate, problem.initial_state, parameters)).values_
        found = {
            member.label: float(np.real(np.ravel(problem.states.read(solved, f'{ELEMENT_LABEL}.x_{position}'))[0]))
            for position, member in enumerate(self._members)
        }
        return found if sum(found.values()) >= FULL_ABUNDANCE else None

    def _largest_rate(self) -> float:
        """Return a rate no member can reach, from the upper bounds of the reactions of its growth."""
        rates = []
        for member in self._members:
            weights = member.objective or member.default_objective
            rates.append(
                sum(
                    abs(float(weight)) * float(getattr(member, f'upper_{member._identifier(name)}'))
                    for name, weight in weights.items()
                ),
            )
        return max(rates)

    def _run_steadycom(self) -> None:
        """Set the largest common growth rate and the abundances, by a bisection on the rate."""
        problem = self._community_problem()
        kernel = problem.programs[0]
        low, high = 0.0, self._largest_rate()
        best = self._abundances_at(problem, kernel, low) or {member.label: 0.0 for member in self._members}
        while high - low > RATE_TOLERANCE * max(1.0, high):
            middle = 0.5 * (low + high)
            found = self._abundances_at(problem, kernel, middle)
            if found is None:
                high = middle
            else:
                low, best = middle, found
        self.growth_ = low
        self.abundances_ = best

    def _run_tradeoff(self) -> None:
        """Set the community growth rate and the growth of every member of the cooperative trade-off."""
        cobra = load_extra(type(self).__name__)
        micom = import_extra('micom', type(self).__name__, 'metabolism')
        members = self._members
        shares = self.abundances or [1.0 / len(members)] * len(members)
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
        solution = community.cooperative_tradeoff(fraction=self.fraction, fluxes=False, pfba=False)
        self.growth_ = float(solution.growth_rate)
        rates = solution.members.growth_rate
        self.members_ = {member.label: float(rates[member.label]) for member in members}

    def _run(self, inputs: object = None) -> None:
        """Run the kind of the analysis.

        Args:
            inputs:
                Not read.
        """
        if self.kind == 'steadycom':
            self._run_steadycom()
        else:
            self._run_tradeoff()
