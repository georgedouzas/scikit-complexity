"""Plan the partitions of a system from the census of its laws and their classes once lowered."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable, Mapping
from typing import Any, cast

from ....core.errors import IncompatibleLawsError
from ....core.system import System
from .._base import BasePass
from .._census import Census
from .._issue import Issue
from .._owners import _Owners
from .._plan import _Plan
from .._planned_partition import _PlannedPartition


class PlanPass(BasePass):
    """Pass that plans the partitions of the system with the registered forms and solvers."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('census', 'kinds')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('plan',)

    def _run(self, inputs: object = None) -> None:
        """Plan the partitions of the system."""
        compilation = self.compilation_
        compilation.plan = self._plan_partitions(
            cast('Census', compilation.census),
            cast('dict[str, set[type]]', compilation.kinds),
            [] if compilation.solver is None else [type(compilation.solver)],
        )

    @staticmethod
    def _listed(owners: Mapping[str, Iterable[object]]) -> str:
        """Return the components of every law kind or class as one list."""
        classes: dict[str, list[str]] = {}
        for owner, kinds in owners.items():
            for kind in kinds:
                classes.setdefault(getattr(kind, '__name__', f'{kind}'), []).append(owner)
        return '; '.join(f'{kind}: {", ".join(sorted(set(paths)))}' for kind, paths in sorted(classes.items()))

    @staticmethod
    def _mixed_issue(
        path: str,
        declared: Mapping[str, set[type]],
        lowered: Mapping[str, set[type]],
        solvers: list[Any],
    ) -> Issue:
        """Return the issue of a system whose own laws no solver runs or no form takes, by law kind or class."""
        fix = 'Move the components of each law kind into their own nested system, so they run as coupled partitions.'
        if not _Owners._solved(set().union(*declared.values()), solvers):
            message = f'No solver runs the laws of this system together ({PlanPass._listed(declared)}).'
        else:
            message = f'No form takes the laws of this system together ({PlanPass._listed(lowered)}).'
        return Issue('laws-mixed', path, message, fix, error=IncompatibleLawsError)

    @staticmethod
    def _planned(system: System, owners: _Owners, scope: str = '') -> tuple[list['_PlannedPartition'], list[Issue]]:
        """Return the partitions of a system tree and the issues of the subsystems whose own laws fit no form."""
        name = scope.rstrip('/') or system.label
        declared, lowered = owners.under(scope, nested=True)
        if owners.fit(declared, lowered):
            return ([_PlannedPartition(name, scope, nested=True)] if declared or lowered else []), []
        own_declared, own_lowered = owners.under(scope, nested=False)
        if not system.systems or not owners.fit(own_declared, own_lowered):
            mixed = (own_declared, own_lowered) if system.systems else (declared, lowered)
            return [], [PlanPass._mixed_issue(name, *mixed, owners.solvers)]
        partitions = [_PlannedPartition(name, scope, nested=False)] if own_declared or own_lowered else []
        issues: list[Issue] = []
        for child in system.systems:
            found, mixed_issues = PlanPass._planned(child, owners, f'{scope}{child.label}/')
            partitions += found
            issues += mixed_issues
        return partitions, issues

    @staticmethod
    def _plan_partitions(
        census: Census,
        kinds: Mapping[str, set[type]],
        solvers: Iterable[Any] | None = None,
    ) -> '_Plan':
        """Return the partitions of a system, the subsystems whose laws compile to one form that one solver runs."""
        declared: dict[str, set[type]] = {}
        for record in census.records:
            declared.setdefault(record.owner, set()).add(type(record.law))
        found = list(solvers or ())
        partitions, issues = PlanPass._planned(census.system, _Owners(declared, dict(kinds), found))
        return _Plan(tuple(partitions), tuple(issues))


