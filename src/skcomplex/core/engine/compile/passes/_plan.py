"""Plan the partitions of a system from the census of its laws.

The plan also uses their classes once lowered.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable, Mapping

from ....components.specs import CensusSpec
from ....components.system import System
from ....errors import IncompatibleLawsError
from ....specs import IssueSpec
from ....specs._owners_spec import _OwnersSpec
from ....specs._plan_spec import _PlanSpec
from ....specs._planned_partition_spec import _PlannedPartitionSpec
from ....types import FormKind
from .._base import BaseCompiler, BasePass
from .._compilation import Compilation
from .._forms import _FORMS
from ._form import FormPass


class PlanPass(BasePass):
    """Pass that plans the partitions of the system.

    It uses the registered forms and solvers.
    """

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
            Compilation._required(compilation.census, 'census'),
            Compilation._required(compilation.kinds, 'kinds'),
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
        solvers: list[type],
    ) -> IssueSpec:
        """Return the issue of a system whose own laws fit no solver or form.

        No solver runs them or no form takes them, by law kind or class.
        """
        fix = 'Move the components of each law kind into their own nested system, so they run as coupled partitions.'
        if not PlanPass._solved(set().union(*declared.values()), solvers):
            message = f'No solver runs the laws of this system together ({PlanPass._listed(declared)}).'
        else:
            message = f'No form takes the laws of this system together ({PlanPass._listed(lowered)}).'
        return IssueSpec('laws-mixed', path, message, fix, error=IncompatibleLawsError)

    @staticmethod
    def _fits(owners: _OwnersSpec, declared: dict[str, set[type]], lowered: dict[str, set[type]]) -> bool:
        """Check whether laws compile to one form and one solver of the owners runs them."""
        kinds = set().union(*declared.values())
        return PlanPass._fits_form(lowered.values()) and PlanPass._solved(kinds, owners.solvers)

    @staticmethod
    def _fits_form(owners: Iterable[set[type]]) -> bool:
        """Check whether the law classes of components compile to one form.

        Args:
            owners:
                The law classes of every component.

        Returns:
            fits:
                Whether a form takes the law classes.
        """
        return FormPass._form_kinds(owners) in _FORMS

    @staticmethod
    def _solved(kinds: set[type], solvers: list[type]) -> bool:
        """Check whether the solver of a single form passed runs every law kind of a set.

        It is also true when none is passed. Co-simulation is left out, since it runs partitions that each need a solver
        of their own.
        """
        single = [solver for solver in solvers if BaseCompiler._form_of(solver) != FormKind.COUPLED]
        return not single or any(not BaseCompiler._support_of(solver).missing('laws', kinds) for solver in single)

    @staticmethod
    def _planned(
        system: System,
        owners: _OwnersSpec,
        scope: str = '',
    ) -> tuple[list['_PlannedPartitionSpec'], list[IssueSpec]]:
        """Return the partitions of a system tree and the issues of unfit subsystems.

        A subsystem is unfit when its own laws fit no form.
        """
        name = scope.rstrip('/') or system.label
        declared, lowered = owners.under(scope, nested=True)
        if PlanPass._fits(owners, declared, lowered):
            return ([_PlannedPartitionSpec(name, scope, nested=True)] if declared or lowered else []), []
        own_declared, own_lowered = owners.under(scope, nested=False)
        if not system.systems or not PlanPass._fits(owners, own_declared, own_lowered):
            mixed = (own_declared, own_lowered) if system.systems else (declared, lowered)
            return [], [PlanPass._mixed_issue(name, *mixed, owners.solvers)]
        partitions = [_PlannedPartitionSpec(name, scope, nested=False)] if own_declared or own_lowered else []
        issues: list[IssueSpec] = []
        for child in system.systems:
            found, mixed_issues = PlanPass._planned(child, owners, f'{scope}{child.label}/')
            partitions += found
            issues += mixed_issues
        return partitions, issues

    @staticmethod
    def _plan_partitions(
        census: CensusSpec,
        kinds: Mapping[str, set[type]],
        solvers: Iterable[type] | None = None,
    ) -> '_PlanSpec':
        """Return the partitions of a system.

        They are the subsystems whose laws compile to one form that one solver runs.
        """
        declared: dict[str, set[type]] = {}
        for record in census.records:
            declared.setdefault(record.owner, set()).add(type(record.law))
        found = list(solvers or ())
        partitions, issues = PlanPass._planned(census.system, _OwnersSpec(declared, dict(kinds), found))
        return _PlanSpec(tuple(partitions), tuple(issues))
