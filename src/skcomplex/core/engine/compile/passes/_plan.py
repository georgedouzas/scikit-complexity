"""Split a system into the partitions that one solver each can run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ....components.laws import BaseLaw
from ....enums import FormKind
from ....errors import IncompatibleLawsError, UnregisteredLawError
from ...problems._composite_system import _CompositeSystem
from .._base import BaseCompiler, BasePass
from .._compilation import Compilation, _Census, _Issue, _PlannedPartition
from .._forms import FORMS
from .._laws import FORMLESS_LAWS
from ._form import FormPass
from ._kernels import KernelsPass


@dataclass(frozen=True)
class _Owners:
    """Law kinds and classes of the owners of a system, with the solvers to run them.

    The kinds are as declared and the classes as lowered.
    """

    declared: dict[str, set[type[BaseLaw]]]
    lowered: dict[str, set[type[BaseLaw]]]
    solvers: list[type]

    def get_under(
        self,
        scope: str,
        *,
        nested: bool,
    ) -> tuple[dict[str, set[type[BaseLaw]]], dict[str, set[type[BaseLaw]]]]:
        """Return the declared kinds and lowered classes of the owners of a scope.

        With its own components alone, it returns those of the scope's own components.

        Args:
            scope:
                The path prefix of the scope.

            nested:
                Whether the owners of nested systems count.

        Returns:
            declared:
                The declared kinds of the owners of the scope.

            lowered:
                The lowered classes of the owners of the scope.
        """

        def holds(owner: str) -> bool:
            return owner.startswith(scope) and (nested or '/' not in owner[len(scope) :])

        return (
            {owner: kinds for owner, kinds in self.declared.items() if holds(owner)},
            {owner: kinds for owner, kinds in self.lowered.items() if holds(owner)},
        )


class PlanPass(BasePass):
    """A pass that splits a system into the partitions that one solver each can run.

    Most systems form one partition that one solver runs as a whole. A system whose components need different solvers,
    such as a field of rates beside agents that follow rules, is split into partitions that advance side by side, and a
    system that no solver can run is reported.

    Read more in the user guide, under Extend the library.

    See Also:
        CensusPass: Lists the laws the plan is made from.
        FormPass: Finds the kind of problem of a partition.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> compilation = PassCompiler().prepare(LotkaVolterraSystem()).compilation_
        >>> [partition.name for partition in compilation.plan[0]]
        ['system']
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
            Compilation._get_required(compilation.census, 'census'),
            Compilation._get_required(compilation.kinds, 'kinds'),
            [] if compilation.solver is None else [type(compilation.solver)],
        )

    @staticmethod
    def _get_listed(owners: Mapping[str, Iterable[type[BaseLaw]]]) -> str:
        """Return the components of every law kind or class as one list."""
        classes: dict[str, list[str]] = {}
        for owner, kinds in owners.items():
            for kind in kinds:
                classes.setdefault(kind._get_name(), []).append(owner)
        return '; '.join(f'{kind}: {", ".join(sorted(set(paths)))}' for kind, paths in sorted(classes.items()))

    @staticmethod
    def _get_mixed_issue(
        path: str,
        declared: Mapping[str, set[type[BaseLaw]]],
        lowered: Mapping[str, set[type[BaseLaw]]],
        solvers: list[type],
    ) -> _Issue:
        """Return the issue of a system whose own laws fit no solver or form.

        No solver runs them or no form takes them, by law kind or class.
        """
        fix = 'Move the components of each law kind into their own nested system, so they run as coupled partitions.'
        if not PlanPass._get_solved(set().union(*declared.values()), solvers):
            message = f'No solver runs the laws of this system together ({PlanPass._get_listed(declared)}).'
        else:
            message = f'No form takes the laws of this system together ({PlanPass._get_listed(lowered)}).'
        return _Issue('laws-mixed', path, message, fix, error=IncompatibleLawsError)

    @staticmethod
    def _get_unregistered_issues(kinds: Mapping[str, set[type[BaseLaw]]]) -> list[_Issue]:
        """Return the issues of the law classes that no form takes and that need one.

        Args:
            kinds:
                The law classes of every component, by path.

        Returns:
            issues:
                One issue for every component and law class that has no form.
        """
        registered = set().union(*FORMS)
        fix = 'Write the law with a class the core provides, such as `RateLaw` or `UpdateLaw`.'
        return [
            _Issue(
                'law-unregistered',
                owner,
                f'The law class {kind._get_name()} has no form the engine knows.',
                fix,
                error=UnregisteredLawError,
            )
            for owner, classes in sorted(kinds.items())
            for kind in sorted(classes, key=lambda kind: kind._get_name())
            if kind not in registered and not issubclass(kind, FORMLESS_LAWS)
        ]

    @staticmethod
    def _fits(
        owners: _Owners,
        declared: dict[str, set[type[BaseLaw]]],
        lowered: dict[str, set[type[BaseLaw]]],
    ) -> bool:
        """Check whether laws compile to one form and one solver of the owners runs them."""
        kinds = set().union(*declared.values())
        return PlanPass._fits_form(lowered.values()) and PlanPass._get_solved(kinds, owners.solvers)

    @staticmethod
    def _fits_form(owners: Iterable[set[type[BaseLaw]]]) -> bool:
        """Check whether the law classes of components compile to one form.

        Args:
            owners:
                The law classes of every component.

        Returns:
            fits:
                Whether a form takes the law classes.
        """
        return FormPass._get_form_kinds(owners) in FORMS

    @staticmethod
    def _get_solved(kinds: set[type[BaseLaw]], solvers: list[type]) -> bool:
        """Check whether the solver of a single form passed runs every law kind of a set.

        It is also true when none is passed. Co-simulation is left out, since it runs partitions that each need a solver
        of their own.
        """
        single = [solver for solver in solvers if BaseCompiler._get_form_of(solver) != FormKind.COUPLED]
        return not single or any(not BaseCompiler._support_of(solver).get_missing('laws', kinds) for solver in single)

    @staticmethod
    def _get_planned(
        system: _CompositeSystem,
        owners: _Owners,
        scope: str = '',
    ) -> tuple[list[_PlannedPartition], list[_Issue]]:
        """Return the partitions of a system tree and the issues of unfit subsystems.

        A subsystem is unfit when its own laws fit no form.
        """
        name = scope.rstrip('/') or system.label
        declared, lowered = owners.get_under(scope, nested=True)
        if PlanPass._fits(owners, declared, lowered):
            return ([_PlannedPartition(name, scope, nested=True)] if declared or lowered else []), []
        own_declared, own_lowered = owners.get_under(scope, nested=False)
        if not system.systems or not PlanPass._fits(owners, own_declared, own_lowered):
            mixed = (own_declared, own_lowered) if system.systems else (declared, lowered)
            return [], [PlanPass._get_mixed_issue(name, *mixed, owners.solvers)]
        partitions = [_PlannedPartition(name, scope, nested=False)] if own_declared or own_lowered else []
        issues: list[_Issue] = []
        for child in system.systems:
            found, mixed_issues = PlanPass._get_planned(child, owners, f'{scope}{child.label}/')
            partitions += found
            issues += mixed_issues
        return partitions, issues

    @staticmethod
    def _plan_partitions(
        census: _Census,
        kinds: Mapping[str, set[type[BaseLaw]]],
        solvers: Iterable[type] | None = None,
    ) -> tuple[tuple[_PlannedPartition, ...], tuple[_Issue, ...]]:
        """Return the partitions of a system.

        They are the subsystems whose laws compile to one form that one solver runs, with the issues of the law mixes
        that no form or solver takes.
        """
        declared: dict[str, set[type[BaseLaw]]] = {}
        for record in census.records:
            declared.setdefault(record.owner, set()).add(KernelsPass._get_kind_of(record.law))
        unregistered = PlanPass._get_unregistered_issues(kinds)
        if unregistered:
            return (), tuple(unregistered)
        found = list(solvers or ())
        partitions, issues = PlanPass._get_planned(census.system, _Owners(declared, dict(kinds), found))
        return tuple(partitions), tuple(issues)
