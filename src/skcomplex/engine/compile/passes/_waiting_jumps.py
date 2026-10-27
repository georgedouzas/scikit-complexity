"""Lower the jumps on clocks of waiting times into renewals, which reset states after drawn waiting times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from dataclasses import replace
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import NN, Expression

from ....core.laws import BaseLaw, JumpLaw
from ...problems import Problem, symbol
from .._census import renewal_group
from .._compilation import BlockLaws, Compilation, Laws
from .._declarations import Declarations
from .._pipeline import BasePass
from .._renewal_law import RenewalLaw
from .._renewals import BoundRenewal, CompiledRenewal, compile_renewal
from .._specs import KernelSpec


def _declare_renewal(spec: KernelSpec, law: RenewalLaw, declarations: Declarations, label: str) -> BoundRenewal:
    """Return a renewal law with the hidden state that counts the renewals of every member, after declaring it."""
    group = renewal_group(spec, law)
    rows = spec.rows
    sanitized = re.sub(r'[^0-9A-Za-z]', '_', label)
    entry = declarations.declare_state(
        f'{group}.{sanitized}~renewals',
        np.zeros(rows),
        NN,
        (rows,) if rows > 1 else (),
        rows if rows > 1 else None,
    )
    return BoundRenewal(
        spec,
        law,
        label,
        group,
        entry,
        symbol(f'renewals_{sanitized}', NN),
    )


def _declared_renewals(spec: KernelSpec, laws: list[BaseLaw], declarations: Declarations) -> list[BoundRenewal]:
    """Return the renewal laws of a spec, each with the hidden state that counts its renewals, after declaring it."""
    renewals = [law for law in laws if isinstance(law, RenewalLaw)]
    declared = []
    for law in renewals:
        position = renewals.index(law)
        label = law.label or (spec.owner if len(renewals) == 1 else f'{spec.owner}:{position}')
        declared.append(_declare_renewal(spec, law, declarations, label))
    return declared


def _compile_renewals(
    renewals: list[BoundRenewal],
    time: Expression,
    owners: frozenset[str] | None,
) -> tuple[CompiledRenewal, ...]:
    """Return the renewal laws of the components a problem holds, compiled for their members."""
    return tuple(
        compile_renewal(
            item.spec.owner,
            item.law,
            item.spec.references,
            time=time,
            rows=item.spec.rows,
            label=item.label,
            group=item.group,
            counts=item.entry.start + np.arange(item.entry.size, dtype=np.intp),
        )
        for item in renewals
        if owners is None or item.group in owners
    )


class UnpackRenewalsPass(BasePass):
    """Pass that replaces every jump on a clock of waiting times by the renewal the steppers run for it."""

    reads = ('collected',)
    writes = ('collected',)

    def _run(self, compilation: Compilation) -> None:
        """Replace the jumps on clocks of waiting times by renewals.

        Args:
            compilation:
                The compilation state.
        """
        compilation.collected = [
            (
                spec,
                [
                    (
                        RenewalLaw(law.waiting, law.reset, label=law.label).check().prepare(law.owner_)
                        if isinstance(law, JumpLaw) and law.waiting is not None
                        else law
                    )
                    for law in laws
                ],
                block,
            )
            for spec, laws, block in cast('BlockLaws', compilation.collected)
        ]


class DeclareRenewalsPass(BasePass):
    """Pass that declares the hidden state that counts the renewals of every renewal law, once delays are lowered."""

    reads = ('delayed', 'layout')
    writes = ('delayed', 'renewals')

    def _run(self, compilation: Compilation) -> None:
        """Declare the counts of the renewal laws.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        for spec, laws in cast('Laws', compilation.delayed):
            compilation.renewals += _declared_renewals(spec, laws, declarations)


class SetRenewalsApartPass(BasePass):
    """Pass that takes the renewal laws out of the laws lowered into kernels, since they compile on their own."""

    reads = ('reduced',)
    writes = ('reduced',)

    def _run(self, compilation: Compilation) -> None:
        """Take the renewal laws out of the reduced laws.

        Args:
            compilation:
                The compilation state.
        """
        compilation.reduced = [
            (spec, [law for law in laws if not isinstance(law, RenewalLaw)])
            for spec, laws in cast('Laws', compilation.reduced)
        ]


class RenewalProblemPass(BasePass):
    """Pass that adds the compiled renewal laws and the symbols of their counts to the problem."""

    reads = ('problem', 'renewals', 'layout')
    writes = ('problem',)

    def _run(self, compilation: Compilation) -> None:
        """Add the renewal laws to the problem.

        Args:
            compilation:
                The compilation state.
        """
        problem = cast('Problem', compilation.problem)
        time = cast('Declarations', compilation.layout).time
        symbols = {item.entry.name: item.symbol for item in compilation.renewals}
        compilation.problem = replace(
            problem,
            renewals=_compile_renewals(compilation.renewals, time, compilation.owners),
            symbols={**problem.symbols, **symbols},
        )
