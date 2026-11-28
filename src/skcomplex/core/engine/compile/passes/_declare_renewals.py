"""Add the hidden state that counts the renewals of every renewal law."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re

import numpy as np
from sage.all__sagemath_symbolics import NN

from ....components import EntrySpec
from ....components.laws import BaseLaw
from ..._utils import symbol
from ...problems._kernel_spec import KernelSpec
from .._base import BasePass
from .._compilation import Compilation, _BoundRenewal, _Declarations
from .._renewal_law import RenewalLaw
from ._census import CensusPass
from ._layout import LayoutPass


class DeclareRenewalsPass(BasePass):
    """A pass that adds the hidden state that counts the renewals of every renewal law.

    Every member renews on its own clock, and the pass adds a state that counts its renewals, which a trajectory
    records.

    Read more in the user guide, under Extend the library.

    See Also:
        RenewalLaw: Renews states after random waiting times.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.physics.soft_matter import ContinuousTimeRandomWalkSystem
        >>> compilation = PassCompiler().prepare(ContinuousTimeRandomWalkSystem()).compilation_
        >>> [entry.name for entry in compilation.layout.states]
        ['walkers.x', 'walkers.jumps~renewals']
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('delayed', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('delayed', 'renewals')

    def _run(self, inputs: object = None) -> None:
        """Declare the counts of the renewal laws."""
        compilation = self.compilation_
        declarations = Compilation._get_required(compilation.layout, 'layout')
        renewals: list[_BoundRenewal] = []
        for spec, laws in Compilation._get_required(compilation.delayed, 'delayed'):
            declarations, declared = self._get_declared_renewals(spec, laws, declarations)
            renewals += declared
        compilation.layout, compilation.renewals = declarations, renewals

    @staticmethod
    def _declare_renewal(
        spec: KernelSpec,
        law: RenewalLaw,
        declarations: _Declarations,
        label: str,
    ) -> tuple[_Declarations, _BoundRenewal]:
        """Return the declarations with the hidden state that counts renewals.

        The state counts the renewals of every member of a renewal law. Returns the declarations and the renewal law
        with its hidden state.
        """
        group = CensusPass._get_renewal_group(spec, law)
        rows = spec.rows
        sanitized = re.sub(r'[^0-9A-Za-z]', '_', label)
        shape, members = ((rows,), rows) if rows > 1 else ((), None)
        entry = EntrySpec(f'{group}.{sanitized}~renewals', declarations.ends[True], shape, NN, members)
        declarations = LayoutPass._copy_with_entry(declarations, entry, np.zeros(rows), is_state=True)
        return declarations, _BoundRenewal(
            spec,
            law,
            label,
            group,
            entry,
            symbol(f'renewals_{sanitized}', NN),
        )

    @staticmethod
    def _get_declared_renewals(
        spec: KernelSpec,
        laws: list[BaseLaw],
        declarations: _Declarations,
    ) -> tuple[_Declarations, list[_BoundRenewal]]:
        """Return the declarations with the hidden states that count renewals.

        They count the renewals of a spec, and its renewal laws are also returned.
        """
        renewals = [law for law in laws if isinstance(law, RenewalLaw)]
        declared = []
        for position, law in enumerate(renewals):
            label = law.label or (spec.owner if len(renewals) == 1 else f'{spec.owner}:{position}')
            declarations, renewal = DeclareRenewalsPass._declare_renewal(spec, law, declarations, label)
            declared.append(renewal)
        return declarations, declared
