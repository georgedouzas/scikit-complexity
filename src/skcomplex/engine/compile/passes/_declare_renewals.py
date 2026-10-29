"""Pass that declares the hidden state that counts the renewals of every renewal law, once delays are lowered."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import NN

from ....core.laws import BaseLaw
from ....specs import EntrySpec
from ..._utils import symbol
from ...problems import DeclarationsSpec, KernelSpec
from .._base import BasePass
from .._bound_renewal_spec import BoundRenewalSpec
from .._compilation import Laws
from .._renewal_law import RenewalLaw
from ._census import CensusPass
from ._layout import LayoutPass


class DeclareRenewalsPass(BasePass):
    """Pass that declares the hidden state that counts the renewals of every renewal law, once delays are lowered."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('delayed', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('delayed', 'renewals')

    def _run(self, inputs: object = None) -> None:
        """Declare the counts of the renewal laws."""
        compilation = self.compilation_
        declarations = cast('DeclarationsSpec', compilation.layout)
        renewals: list[BoundRenewalSpec] = []
        for spec, laws in cast('Laws', compilation.delayed):
            declarations, declared = self._declared_renewals(spec, laws, declarations)
            renewals += declared
        compilation.layout, compilation.renewals = declarations, renewals

    @staticmethod
    def _declare_renewal(
        spec: KernelSpec,
        law: RenewalLaw,
        declarations: DeclarationsSpec,
        label: str,
    ) -> tuple[DeclarationsSpec, BoundRenewalSpec]:
        """Return the declarations with the hidden state that counts the renewals of every member of a renewal law.

        Returns the declarations and the renewal law with its hidden state.
        """
        group = CensusPass._renewal_group(spec, law)
        rows = spec.rows
        sanitized = re.sub(r'[^0-9A-Za-z]', '_', label)
        shape, members = ((rows,), rows) if rows > 1 else ((), None)
        entry = EntrySpec(f'{group}.{sanitized}~renewals', declarations.ends[True], shape, NN, members)
        declarations = LayoutPass._with_entry(declarations, entry, np.zeros(rows), is_state=True)
        return declarations, BoundRenewalSpec(
            spec,
            law,
            label,
            group,
            entry,
            symbol(f'renewals_{sanitized}', NN),
        )

    @staticmethod
    def _declared_renewals(
        spec: KernelSpec,
        laws: list[BaseLaw],
        declarations: DeclarationsSpec,
    ) -> tuple[DeclarationsSpec, list[BoundRenewalSpec]]:
        """Return the declarations with the hidden states counting the renewals of a spec, and its renewal laws."""
        renewals = [law for law in laws if isinstance(law, RenewalLaw)]
        declared = []
        for law in renewals:
            position = renewals.index(law)
            label = law.label or (spec.owner if len(renewals) == 1 else f'{spec.owner}:{position}')
            declarations, renewal = DeclareRenewalsPass._declare_renewal(spec, law, declarations, label)
            declared.append(renewal)
        return declarations, declared
