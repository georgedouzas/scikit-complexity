"""Pass that declares the hidden state that counts the renewals of every renewal law, once delays are lowered."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import NN

from ....core.laws import BaseLaw
from ...problems import symbol
from .._bound_renewal import BoundRenewal
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._kernel_spec import KernelSpec
from .._pipeline import BasePass
from .._renewal_law import RenewalLaw
from ._census import CensusPass


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
            compilation.renewals += self._declared_renewals(spec, laws, declarations)

    @staticmethod
    def _declare_renewal(spec: KernelSpec, law: RenewalLaw, declarations: Declarations, label: str) -> BoundRenewal:
        """Return a renewal law with the hidden state that counts the renewals of every member, after declaring it."""
        group = CensusPass._renewal_group(spec, law)
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

    @staticmethod
    def _declared_renewals(spec: KernelSpec, laws: list[BaseLaw], declarations: Declarations) -> list[BoundRenewal]:
        """Return the renewal laws of a spec, each with the hidden state counting its renewals, after declaring it."""
        renewals = [law for law in laws if isinstance(law, RenewalLaw)]
        declared = []
        for law in renewals:
            position = renewals.index(law)
            label = law.label or (spec.owner if len(renewals) == 1 else f'{spec.owner}:{position}')
            declared.append(DeclareRenewalsPass._declare_renewal(spec, law, declarations, label))
        return declared
