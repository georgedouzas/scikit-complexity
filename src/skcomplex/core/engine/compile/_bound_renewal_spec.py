"""Define the spec of a renewal law with the group it renews."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec
from ...components import EntrySpec
from ..problems._kernel_spec import KernelSpec
from ._renewal_law import RenewalLaw


@dataclass(frozen=True)
class BoundRenewalSpec(BaseSpec):
    """A spec of a renewal law with the group it renews and the state that counts it.

    A renewal law restarts the members of a group after waiting times, such as a walker that rests for a random time and
    then jumps. The compiler binds every renewal law to the group it renews and adds a hidden state that counts the
    renewals of every member, which this spec holds. It describes the jumps of walkers, the failures and repairs of
    machines, and the firing of neurons.

    Read more in the user guide, under Extend the library.

    Args:
        spec:
            The kernel spec of the component.

        law:
            The renewal law.

        label:
            The label of the renewal.

        group:
            The path of the element or group the renewal resets.

        entry:
            The hidden state that counts the renewals of every member.

        symbol:
            The symbol of the count.

    See Also:
        RenewalLaw: Renews states after random waiting times.
        DeclareRenewalsPass: Adds the state that counts the renewals.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.physics.soft_matter import ContinuousTimeRandomWalkSystem
        >>> compilation = PassCompiler().prepare(ContinuousTimeRandomWalkSystem()).compilation_
        >>> [(renewal.label, renewal.group) for renewal in compilation.renewals]
        [('jumps', 'walkers')]
    """

    spec: KernelSpec
    law: RenewalLaw
    label: str
    group: str
    entry: EntrySpec
    symbol: Expression
