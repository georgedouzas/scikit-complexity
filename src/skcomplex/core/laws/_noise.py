"""Define noise laws, amplitudes of the Wiener processes that drive states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...errors import ValidationError
from ...specs import RequirementSpec
from ...types import Process, Target
from ..spaces import ContinuousTime
from ._base import BaseLaw

NOISE_CONVENTIONS = {'ito': 0.0, 'stratonovich': 0.5, 'anti-ito': 1.0}


class NoiseLaw(BaseLaw):
    """Amplitude of a Wiener process that drives the highest time derivative of a state.

    Without a process, every component of the target and every member of a group gets its own independent Wiener
    process. Laws of one component, or of one member, that name the same local process share its increment. Laws that
    name a process of a `WienerProcessSpec` through a role share it across components and members.

    A conserved noise on a state of a field on a lattice adds `div(value eta)` instead, the finite-difference divergence
    of a random flux with one Wiener process per link between neighbouring sites, so the total of the field changes
    only by its rate laws.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression, a `Call`, or a callable `f(t, values)`. A vector target gets one Wiener process per
            component.

        order:
            The order of the derivative the noise drives, checked against the state's order, or `None` to use it.

        convention:
            The stochastic calculus of the law, `'ito'`, `'stratonovich'`, or `'anti-ito'`. The compiler adds the
            drift correction of the convention, so every solver integrates the Itô form.

        process:
            `None` for independent processes, a label local to the element, a tuple of labels with one per component
            of a vector target, or the handle `namespace.<role>.dW` or `namespace.<role>.dW[k]` of the processes of a
            `WienerProcessSpec`.

        conserved:
            Whether the noise is the divergence of a random flux on the links of the lattice of a field.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import NoiseLaw
        >>> price = var('price')
        >>> NoiseLaw(price, 0.2 * price, process='market').check().run({'price': 10.0})
        2.0
    """

    def __init__(
        self,
        target: Target,
        value: object,
        order: int | None = None,
        *,
        convention: str = 'ito',
        process: Process = None,
        conserved: bool = False,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.value = value
        self.order = order
        self.convention = convention
        self.process = process
        self.conserved = conserved

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the noise drives."""
        return (self.target,)

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(ContinuousTime,))

    def _check_process(self) -> None:
        """Raise an error for a process that is not a label, a tuple of labels, or a handle, or a conserved one."""
        process = self.process
        valid = process is None or isinstance(process, str | Expression | FreeModuleElement)
        if isinstance(process, tuple):
            valid = bool(process) and all(isinstance(item, str) for item in process)
        if not valid:
            error_msg = (
                f'The process of the noise law of {self.target} is {process!r}. Give a label, a tuple of labels, or '
                'the handle `namespace.<role>.dW` of a `WienerProcessSpec`.'
            )
            raise ValidationError(error_msg)
        if self.conserved and process is not None:
            error_msg = (
                f'The conserved noise law of {self.target} names the process {process!r}, but it draws one process '
                'per link of the lattice. Remove the process.'
            )
            raise ValidationError(error_msg)

    def _check(self) -> Self:
        """Raise an error for a value that is a NumPy array, an unknown convention, or an invalid process.

        Returns:
            law:
                The law itself.
        """
        self._check_array('value')
        if self.convention not in NOISE_CONVENTIONS:
            error_msg = f'Unknown noise convention {self.convention!r}. Use one of {", ".join(NOISE_CONVENTIONS)}.'
            raise ValidationError(error_msg)
        self._check_process()
        return self

    def _run(self, inputs: object) -> object:
        """Return the amplitude at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.

        Returns:
            amplitude:
                The amplitude, a number when nothing symbolic is left.
        """
        return self._evaluate(self.value, inputs)
