"""Define noise laws, amplitudes of the noise processes that drive states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import ContinuousTime
from ..types import Process, Target
from ._base import BaseLaw

NOISE_CONVENTIONS = {'ito': 0.0, 'stratonovich': 0.5, 'anti-ito': 1.0}


class NoiseLaw(BaseLaw):
    """Amplitude of a noise process that drives a state's highest time derivative.

    The process is a Wiener process, or a fractional Brownian motion of a Hurst index.

    Without a process, every component of the target and every member of a group gets its own independent Wiener
    process. Laws of one component, or of one member, that name the same local process share its increment. Laws that
    name a process of a `WienerProcessSpec` through a role share it across components and members.

    A conserved noise on a state of a field on a lattice adds `div(value eta)` instead, the finite-difference divergence
    of a random flux with one Wiener process per link between neighbouring sites, so the total of the field changes
    only by its rate laws.

    A Hurst index makes the process a fractional Brownian motion. Its increments are fractional Gaussian noise,
    correlated over the whole run. A Hurst index of 1/2 gives the Wiener process, below 1/2 the increments
    anticorrelate, and above 1/2 they correlate. For other indices than 1/2 the amplitude must not depend on any state,
    and the law takes neither a process nor a conserved noise.

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

        hurst:
            The Hurst index `0 < H < 1` of a fractional Brownian motion, a number or a parameter, or `None` for a
            Wiener process.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import NoiseLaw
        >>> x = var('x')
        >>> NoiseLaw(x, 0.2 * x, process='shared').prepare().run({'x': 10.0}).value_
        2.0
        >>> D = var('D')
        >>> NoiseLaw(x, 2 * D, hurst=0.7).hurst
        0.7
    """

    law_tag: ClassVar[str] = 'noise'

    def __init__(
        self,
        target: Target,
        value: object,
        order: int | None = None,
        *,
        convention: str = 'ito',
        process: Process = None,
        conserved: bool = False,
        hurst: object = None,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.value = value
        self.order = order
        self.convention = convention
        self.process = process
        self.conserved = conserved
        self.hurst = hurst

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the noise drives."""
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(ContinuousTime,))

    def _check_process(self) -> None:
        """Raise an error for a process that is not a label, labels, or a handle.

        It also raises it for a conserved process, and the labels are a tuple of labels.
        """
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

    def _check_hurst(self) -> None:
        """Raise an error for a Hurst index outside `(0, 1)`.

        It also raises it for a Hurst index with a process or on a conserved noise.
        """
        if self.hurst is None:
            return
        if isinstance(self.hurst, int | float) and not 0 < self.hurst < 1:
            error_msg = f'The Hurst index of the noise of {self.target} is {self.hurst}. Give a number in (0, 1).'
            raise ValidationError(error_msg)
        if self.process is not None:
            error_msg = (
                f'The noise law of {self.target} has a Hurst index and names the process {self.process!r}, but a '
                'fractional Brownian motion is drawn for the law alone. Remove the process.'
            )
            raise ValidationError(error_msg)
        if self.conserved:
            error_msg = (
                f'The conserved noise law of {self.target} has a Hurst index, but a conserved noise is Brownian. '
                'Remove the Hurst index.'
            )
            raise ValidationError(error_msg)

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for an array value, an unknown convention, or an invalid process.

        It also raises it for an invalid Hurst index.

        The array is a NumPy array.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        self._check_array('value')
        if self.convention not in NOISE_CONVENTIONS:
            error_msg = f'Unknown noise convention {self.convention!r}. Use one of {", ".join(NOISE_CONVENTIONS)}.'
            raise ValidationError(error_msg)
        self._check_process()
        self._check_hurst()

    def _run(self, inputs: object) -> None:
        """Set `value_` to the amplitude at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(self.value, inputs)
