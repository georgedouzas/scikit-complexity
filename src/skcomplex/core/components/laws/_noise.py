"""Define the law that gives the amplitude of a noise that drives a state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...errors import ValidationError
from ..spaces._continuous_time import _ContinuousTime
from ..types import Process, Target
from ._base import BaseLaw
from ._requirement_spec import RequirementSpec

NOISE_CONVENTIONS = {'ito': 0.0, 'stratonovich': 0.5, 'anti-ito': 1.0}


@dataclass(frozen=True)
class NoiseLaw(BaseLaw):
    """A law that adds a random change to a time derivative of a value.

    It shakes a value at random in continuous time, adding `value dW` to the derivative it targets, where `dW` is the
    random step of a Wiener process over a time step, such as `NoiseLaw(s.stock.p, s.stock.sigma * s.stock.p)` for the
    random moves of a stock price. A `RateLaw` on the same value gives the mean change, and together they form a
    stochastic differential equation. Every member and every entry of a vector draws its own random steps, unless laws
    name the same process and share them. A conserved noise on a lattice moves amounts between neighbouring sites, so
    the total stays the same, and a Hurst index makes the steps remember the past, as in a fractional Brownian motion.
    A stock price moves at random, a particle diffuses in a fluid, molecules shake the expression of a gene, and
    turbulence stirs the temperature of a fluid.

    Read more in the user guide, under Laws.

    Args:
        target:
            The symbol of a scalar value, a derivative of it such as `x.diff(t)`, or the column of symbols of a vector
            value.

        value:
            The size of the random change, a symbolic expression, a `CallSpec`, or a function `f(t, values)`. A vector
            target draws one process per entry.

        convention:
            How the random change reads the value it changes, `'ito'` at the start of the step, `'stratonovich'` at
            its middle, or `'anti-ito'` at its end. The compiler rewrites the law in the form of `'ito'`, which every
            solver runs.

        process:
            `None` for random steps of its own, a label that laws of the element share, a tuple of labels with one per
            entry of a vector target, or the random steps `s.process.dW` or `s.process.dW[k]` of a
            `WienerProcessSpec` that interactions share.

        conserved:
            Whether the noise moves amounts between neighbouring sites of a lattice, with random steps of its own on
            every link, so the total of the field changes only by its rate laws.

        hurst:
            The Hurst index `0 < H < 1`, a number or a parameter, whose steps go on in the same direction above 1/2
            and turn back below it, or `None` for independent steps.

        label:
            The name of the law, or the empty string for a law without one.

    See Also:
        RateLaw: Adds the mean change to a derivative of a value.
        WienerProcessSpec: Random steps that several interactions share.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import NoiseLaw
        >>> p, sigma = var('p sigma')
        >>> NoiseLaw(p, sigma * p, process='market').value
        p*sigma
    """

    law_tag: ClassVar[str] = 'noise'

    target: Target
    value: object
    _: KW_ONLY
    convention: str = 'ito'
    process: Process = None
    conserved: bool = False
    hurst: object = None
    label: str = ''

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the noise drives."""
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(_ContinuousTime,))

    def _get_order(self) -> float | None:
        """Return the order of the derivative the noise drives, `None` for its state."""
        return None

    def _check_process(self) -> None:
        """Raise an error for a process that is not a label, labels, or shared steps.

        It also raises it for a conserved noise that names a process.
        """
        process = self.process
        valid = process is None or isinstance(process, str | Expression | FreeModuleElement)
        if isinstance(process, tuple):
            valid = bool(process) and all(isinstance(item, str) for item in process)
        if not valid:
            error_msg = (
                f'The process of the noise law of {self.target} is {process!r}. Give a label, a tuple of labels, or '
                'the random steps `s.process.dW` of a `WienerProcessSpec`.'
            )
            raise ValidationError(error_msg)
        if self.conserved and process is not None:
            error_msg = (
                f'The conserved noise law of {self.target} names the process {process!r}, but it draws its own '
                'random steps on every link of the lattice. Remove the process.'
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
                'noise with a Hurst index draws its own random steps. Remove the process.'
            )
            raise ValidationError(error_msg)
        if self.conserved:
            error_msg = (
                f'The conserved noise law of {self.target} has a Hurst index, but a conserved noise draws '
                'independent steps. Remove the Hurst index.'
            )
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for a NumPy array, an unknown convention, or an invalid process.

        It also raises it for an invalid Hurst index.
        """
        super()._check()
        self._check_array('value')
        if self.convention not in NOISE_CONVENTIONS:
            error_msg = f'Unknown noise convention {self.convention!r}. Use one of {", ".join(NOISE_CONVENTIONS)}.'
            raise ValidationError(error_msg)
        self._check_process()
        self._check_hurst()

    def _get_value(self, inputs: object) -> object:
        """Return the amplitude at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        return self._evaluate(self.value, inputs)
