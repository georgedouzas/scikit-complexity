"""Measure spectra: power spectra of series, structure factors of fields, and dispersion relations of field models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR
from scipy import optimize, signal

from ..core.elements import FieldElement
from ..core.errors import AnalysisError
from ..core.laws import RateLaw
from ..core.spaces import LatticeSpace
from ..core.system import System
from ..engine.problems import VariableKind, declarations_of, symbol
from ..engine.runtime import Trajectory
from ._base import BaseAnalysis, Source
from ._kinds import DataKind
from ._utils import real_array

type Array = npt.NDArray[np.float64]
KINDS = ('power', 'structure-factor', 'dispersion')
SETTINGS = {
    'power': {'dt'},
    'structure-factor': {'spacing', 'frames'},
    'dispersion': {'at'},
}
"""Settings every kind reads, besides `kind` and `variables`."""
OPERATOR_NAMES = ('laplacian', 'partial', 'upwind', 'inverse_laplacian')
EIGENVALUE = symbol('_eigenvalue_')


def _power_spectrum(values: npt.ArrayLike, dt: float = 1.0) -> tuple[Array, Array]:
    """Return the frequencies and the power spectral density of a series by Welch's method."""
    array = real_array(values, 'values')
    return signal.welch(array, fs=1 / dt, nperseg=min(array.size, 1024))


def _wavenumbers(shape: tuple[int, ...], spacing: float) -> Array:
    """Return the modulus of the wave vector of every Fourier mode of a periodic grid."""
    axes = np.meshgrid(*(2 * np.pi * np.fft.fftfreq(size, spacing) for size in shape), indexing='ij')
    return np.sqrt(sum(axis**2 for axis in axes))


def _mode_power(grids: Array) -> Array:
    """Return the power of every Fourier mode of the fluctuations, averaged over frames."""
    fluctuations = grids - grids.mean(axis=tuple(range(1, grids.ndim)), keepdims=True)
    axes = tuple(range(1, grids.ndim))
    return np.mean(np.abs(np.fft.fftn(fluctuations, axes=axes)) ** 2, axis=0) / np.prod(grids.shape[1:])


def _structure_factor(grids: Array, spacing: float = 1.0) -> tuple[Array, Array]:
    """Return the wavenumbers and the radially averaged structure factor of frames, in shells of `2 pi / side`."""
    power = _mode_power(grids)
    moduli = _wavenumbers(grids.shape[1:], spacing)
    width = 2 * np.pi / (max(grids.shape[1:]) * spacing)
    shells = np.rint(moduli / width).astype(int).ravel()
    sums = np.bincount(shells, power.ravel())
    counts = np.bincount(shells)
    used = counts > 0
    used[0] = False
    return np.flatnonzero(used) * width, sums[used] / counts[used]


def _with_eigenvalue(expression: Any, eigenvalue: Any) -> Any:  # noqa: ANN401
    """Return an expression with every Laplacian, nested ones included, as a multiplication by its eigenvalue."""
    operator = expression.operator()
    if operator is None:
        return expression
    name = getattr(operator, 'name', lambda: '')()
    if name == 'laplacian':
        return eigenvalue * _with_eigenvalue(expression.operands()[0], eigenvalue)
    if name in OPERATOR_NAMES:
        error_msg = f'The dispersion relation linearises laplacian only, but a law applies {name}.'
        raise AnalysisError(error_msg)
    return operator(*[_with_eigenvalue(item, eigenvalue) for item in expression.operands()])


def _laplacian_eigenvalues(field: FieldElement, lattice: LatticeSpace, waves: Array) -> Array:
    """Return the eigenvalue of the field's Laplacian at every wave vector, for its discretization."""
    spacings = np.asarray(lattice.spacings_, dtype=float)
    if field.discretization == 'spectral':
        return -np.sum(waves**2, axis=1)
    return np.sum((2 * np.cos(waves * spacings) - 2) / spacings**2, axis=1)


def _single_field(system: System) -> tuple[FieldElement, LatticeSpace]:
    """Return the one field of a system and its periodic lattice, or raise an error."""
    fields = [element for element in system.elements if isinstance(element, FieldElement)]
    if len(fields) != 1 or len(system.elements) != 1 or system.interactions or system.systems:
        error_msg = 'The dispersion relation takes a system of one field on a lattice.'
        raise AnalysisError(error_msg)
    field = fields[0]
    if not isinstance(field.space, LatticeSpace) or field.space.boundary != 'periodic':
        error_msg = f'The field {field.label} is not on a periodic lattice, so its modes are not Fourier modes.'
        raise AnalysisError(error_msg)
    return field, field.space.check().prepare()


def _linearised_rates(field: FieldElement) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the rate of every state with its Laplacians as eigenvalues, the state symbols, and the field's values."""
    states = list(declarations_of(field.element, VariableKind.STATE))
    member = field.element(field.label)
    values = {name: getattr(member, name) for name in states} | dict(field.values or {})
    parameters = {}
    for name in declarations_of(field.element, VariableKind.PARAMETER):
        value = values.get(name, getattr(member, name))
        if np.ndim(value):
            error_msg = f'The parameter {field.label}.{name} varies over the lattice, so there is no uniform state.'
            raise AnalysisError(error_msg)
        parameters[name] = float(value)
    names = {name: symbol(f'_{name}_') for name in states}
    rates: dict[str, Any] = {}
    for law in member.check().run(SimpleNamespace(**names, **parameters, t=symbol('t'))):
        targets = law.check()._get_targets() if isinstance(law, RateLaw) else ()
        target = next((name for name, item in names.items() if targets and targets[0] is item), None)
        if not isinstance(law, RateLaw) or target is None:
            error_msg = f'The dispersion relation takes rate laws of the states of {field.label}, but got {law!r}.'
            raise AnalysisError(error_msg)
        rates[target] = SR(rates.get(target, 0) + _with_eigenvalue(SR(law.value), EIGENVALUE))
    return {name: rates[name] for name in states if name in rates}, names, values


def _fixed_point(rates: dict[str, Any], variables: list[Any]) -> Any:  # noqa: ANN401
    """Return the function that finds the uniform fixed point of the rates nearest a starting point."""
    local = [rate.subs({EIGENVALUE: 0}) for rate in rates.values()]

    def residual(point: Array) -> list[float]:
        substitution = dict(zip(variables, map(float, point), strict=True))
        return [float(item.subs(substitution)) for item in local]

    return lambda start: [float(item) for item in optimize.fsolve(residual, start)]


def _eigenvalue_terms(entry: Any, name: str) -> list[tuple[float, int]]:  # noqa: ANN401
    """Return the coefficients and powers of the Laplacian eigenvalue in an entry of the linearisation."""
    try:
        return [(float(coefficient), int(power)) for coefficient, power in entry.coefficients(EIGENVALUE)]
    except (TypeError, ValueError) as error:
        error_msg = f'The linearisation of {name} is not a polynomial in its Laplacians: {entry}.'
        raise AnalysisError(error_msg) from error


def _dispersion_relation(
    system: System,
    at: Mapping[str, float] | None = None,
) -> tuple[Array, Array, Array, Array, dict[str, float]]:
    """Return the wave vectors, their lengths, the growth rates, the frequencies, and the uniform state of a field."""
    field, lattice = _single_field(system)
    rates, names, values = _linearised_rates(field)
    order = list(rates)
    variables = [names[name] for name in order]
    start = [float(np.mean((at or {}).get(name, values.get(name, 0.0)))) for name in order]
    uniform = start if at is not None else _fixed_point(rates, variables)(start)
    substitution = dict(zip(variables, uniform, strict=True))
    axes = [2 * np.pi * np.fft.fftfreq(n, d=h) for n, h in zip(lattice.shape_, lattice.spacings_, strict=True)]
    waves = np.stack([grid.ravel() for grid in np.meshgrid(*axes, indexing='ij')], axis=1)
    eigenvalues = _laplacian_eigenvalues(field, lattice, waves)
    matrices = np.zeros((len(waves), len(order), len(order)))
    for row, a in enumerate(order):
        for column, b in enumerate(order):
            entry = SR(rates[a].diff(names[b]).subs(substitution)).expand()
            for coefficient, power in _eigenvalue_terms(entry, f'{field.label}.{a}'):
                matrices[:, row, column] += coefficient * eigenvalues**power
    spectra = np.linalg.eigvals(matrices)
    leading = spectra[np.arange(len(waves)), np.argmax(spectra.real, axis=1)]
    magnitudes = np.linalg.norm(waves, axis=1)
    return waves, magnitudes, leading.real, leading.imag, dict(zip(order, uniform, strict=True))


class SpectrumAnalysis(BaseAnalysis):
    """Spectrum of every series of a source, of a field, or of a field model linearised around a uniform state.

    The kinds read series, fields, or a system:

    - `'power'`: the power spectral density of every series by Welch's method.
    - `'structure-factor'`: the radially averaged structure factor `S(k)` of a field on a periodic grid of any
      dimension, in shells of width `2 pi / (side spacing)`, averaged over frames when `frames` is true.
    - `'dispersion'`: the growth rate of every Fourier mode of a system of one field, a site group on a periodic
      lattice whose laws are rate laws. The laws are linearised symbolically around a uniform state, and every
      Laplacian, nested ones included, becomes its eigenvalue on a Fourier mode of the lattice,
      `Σ (2 cos(k h) - 2) / h²` for finite differences and `-|k|²` for a spectral field. Without `at`, the uniform
      state is the fixed point of the laws nearest the mean initial values.

    A structure factor reads the last frame of a trajectory, or every frame when `frames` is true.

    Args:
        kind:
            The spectrum to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series, or the first field for a field.

        dt:
            The time between samples.

        spacing:
            The distance between neighbouring sites.

        frames:
            Whether the first axis of an array indexes frames, and whether to read every frame of a trajectory.

        at:
            The uniform state of the dispersion relation by state name, or `None` for the nearest fixed point.

    Attributes:
        values_:
            The series, one per column, or the field, or the system.

        labels_:
            The label of every series, or of the field or system.

        frequencies_:
            The frequencies of `'power'`, or the imaginary part of the leading eigenvalue at every mode of
            `'dispersion'`, nonzero for travelling or oscillating modes.

        power_:
            One density per series and frequency for `'power'`, and the mean power in every shell for
            `'structure-factor'`.

        wavenumbers_:
            The wavenumbers of the shells, without 0, for `'structure-factor'`, and the wave vector of every mode, one
            row per mode, for `'dispersion'`.

        magnitudes_:
            The length of every wave vector of `'dispersion'`.

        growth_rates_:
            The largest real part of the eigenvalues of the linearisation at every mode of `'dispersion'`.

        uniform_:
            The uniform state the model is linearised around, by state name.

        band_:
            The lengths of the wave vectors whose modes grow.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import SpectrumAnalysis
        >>> times = np.arange(4096) * 0.01
        >>> spectrum = SpectrumAnalysis(dt=0.01).check().prepare(np.sin(2 * np.pi * 5 * times)).run()
        >>> round(float(spectrum.frequencies_[spectrum.power_[0].argmax()]))
        5
        >>> stripes = np.cos(2 * np.pi * np.arange(64) / 8)[:, None] * np.ones(64)
        >>> factor = SpectrumAnalysis('structure-factor').check().prepare(stripes).run()
        >>> round(float(2 * np.pi / factor.wavenumbers_[factor.power_.argmax()]), 1)
        8.0
        >>> from skcomplex.models.chemistry.pattern_formation import brusselator
        >>> relation = SpectrumAnalysis('dispersion').check().prepare(brusselator((64,), b=1.9, Dv=8.0)).run()
        >>> bool(relation.growth_rates_.max() > 0), bool(relation.growth_rates_[0] < 0)
        (True, True)
    """

    def __init__(
        self,
        kind: str = 'power',
        variables: Sequence[str] | None = None,
        *,
        dt: float = 1.0,
        spacing: float = 1.0,
        frames: bool = False,
        at: Mapping[str, float] | None = None,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.dt = dt
        self.spacing = spacing
        self.frames = frames
        self.at = at

    def _check(self) -> Self:
        """Validate the kind, the numeric settings, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.dt <= 0 or self.spacing <= 0:
            error_msg = f'{type(self).__name__} has dt={self.dt!r} and spacing={self.spacing!r}. Use positive values.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the series kind for power spectra, the field kind for fields, and the system kind for field models."""
        if self.kind == 'power':
            return DataKind.SERIES
        return DataKind.SYSTEM if self.kind == 'dispersion' else DataKind.FIELD

    def _prepare(self, source: Source) -> Self:
        """Read the series as columns, the frames of a field, or the system of a field model."""
        if self.kind == 'power':
            series = self._read_series(source, self.variables)
            self.values_ = series.values
            self.labels_ = series.labels
            return self
        if self.kind == 'dispersion':
            if not isinstance(source, System):
                error_msg = f'A dispersion relation is computed from a system, not from {type(source).__name__}.'
                raise AnalysisError(error_msg)
            self.values_ = source
            self.labels_ = [type(source).__name__]
            return self
        name = self.variables[0] if self.variables else None
        field = self._read_fields(source, name)
        if isinstance(source, Trajectory) and not self.frames:
            field = field[-1]
        self.values_ = field
        self.labels_ = [name or 'field']
        return self

    def _run(self) -> Self:
        """Compute the spectrum of the kind."""
        if self.kind == 'power':
            spectra = [_power_spectrum(column, self.dt) for column in self.values_.T]
            self.frequencies_ = spectra[0][0]
            self.power_ = np.asarray([density for _, density in spectra])
        elif self.kind == 'structure-factor':
            grids = self.values_ if self.frames else self.values_[None]
            self.wavenumbers_, self.power_ = _structure_factor(grids, self.spacing)
        else:
            waves, magnitudes, growth_rates, frequencies, uniform = _dispersion_relation(self.values_, self.at)
            self.wavenumbers_ = waves
            self.magnitudes_ = magnitudes
            self.growth_rates_ = growth_rates
            self.frequencies_ = frequencies
            self.uniform_ = uniform
            self.band_ = np.unique(np.round(magnitudes[growth_rates > 0], 12))
        return self
