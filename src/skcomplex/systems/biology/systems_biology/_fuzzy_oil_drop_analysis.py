"""Compare the hydrophobicity a protein's residues observe with an ideal micelle, after the fuzzy oil drop system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from scipy.special import rel_entr

from ....core.analysis import BaseAnalysis
from ....core.engine.runtime import Trajectory
from ....core.errors import AnalysisError
from ....core.types import Array, DataKind
from ..._extras import import_extra
from ._structure_spec import StructureSpec

KINDS = ('profile', 'relative-distance', 'environment-factor')
SIGMA_FRACTION = 3.0
CUTOFF = 9.0
LARGEST_FACTOR = 5.0
FRAME_DIMENSIONS = 3


def read_kyte_doolittle_scale() -> dict[str, float]:
    """Return the Kyte-Doolittle hydrophobicity of the twenty amino acids of Biopython, rescaled to `[0, 1]`.

    Returns:
        scale:
            The hydrophobicity by the three-letter name of a residue.

    Raises:
        MissingExtraError:
            The extra `structures` is not installed.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import read_kyte_doolittle_scale
        >>> scale = read_kyte_doolittle_scale()
        >>> scale['ILE'], scale['ARG']
        (1.0, 0.0)
    """
    system = 'fuzzy_oil_drop'
    letters = import_extra('Bio.SeqUtils.ProtParamData', system, 'structures').kd
    names = import_extra('Bio.Data.IUPACData', system, 'structures').protein_letters_1to3
    low, high = min(letters.values()), max(letters.values())
    return {names[letter].upper(): (value - low) / (high - low) for letter, value in letters.items()}


class FuzzyOilDropAnalysis(BaseAnalysis):
    """Distributions of hydrophobicity of a structure and their relative distance, after the fuzzy oil drop system.

    The theoretical distribution `T` is a three-dimensional Gaussian over the effective atoms along the principal axes
    of the chain (Eq. 1.1). The observed distribution `O` is Levitt's sum over the pairs of residues within a cutoff
    (Eq. 1.2). The uniform distribution `R` gives every residue the same value. Then
    `RD = D_KL(O|T) / (D_KL(O|T) + D_KL(O|R))` in bits (Eqs. 1.3 to 1.6), and `K` minimises `D_KL(O|M)` for the profile
    `M = [T + K [T_max - T]_n]_n` (Eq. 1.9), where `[...]_n` normalises to a sum of one, by SciPy's bounded scalar
    minimiser. The equations follow Konieczny et al. (2023 §1.3.1), and the curves of Fig. 1.41 give back its `K = 0.5`.
    Each sigma is a third of the largest absolute coordinate along its axis, one reading of the book's Gaussian that
    reaches near zero at `3 sigma` (p. 33 and Fig. 1.41), which gives no formula for sigma. The book names no scale, no
    orientation of the axes, no fit of `K`, and no upper bound of it, for which it reports values above 3 (p. 41), so
    the Kyte-Doolittle scale, the principal axes, the minimisation, and the bound 5 are choices of this class, as is
    leaving a residue out of its own sum in Eq. 1.2, where the book speaks of interactions between residues.

    The bundled entries do not reach the `RD < 0.5` and `K` below 0.5 that the book expects of fast-folding and
    single-chain proteins (p. 40): the mean `RD` and `K` are 0.77 and 2.2 for 1L2Y, 0.70 and 0.98 for 1UBQ, and 0.77 and
    2.2 for 5PTI. The gap comes from sigma, since a wider Gaussian closes it and no other choice does.

    Args:
        structure:
            The structure that gives the residues, and the systems when the source is `None`.

        kind:
            `'profile'`, `'relative-distance'`, or `'environment-factor'`.

        scale:
            The hydrophobicity by the three-letter name of a residue, or `None` for the Kyte-Doolittle scale rescaled to
            `[0, 1]`.

        cutoff:
            The cutoff distance of the observed distribution in ångström.

        observed:
            The observed distribution to use in place of Levitt's sum, one value per residue, or `None`.

        state:
            The dotted name of the lattice state of a trajectory that holds the coordinates, or `None` for the first
            state.

    Attributes:
        positions_:
            The effective atoms, frames by residues by 3.

        theoretical_:
            The distribution `T`, frames by residues.

        observed_:
            The distribution `O`, frames by residues.

        uniform_:
            The distribution `R`, one value per residue.

        divergences_:
            `D_KL(O|T)` and `D_KL(O|R)` in bits, frames by 2.

        values_:
            For the kind `'relative-distance'` the `RD`, and for the kind `'environment-factor'` the `K`, per frame.

    Raises:
        AnalysisError:
            The kind is unknown, the cutoff is not positive, the structure holds a residue the scale does not cover, or
            the positions do not hold one row per residue.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import FuzzyOilDropAnalysis, read_structure
        >>> values = FuzzyOilDropAnalysis(read_structure('1L2Y')).prepare().run().values_
        >>> values.shape, bool(values.min() >= 0 and values.max() <= 1)
        ((38,), True)
    """

    def __init__(
        self,
        structure: StructureSpec,
        kind: str = 'relative-distance',
        scale: Mapping[str, float] | None = None,
        cutoff: float = CUTOFF,
        *,
        observed: npt.ArrayLike | None = None,
        state: str | None = None,
    ) -> None:
        self.structure = structure
        self.kind = kind
        self.scale = scale
        self.cutoff = cutoff
        self.observed = observed
        self.state = state

    def _get_data_kind(self) -> DataKind:
        """Return the agents kind, since the source holds the positions of residues."""
        return DataKind.AGENTS

    def _check(self) -> None:
        """Raise an error for an unknown kind, or a cutoff that is not positive."""
        self._check_choice('kind', self.kind, KINDS)
        if self.cutoff <= 0:
            error_msg = f'The cutoff of the observed distribution must be positive, but it is {self.cutoff}.'
            raise AnalysisError(error_msg)

    def _read_hydrophobicity(self) -> Array:
        """Return the hydrophobicity of every residue, or raise an error that names one the scale does not cover."""
        scale = read_kyte_doolittle_scale() if self.scale is None else self.scale
        name = 'Kyte-Doolittle' if self.scale is None else 'given'
        for chain, number, residue in self.structure.residues:
            if residue not in scale:
                error_msg = (
                    f'The residue {residue} at number {number} of chain {chain} is not on the {name} scale, '
                    f'which covers {", ".join(sorted(scale))}. Pass a scale that covers it.'
                )
                raise AnalysisError(error_msg)
        return np.array([scale[residue] for _, _, residue in self.structure.residues], dtype=float)

    def _read_positions(self, source: object) -> Array:
        """Return the effective atoms of the systems, of a lattice state of a trajectory, or of an array."""
        if source is None:
            return self.structure.effective_atoms
        if isinstance(source, Trajectory):
            name = self.state if self.state is not None else source.problem.states.names[0]
            return np.asarray(source.lattice(name), dtype=float)
        return np.asarray(source, dtype=float)

    def _prepare(self, source: object) -> None:
        """Read the hydrophobicity of the residues and the effective atoms of the source."""
        self.hydrophobicity_ = self._read_hydrophobicity()
        self.positions_ = self._read_positions(source)
        count = len(self.hydrophobicity_)
        if self.positions_.ndim != FRAME_DIMENSIONS or self.positions_.shape[1] != count:
            error_msg = f'The positions are frames by {count} residues by 3, not the shape {self.positions_.shape}.'
            raise AnalysisError(error_msg)

    def _read_theoretical(self, positions: Array) -> Array:
        """Return the Gaussian over the positions along their principal axes, normalised."""
        centred = positions - positions.mean(axis=0)
        axes = np.linalg.svd(centred, full_matrices=False)[2]
        aligned = centred @ axes.T
        sigma = np.abs(aligned).max(axis=0) / SIGMA_FRACTION
        values = np.exp(-np.sum(aligned**2 / (2 * sigma**2), axis=1))
        return values / values.sum()

    def _read_observed(self, positions: Array) -> Array:
        """Return Levitt's sum over the pairs of residues within the cutoff, normalised."""
        ratio = np.linalg.norm(positions[:, None] - positions[None], axis=2) / self.cutoff
        weights = 1 - 0.5 * (7 * ratio**2 - 9 * ratio**4 + 5 * ratio**6 - ratio**8)
        weights[ratio > 1] = 0.0
        np.fill_diagonal(weights, 0.0)
        values = np.sum((self.hydrophobicity_[:, None] + self.hydrophobicity_[None]) * weights, axis=1)
        return values / values.sum()

    def _read_divergence(self, observed: Array, reference: Array) -> float:
        """Return the divergence of a distribution from a reference in bits."""
        return float(np.sum(rel_entr(observed, reference)) / np.log(2))

    def _read_environment_factor(self, observed: Array, theoretical: Array) -> float:
        """Return the factor `K` of the profile closest to the observed distribution."""
        complement = (theoretical.max() - theoretical) / (theoretical.max() - theoretical).sum()

        def divergence(factor: float) -> float:
            """Return the divergence of the observed distribution from the profile of a factor."""
            profile = theoretical + factor * complement
            return self._read_divergence(observed, profile / profile.sum())

        return float(minimize_scalar(divergence, bounds=(0.0, LARGEST_FACTOR), method='bounded').x)

    def _run(self, inputs: object = None) -> None:
        """Set the distributions, the divergences, and the values of the kind."""
        frames = len(self.positions_)
        self.theoretical_ = np.array([self._read_theoretical(positions) for positions in self.positions_])
        if self.observed is None:
            self.observed_ = np.array([self._read_observed(positions) for positions in self.positions_])
        else:
            self.observed_ = np.tile(np.asarray(self.observed, dtype=float), (frames, 1))
        self.uniform_ = np.full(self.positions_.shape[1], 1 / self.positions_.shape[1])
        self.divergences_ = np.array(
            [
                [self._read_divergence(observed, theoretical), self._read_divergence(observed, self.uniform_)]
                for observed, theoretical in zip(self.observed_, self.theoretical_, strict=True)
            ],
        )
        total = self.divergences_.sum(axis=1)
        if self.kind == 'relative-distance':
            self.values_ = np.divide(self.divergences_[:, 0], total, out=np.zeros(frames), where=total > 0)
        if self.kind == 'environment-factor':
            self.values_ = np.array(
                [
                    self._read_environment_factor(observed, theoretical)
                    for observed, theoretical in zip(self.observed_, self.theoretical_, strict=True)
                ],
            )
