"""Define mobility, the linear map of a Stokes fluid from the forces and torques on particles to their velocities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numba
import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, fast_callable, oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import CallSpec, ParameterSpec, RoleSpec
from ...types import Array, ElementKind
from ..laws import BaseLaw, RateLaw, UpdateLaw
from ..types import NamespaceFunction, PairPotential, Target
from ._interaction import Interaction

TENSORS = ('free-draining', 'oseen', 'rotne-prager-yamakawa')
FORMS = ('rate', 'map')
ROOTS = ('cholesky', 'lanczos')
DIMENSIONS = (2, 3)
SPACE = 3
FREE_DRAINING = TENSORS.index('free-draining')
REGULARIZED = TENSORS.index('rotne-prager-yamakawa')


@numba.njit(cache=True)
def _pair_coefficients(
    r: float,
    a: float,
    eta: float,
    regular: bool,
) -> tuple[float, float, float, float, float]:  # pragma: no cover
    """Return the coefficients of the mobility blocks of a pair of spheres at a distance `r`.

    The translational block is `t_i I + t_o u u` and the rotational block `r_i I + r_o u u`, with the unit separation
    `u`, and the rotation-translation block is `c [u]_x`, which gives both the angular velocity of a sphere from the
    force on the other and its velocity from the torque on the other. The regular coefficients are those of the Rotne-
    Prager-Yamakawa tensor, which overlapping spheres read in their regularized form, and the others those of the Oseen
    tensor and the rotlet.
    """
    if regular and r < 2 * a:
        ratio = r / a
        translation = 6 * math.pi * eta * a
        rotation = 8 * math.pi * eta * a**3
        return (
            (1 - 9 * ratio / 32) / translation,
            3 * ratio / 32 / translation,
            -ratio * (1 - 3 * ratio / 8) / (16 * math.pi * eta * a**2),
            (1 - 27 * ratio / 32 + 5 * ratio**3 / 64) / rotation,
            (9 * ratio / 32 - 3 * ratio**3 / 64) / rotation,
        )
    oseen = 1 / (8 * math.pi * eta * r)
    identity, outer = oseen, oseen
    if regular:
        identity, outer = oseen * (1 + 2 * a**2 / (3 * r**2)), oseen * (1 - 2 * a**2 / r**2)
    spin = 1 / (16 * math.pi * eta * r**3)
    return identity, outer, -1 / (8 * math.pi * eta * r**2), -spin, 3 * spin


@numba.njit(cache=True)
def _mobility_product(
    x: Array,
    force: Array,
    torque: Array,
    fluid: tuple[float, float],
    kind: int,
) -> tuple[Array, Array]:  # pragma: no cover
    """Return the velocities and angular velocities of the grand mobility applied to forces and torques.

    The fluid holds the radius of the spheres and the viscosity, and the kind is the index of the tensor in `TENSORS`.
    The free-draining kind couples no pair.
    """
    a, eta = fluid
    size = x.shape[0]
    velocity = force / (6 * math.pi * eta * a)
    angular = torque / (8 * math.pi * eta * a**3)
    if kind == FREE_DRAINING:
        return velocity, angular
    unit = np.zeros(3)
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            r = 0.0
            for k in range(3):
                unit[k] = x[i, k] - x[j, k]
                r += unit[k] ** 2
            r = math.sqrt(r)
            unit /= r
            t_i, t_o, c, r_i, r_o = _pair_coefficients(r, a, eta, kind == REGULARIZED)
            along_force = unit[0] * force[j, 0] + unit[1] * force[j, 1] + unit[2] * force[j, 2]
            along_torque = unit[0] * torque[j, 0] + unit[1] * torque[j, 1] + unit[2] * torque[j, 2]
            for k in range(3):
                first, second = (k + 1) % 3, (k + 2) % 3
                cross_force = unit[first] * force[j, second] - unit[second] * force[j, first]
                cross_torque = unit[first] * torque[j, second] - unit[second] * torque[j, first]
                velocity[i, k] += t_i * force[j, k] + t_o * along_force * unit[k] + c * cross_torque
                angular[i, k] += c * cross_force + r_i * torque[j, k] + r_o * along_torque * unit[k]
    return velocity, angular


@numba.njit(cache=True)
def _mobility_matrix(x: Array, fluid: tuple[float, float], kind: int) -> Array:  # pragma: no cover
    """Return the grand mobility, translations of every member first and then rotations, three components each."""
    a, eta = fluid
    size = x.shape[0]
    matrix = np.zeros((6 * size, 6 * size))
    shift = 3 * size
    unit = np.zeros(3)
    for i in range(size):
        for k in range(3):
            matrix[3 * i + k, 3 * i + k] = 1 / (6 * math.pi * eta * a)
            matrix[shift + 3 * i + k, shift + 3 * i + k] = 1 / (8 * math.pi * eta * a**3)
        for j in range(size):
            if i == j or kind == FREE_DRAINING:
                continue
            r = 0.0
            for k in range(3):
                unit[k] = x[i, k] - x[j, k]
                r += unit[k] ** 2
            r = math.sqrt(r)
            unit /= r
            t_i, t_o, c, r_i, r_o = _pair_coefficients(r, a, eta, kind == REGULARIZED)
            for k in range(3):
                first, second = (k + 1) % 3, (k + 2) % 3
                for m in range(3):
                    delta = 1.0 if k == m else 0.0
                    matrix[3 * i + k, 3 * j + m] = t_i * delta + t_o * unit[k] * unit[m]
                    matrix[shift + 3 * i + k, shift + 3 * j + m] = r_i * delta + r_o * unit[k] * unit[m]
                matrix[3 * i + k, shift + 3 * j + second] = c * unit[first]
                matrix[3 * i + k, shift + 3 * j + first] = -c * unit[second]
                matrix[shift + 3 * i + k, 3 * j + second] = c * unit[first]
                matrix[shift + 3 * i + k, 3 * j + first] = -c * unit[second]
    return matrix


@numba.njit(cache=True)
def _source_flows(x: Array, stresslet: Array, dipole: Array, eta: float) -> tuple[Array, Array]:  # pragma: no cover
    """Return the flow and half the vorticity at every member of the stresslets and potential dipoles of the others."""
    size = x.shape[0]
    velocity = np.zeros((size, 3))
    angular = np.zeros((size, 3))
    unit = np.zeros(3)
    stretched = np.zeros(3)
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            r = 0.0
            for k in range(3):
                unit[k] = x[i, k] - x[j, k]
                r += unit[k] ** 2
            r = math.sqrt(r)
            unit /= r
            for k in range(3):
                stretched[k] = (
                    stresslet[j, k, 0] * unit[0] + stresslet[j, k, 1] * unit[1] + stresslet[j, k, 2] * unit[2]
                )
            normal = unit[0] * stretched[0] + unit[1] * stretched[1] + unit[2] * stretched[2]
            along = unit[0] * dipole[j, 0] + unit[1] * dipole[j, 1] + unit[2] * dipole[j, 2]
            for k in range(3):
                first, second = (k + 1) % 3, (k + 2) % 3
                velocity[i, k] += (
                    -3 * normal * unit[k] / (8 * math.pi * eta * r**2) + (3 * along * unit[k] - dipole[j, k]) / r**3
                )
                angular[i, k] += (
                    3 * (unit[first] * stretched[second] - unit[second] * stretched[first]) / (8 * math.pi * eta * r**3)
                )
    return velocity, angular


class MobilityInteraction(Interaction):
    """Mobility of spheres in a Stokes fluid, the linear map from forces and torques to velocities and spins.

    Every member of the group `target` of spheres of radius `a` in a fluid of viscosity `eta` moves at the velocity
    `U_i` and spins at the angular velocity `Omega_i` of

    `(U_i, Omega_i) = sum_j M_ij (F_j, T_j) + (V_i, 0) + sum_{j != i} (u_j(x_i), w_j(x_i) / 2)`

    where `M` is the grand mobility, `F_j` and `T_j` the force and torque on member `j`, `V_i` an intrinsic velocity,
    and `u_j` and `w_j` the flow and vorticity of the force-free sources of member `j`. The self mobilities are
    `1 / (6 pi eta a)` and `1 / (8 pi eta a^3)`. Between members at the separation `r = x_i - x_j` the translational
    block is the Oseen tensor `(I + r r / r^2) / (8 pi eta r)`, or with `tensor='rotne-prager-yamakawa'` the
    Rotne-Prager-Yamakawa tensor, which adds the finite size `a^2 (I / 3 - r r / r^2) / (4 pi eta r^3)` and turns
    to its regularized form when the spheres overlap. The rotlet couples a torque on `j` to the velocity
    `T_j x r / (8 pi eta r^3)` of `i`, a force on `j` to the angular velocity `F_j x r / (8 pi eta r^3)` of `i`, and a
    torque to the angular velocity `(3 r r / r^2 - I) T_j / (16 pi eta r^3)`. With `tensor='free-draining'` no member
    moves another, the baseline without hydrodynamic interactions.

    The sources are a stresslet `S_j`, a symmetric traceless tensor whose flow is `-3 (r.S.r) r / (8 pi eta r^5)`, and
    a potential dipole `D_j`, whose flow is `(3 r r / r^2 - I) D_j / r^3`. Together with the intrinsic velocity they
    carry the active slip of a swimmer. These are point singularities, accurate at separations of a few radii. The
    forces of a pair `potential(namespace, r)` of the distance, a function of the distance and of the parameters of
    the interaction, act on every pair closer than `cutoff`, and add to the forces `F_j`. With an `orientation`, a unit
    vector state, the orientation turns as `dp/dt = Omega x p`.

    In the form `'rate'` the velocities are the rates of the position and the orientation in continuous time. In the
    form `'map'`, in discrete time, every step of length `dt` is the Ermak-McCammon step
    `x + U dt + sqrt(2 kT dt) (B xi)_x` at the temperature `kT`, where `B B^T = M` and `xi` is standard normal, and the
    orientation turns by `Omega dt + sqrt(2 kT dt) (B xi)_p` and is normalized. The divergence of both tensors vanishes
    in an unbounded fluid, so the step needs no drift correction. `root='cholesky'` factors the dense mobility, and
    `root='lanczos'` applies a Krylov approximation of its square root to `xi` with the matrix-free product, until the
    relative change of the sample falls below `tolerance`. In the map form the time of the namespace counts steps, so
    a force of the physical time reads `s.t * s.dt`.

    Positions in the plane are the plane `z = 0` of a three-dimensional fluid. A torque is then a number, the torque
    about `z`, and the stresslet is its block in the plane. The product costs `O(N^2)` per evaluation, Cholesky
    `O(N^3)`, and every evaluation runs as one call over the group.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of spheres.

        position:
            The name of the vector state of the positions, of two or three components.

        orientation:
            The name of the vector state of unit orientations, or `None` for spheres whose rotation is not tracked.

        tensor:
            `'free-draining'`, `'oseen'`, or `'rotne-prager-yamakawa'`.

        form:
            `'rate'` for continuous time without noise, or `'map'` for the Ermak-McCammon step in discrete time.

        force:
            The force on a member, a callable of the namespace that returns a vector, or `None` for no force.

        torque:
            The torque on a member, a callable of the namespace that returns a vector, a number in the plane, or `None`.

        velocity:
            The intrinsic velocity of a member, a callable of the namespace that returns a vector, or `None`.

        stresslet:
            The stresslet of a member, a callable of the namespace that returns a square matrix or nested sequences,
            or `None`.

        dipole:
            The potential dipole of a member, a callable of the namespace that returns a vector, or `None`.

        potential:
            The pair potential, a callable of the namespace and the distance, or `None` for no pair forces.

        cutoff:
            The largest distance of a pair of the potential, or `None` for every pair.

        root:
            `'cholesky'` or `'lanczos'`, the square root of the mobility the noise of the map form draws with.

        tolerance:
            The relative change of the Lanczos sample at which its iteration stops.

        radius:
            The radius `a` of the spheres.

        viscosity:
            The viscosity `eta` of the fluid.

        temperature:
            The thermal energy `kT` of the Brownian displacements of the map form, 0 for none.

        dt:
            The length of a step of the map form.

    Raises:
        ValidationError:
            A variant is unknown, a function is not callable, the rate form has a positive temperature, the positions
            have neither two nor three components, or noise draws from an Oseen mobility that is not positive
            definite.

    Examples:
        >>> import math
        >>> from typing import Annotated
        >>> from sage.all__sagemath_symbolics import vector
        >>> from skcomplex.core.components.elements import Element, GroupElement
        >>> from skcomplex.core.components.interactions import MobilityInteraction
        >>> from skcomplex.core.components.spaces import ContinuousSpace
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.core.specs import StateSpec
        >>> class SphereElement(Element):
        ...     def __init__(self, label, *, x: Annotated[list, StateSpec(vector=True, role='position')] = 0.0):
        ...         super().__init__(label)
        ...         self.x = x
        >>> pull = MobilityInteraction('stokes', target='spheres', force=lambda s: vector([1.0, 0.0, 0.0]))
        >>> spheres = GroupElement('spheres', element=SphereElement, size=2, values={'x': [[0, 0, 0], [0, 4, 0]]})
        >>> system = System(elements=[spheres], interactions=[pull], space=ContinuousSpace(n_dim=3))
        >>> trajectory = Simulation(system, dt=1.0).prepare().run(until=1.0).trajectory_
        >>> (trajectory.values('spheres.x')[-1, :, 0] * 6 * math.pi).round(4).tolist()
        [1.1953, 1.1953]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw, UpdateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.GROUP,), states=('position', 'orientation'))],
        position: str = 'x',
        orientation: str | None = None,
        tensor: str = 'rotne-prager-yamakawa',
        form: str = 'rate',
        force: NamespaceFunction | None = None,
        torque: NamespaceFunction | None = None,
        velocity: NamespaceFunction | None = None,
        stresslet: NamespaceFunction | None = None,
        dipole: NamespaceFunction | None = None,
        potential: PairPotential | None = None,
        cutoff: float | None = None,
        root: str = 'cholesky',
        tolerance: float = 1e-6,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        viscosity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        dt: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.position = position
        self.orientation = orientation
        self.tensor = tensor
        self.form = form
        self.force = force
        self.torque = torque
        self.velocity = velocity
        self.stresslet = stresslet
        self.dipole = dipole
        self.potential = potential
        self.cutoff = cutoff
        self.root = root
        self.tolerance = tolerance
        self.radius = radius
        self.viscosity = viscosity
        self.temperature = temperature
        self.dt = dt

    def _check(self) -> None:
        """Raise an error for an unknown variant, a function that is not callable, or noise in continuous time.

        Raises:
            ValidationError:
                A variant is unknown, a function is not callable, or the rate form has a positive temperature.
        """
        super()._check()
        self._check_variant('tensor', self.tensor, TENSORS)
        self._check_variant('form', self.form, FORMS)
        self._check_variant('root', self.root, ROOTS)
        for name in ('force', 'torque', 'velocity', 'stresslet', 'dipole', 'potential'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace or `None`.'
                raise ValidationError(error_msg)
        if self.form == 'rate' and isinstance(self.temperature, int | float) and self.temperature > 0:
            error_msg = (
                f'{type(self).__name__} {self.label!r} has the temperature {self.temperature!r} in the form `rate`, '
                "which has no noise. Use `form='map'` in discrete time for Brownian displacements."
            )
            raise ValidationError(error_msg)

    def _get_vector(self, function: NamespaceFunction | None, namespace: SimpleNamespace, size: int) -> list[Target]:
        """Return the components of a vector function of the namespace, zeros when it is `None`."""
        if function is None:
            return [SR(0)] * size
        values = function(namespace)
        components = list(values) if hasattr(values, '__len__') else [values]
        if len(components) != size:
            error_msg = f'A function of {self.label!r} returns {values!r}, but {size} components are expected.'
            raise ValidationError(error_msg)
        return [SR(component) for component in components]

    def _get_stresslet(self, namespace: SimpleNamespace, dimension: int) -> list[Target]:
        """Return the components of the stresslet of a member by rows, zeros when it is `None`."""
        if self.stresslet is None:
            return [SR(0)] * dimension**2
        matrix = self.stresslet(namespace)
        rows = [list(row) for row in matrix]
        if len(rows) != dimension or any(len(row) != dimension for row in rows):
            error_msg = (
                f'The stresslet of {self.label!r} is {matrix!r}, but a {dimension} by {dimension} matrix is expected.'
            )
            raise ValidationError(error_msg)
        return [SR(item) for row in rows for item in row]

    def _get_pair_force(self, namespace: SimpleNamespace) -> tuple[Callable[..., float] | None, list[Expression]]:
        """Return the radial force `-dU/dr` of the pair potential as a numeric function, with the symbols it reads."""
        if self.potential is None:
            return None, []
        distance = SR.var('pair_distance', domain='positive')
        slope = -SR(self.potential(namespace, distance)).diff(distance)
        symbols = [item for item in slope.variables() if not item.is_trivially_equal(distance)]
        return fast_callable(slope, vars=[distance, *symbols], domain=float), symbols

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate or update laws of the positions and orientations of the members.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the position and, when the members have one, of the orientation.

        Raises:
            ValidationError:
                The positions have neither two nor three components.
        """
        s = namespace
        member = self._get_member(s, 'target')
        position = getattr(member, self.position)
        dimension = len(position)
        if dimension not in DIMENSIONS:
            error_msg = f'The positions of {self.label!r} have {dimension} components. Use two or three.'
            raise ValidationError(error_msg)
        orientation = [] if self.orientation is None else list(getattr(member, self.orientation))
        pair_force, symbols = self._get_pair_force(s)
        arguments = (
            *position,
            *orientation,
            *self._get_vector(self.force, s, dimension),
            *self._get_vector(self.torque, s, SPACE if dimension == SPACE else 1),
            *self._get_vector(self.velocity, s, dimension),
            *self._get_stresslet(s, dimension),
            *self._get_vector(self.dipole, s, dimension),
            *(self._get_parameter_symbol(name, s) for name in ('radius', 'viscosity', 'temperature', 'dt')),
            *symbols,
        )
        law = RateLaw if self.form == 'rate' else UpdateLaw
        random = self.form == 'map'
        targets = [(position, 'position')] + (
            [] if self.orientation is None else [(getattr(member, self.orientation), 'orientation')]
        )
        return [
            law(
                target,
                CallSpec(
                    functools.partial(self._get_values, dimension, output, pair_force),
                    arguments,
                    random=random,
                    vectorized=True,
                ),
            )
            for target, output in targets
        ]

    @staticmethod
    def _embed(values: Array, dimension: int) -> Array:
        """Return vectors of the plane as vectors of space in the plane `z = 0`, or vectors of space as they are."""
        if dimension == SPACE:
            return np.ascontiguousarray(values, dtype=float)
        return np.ascontiguousarray(np.column_stack([values, np.zeros(values.shape[0])]), dtype=float)

    def _get_pair_forces(self, x: Array, pair_force: Callable[..., float] | None, constants: list[float]) -> Array:
        """Return the total force of the pair potential on every member."""
        forces = np.zeros_like(x)
        if pair_force is None:
            return forces
        separations = x[:, None, :] - x[None, :, :]
        distances = np.linalg.norm(separations, axis=-1)
        cutoff = np.inf if self.cutoff is None else self.cutoff
        first, second = np.nonzero(np.triu((distances > 0) & (distances <= cutoff), 1))
        if not first.size:
            return forces
        r = distances[first, second]
        magnitudes = np.array([pair_force(float(item), *constants) for item in r])
        pulls = (magnitudes / r)[:, None] * separations[first, second]
        np.add.at(forces, first, pulls)
        np.add.at(forces, second, -pulls)
        return forces

    def _get_indices(self, size: int, dimension: int) -> Array:
        """Return the indices of the grand mobility that the members move along: translations, then rotations."""
        axes = range(dimension)
        translations = [3 * member + axis for member in range(size) for axis in axes]
        if self.orientation is None:
            return np.array(translations, dtype=np.intp)
        spins = range(3) if dimension == SPACE else (2,)
        rotations = [3 * (size + member) + axis for member in range(size) for axis in spins]
        return np.array(translations + rotations, dtype=np.intp)

    def _get_cholesky_sample(self, x: Array, indices: Array, normal: Array, a: float, eta: float) -> Array:
        """Return `B xi` for the Cholesky factor `B` of the grand mobility on the indices the members move along."""
        matrix = _mobility_matrix(x, (a, eta), TENSORS.index(self.tensor))[np.ix_(indices, indices)]
        try:
            factor = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            error_msg = (
                f'The mobility of {self.label!r} is not positive definite, as the Oseen tensor is for spheres closer '
                "than about 1.5 diameters. Use `tensor='rotne-prager-yamakawa'`."
            )
            raise ValidationError(error_msg) from error
        sample: Array = factor @ normal
        return sample

    def _get_lanczos_sample(self, x: Array, indices: Array, normal: Array, a: float, eta: float) -> Array:
        """Return a Lanczos approximation of `M^(1/2) xi` with the matrix-free product of the grand mobility."""
        size = x.shape[0]
        kind = TENSORS.index(self.tensor)

        def apply(vector: Array) -> Array:
            full = np.zeros(6 * size)
            full[indices] = vector
            velocity, angular = _mobility_product(
                x,
                np.ascontiguousarray(full[: 3 * size].reshape(size, 3)),
                np.ascontiguousarray(full[3 * size :].reshape(size, 3)),
                (a, eta),
                kind,
            )
            product: Array = np.concatenate([velocity.ravel(), angular.ravel()])[indices]
            return product

        norm = float(np.linalg.norm(normal))
        basis = [normal / norm]
        diagonal: list[float] = []
        off: list[float] = []
        sample, previous = np.zeros_like(normal), None
        for _ in range(normal.size):
            work = apply(basis[-1])
            diagonal.append(float(basis[-1] @ work))
            stacked = np.array(basis)
            work -= stacked.T @ (stacked @ work)
            tridiagonal = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
            values, vectors = np.linalg.eigh(tridiagonal)
            if values.min() < -1e-12 * abs(values).max():
                error_msg = (
                    f'The mobility of {self.label!r} is not positive definite, as the Oseen tensor is for spheres '
                    "closer than about 1.5 diameters. Use `tensor='rotne-prager-yamakawa'`."
                )
                raise ValidationError(error_msg)
            root = vectors @ (np.sqrt(np.clip(values, 0, None)) * vectors[0])
            sample = norm * stacked.T @ root
            if previous is not None and np.linalg.norm(sample - previous) <= self.tolerance * np.linalg.norm(sample):
                break
            previous = sample
            beta = float(np.linalg.norm(work))
            if beta <= 1e-12 * norm:
                break
            off.append(beta)
            basis.append(work / beta)
        return sample

    def _get_motion(
        self,
        dimension: int,
        pair_force: Callable[..., float] | None,
        arguments: tuple[Array, ...],
        rng: np.random.Generator | None,
    ) -> tuple[Array, Array]:
        """Return the displacement of the positions per unit time and the rotation vectors of the orientations.

        In the rate form these are the velocities and angular velocities. In the map form they are the displacements and
        rotation vectors of one step, with the Brownian part.
        """
        size = arguments[0].shape[0]
        columns = iter(np.asarray(argument, dtype=float) for argument in arguments)

        def take(count: int) -> Array:
            return np.column_stack([next(columns) for _ in range(count)]) if count else np.zeros((size, 0))

        torques = SPACE if dimension == SPACE else 1
        x = self._embed(take(dimension), dimension)
        if self.orientation is not None:
            take(dimension)
        force = self._embed(take(dimension), dimension)
        torque = take(torques) if dimension == SPACE else np.column_stack([np.zeros((size, 2)), take(1)])
        velocity = self._embed(take(dimension), dimension)
        planar = take(dimension**2).reshape(size, dimension, dimension)
        stresslet = np.zeros((size, 3, 3))
        stresslet[:, :dimension, :dimension] = planar
        dipole = self._embed(take(dimension), dimension)
        a, eta, kT, dt = (float(next(columns)[0]) for _ in range(4))
        constants = [float(column[0]) for column in columns]
        force = force + self._get_pair_forces(x, pair_force, constants)
        kind = TENSORS.index(self.tensor)
        linear, spin = _mobility_product(x, force, np.ascontiguousarray(torque), (a, eta), kind)
        linear = linear + velocity
        if kind != FREE_DRAINING:
            flow, swirl = _source_flows(x, np.ascontiguousarray(stresslet), dipole, eta)
            linear, spin = linear + flow, spin + swirl
        if self.form == 'rate':
            return linear, spin
        linear, spin = linear * dt, spin * dt
        if kT > 0 and rng is not None:
            indices = self._get_indices(size, dimension)
            normal = rng.standard_normal(indices.size)
            draw = self._get_cholesky_sample if self.root == 'cholesky' else self._get_lanczos_sample
            full = np.zeros(6 * size)
            full[indices] = math.sqrt(2 * kT * dt) * draw(x, indices, normal, a, eta)
            linear = linear + full[: 3 * size].reshape(size, 3)
            spin = spin + full[3 * size :].reshape(size, 3)
        return linear, spin

    def _get_values(
        self,
        dimension: int,
        output: str,
        pair_force: Callable[..., float] | None,
        t: float,
        *arguments: Array,
        rng: np.random.Generator | None = None,
    ) -> Array:
        """Return the rates or next values of the positions or orientations of every member.

        The positions and the orientations share one evaluation of a time and arguments, so in the map form both read
        the same Brownian displacement.

        Args:
            dimension:
                The number of components of the positions.

            output:
                `'position'` or `'orientation'`.

            pair_force:
                The radial force of the pair potential, or `None`.

            t:
                The time.

            *arguments:
                The values of the arguments of the call, one array over the members each.

            rng:
                The random stream of the run, in the map form.

        Returns:
            values:
                One row per member and one column per component.
        """
        key = (float(t), *(np.asarray(argument, dtype=float).tobytes() for argument in arguments))
        cache = getattr(self, '_cache', None)
        if cache is None or cache[0] != key:
            cache = (key, self._get_motion(dimension, pair_force, arguments, rng))
            self._cache = cache
        linear, spin = cache[1]
        size = linear.shape[0]
        if output == 'position':
            position = np.column_stack([np.asarray(item, dtype=float) for item in arguments[:dimension]])
            moved: Array = linear[:, :dimension]
            return moved if self.form == 'rate' else position + moved
        heading = self._embed(
            np.column_stack([np.asarray(item, dtype=float) for item in arguments[dimension : 2 * dimension]]),
            dimension,
        )
        turn = np.cross(spin, heading)
        if self.form == 'rate':
            rate: Array = turn[:, :dimension]
            return rate
        turned = heading + turn
        turned /= np.linalg.norm(turned, axis=1, keepdims=True).reshape(size, 1)
        result: Array = turned[:, :dimension]
        return result
