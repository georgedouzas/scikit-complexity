"""Define the Landau-de Gennes free energy of a symmetric traceless alignment tensor."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet
from sage.structure.element import Matrix

from ....core.interactions import GradientInteraction
from ....core.laws.utils import laplacian
from ....specs import ParameterSpec, RoleSpec


class LandauDeGennesEnergyInteraction(GradientInteraction):
    """Gradient flow of a symmetric traceless tensor `Q` in the Landau-de Gennes free energy.

    The free energy is `F = integral a tr(Q^2) / 2 + c tr(Q^2)^2 / 4 + K |grad Q|^2 / 2`. Its negative functional
    derivative among symmetric traceless tensors is the molecular field `H = -a Q - c tr(Q^2) Q + K laplacian(Q)`, and
    the tensor relaxes as `Q_t = H / gamma`. A uniform tensor `S (n n - I / 2)` in two dimensions with `a < 0` relaxes
    to the scalar order `S = sqrt(-2 a / c)`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the field.

        state:
            The name of the state of the field that holds the tensor.

        a:
            The coefficient of `tr(Q^2) / 2`, negative in the ordered phase.

        c:
            The coefficient of `tr(Q^2)^2 / 4`.

        K:
            The elastic constant.

        gamma:
            The rotational viscosity, which divides the molecular field.

        quadratic:
            A function of the namespace that returns the coefficient of `tr(Q^2) / 2` instead of `a`, such as an
            expression of a density, or `None` for `a`.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, matrix, var
        >>> from skcomplex.models.physics.active_matter import LandauDeGennesEnergyInteraction
        >>> q, a, c = var('q, a, c')
        >>> tensor = matrix(SR, [[q, 0], [0, -q]])
        >>> LandauDeGennesEnergyInteraction._molecular_field(tensor, a, c, 0)[0, 0].factor()
        -(2*c*q^2 + a)*q
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'Q',
        a: Annotated[float, ParameterSpec()] = -1.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        quadratic: Callable[[SimpleNamespace], object] | None = None,
    ) -> None:
        super().__init__(label, target=target, states=state, variational=True)
        self.state = state
        self.a = a
        self.c = c
        self.K = K
        self.gamma = gamma
        self.quadratic = quadratic

    @staticmethod
    def _density(tensor: Matrix, a: object, c: object, K: object) -> Expression:
        """Return the density of the free energy, with the gradient term by parts."""
        square = (tensor * tensor).trace()
        return a * square / 2 + c * square**2 / 4 - K * (tensor * tensor.apply_map(laplacian)).trace() / 2

    @classmethod
    def _molecular_field(cls, tensor: Matrix, a: object, c: object, K: object) -> Matrix:
        """Return the molecular field of a tensor, the negative functional derivative of the free energy."""
        energy = cls('energy', target='field')
        return -energy._get_matrix_derivative(cls._density(tensor, a, c, K), tensor)

    def _get_potential(self, namespace: SimpleNamespace) -> Expression:
        """Return the density of the free energy over the rotational viscosity, whose gradient is `H / gamma`.

        Args:
            namespace:
                The namespace of the symbols the density can use.

        Returns:
            density:
                The density that the tensor moves down.
        """
        quadratic = namespace.a if self.quadratic is None else self.quadratic(namespace)
        tensor = getattr(namespace.target, self.state)
        return self._density(tensor, quadratic, namespace.c, namespace.K) / namespace.gamma
