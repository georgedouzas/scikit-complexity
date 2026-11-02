"""Uninfected and infected tumour cells, an oncolytic virus, and a chemotherapeutic drug."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....errors import ParameterError
from ....specs import ParameterSpec, StateSpec, VisualSpec

INFUSIONS = ('constant', 'exponential', 'periodic')


class ChemovirotherapyTumourElement(Element):
    """Uninfected and infected tumour cells, an oncolytic virus, and a chemotherapeutic drug.

    The uninfected cells `AU` grow logistically, the virus `V` infects them, and infected cells `AV` burst at the rate
    `a2`, releasing `beta_v` virions each. The drug `U` kills both kinds of cells and is infused at the rate `u1`,
    constantly, as `u1 exp(-d_u t)`, or as `u1 sin(d_u t)^2`. One unit of cells is a million cells, and one unit of time
    is two days.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `chemovirotherapy` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AU:
            The initial number of uninfected cells.

        AV:
            The initial number of infected cells.

        V:
            The initial number of virions.

        U:
            The initial amount of drug.

        r1:
            The growth rate of uninfected cells.

        b:
            The reciprocal carrying capacity.

        alpha:
            The infection rate.

        a1:
            The drug kill rate of uninfected cells.

        a2:
            The rate at which the virus kills infected cells.

        a3:
            The drug kill rate of infected cells.

        beta_v:
            The burst size.

        d1:
            The decay rate of the virus.

        d_u:
            The decay rate of the drug.

        u1:
            The infusion rate of the drug.

        infusion:
            The form of the infusion, `'constant'`, `'exponential'`, or `'periodic'`.

    Raises:
        ParameterError:
            The infusion is not one of the known forms.

    Examples:
        >>> from skcomplex.models.biology.oncology import ChemovirotherapyTumourElement
        >>> ChemovirotherapyTumourElement('tumour', infusion='periodic').infusion
        'periodic'
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AU: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        AV: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        V: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.206,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1e-06,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5115,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.006,
        beta_v: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        d_u: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.16,
        u1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        infusion: str = 'constant',
    ) -> None:
        super().__init__(label, visual=visual)
        self.AU = AU
        self.AV = AV
        self.V = V
        self.U = U
        self.r1 = r1
        self.b = b
        self.alpha = alpha
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.beta_v = beta_v
        self.d1 = d1
        self.d_u = d_u
        self.u1 = u1
        self.infusion = infusion

    def _check(self) -> None:
        """Raise an error for an unknown infusion.

        Raises:
            ParameterError:
                The infusion is not one of the known forms.
        """
        super()._check()
        if self.infusion not in INFUSIONS:
            error_msg = f'The infusion must be one of {INFUSIONS}, got {self.infusion!r}.'
            raise ParameterError(error_msg)
