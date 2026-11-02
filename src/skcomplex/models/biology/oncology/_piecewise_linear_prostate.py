"""Androgen-dependent, reversibly and irreversibly independent prostate cancer cells under intermittent therapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class PiecewiseLinearProstateElement(Element):
    """Androgen-dependent, reversibly and irreversibly independent prostate cancer cells under intermittent therapy.

    During therapy, `u = 1`, the cells follow `dAD/dt = w11_on AD`, `dAIR/dt = w21_on AD + w22_on AIR`, and
    `dAII/dt = w31_on AD + w32_on AIR + w33_on AII`. Off therapy they follow `dAD/dt = w11_off AD + w12_off AIR`,
    `dAIR/dt = w22_off AIR`, and `dAII/dt = w33_off AII`. Switches on the PSA `x1 AD + x2 AIR + x3 AII` set the
    therapy `u`.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AD:
            The initial androgen-dependent cells.

        AIR:
            The initial reversibly independent cells.

        AII:
            The initial irreversibly independent cells.

        u:
            The initial therapy, 1 on and 0 off.

        w11_on:
            The net growth rate of the dependent cells during therapy.

        w21_on:
            The rate at which dependent cells adapt during therapy.

        w22_on:
            The net growth rate of the adapted cells during therapy.

        w31_on:
            The rate at which dependent cells mutate during therapy.

        w32_on:
            The rate at which adapted cells mutate during therapy.

        w33_on:
            The net growth rate of the mutated cells during therapy.

        w11_off:
            The net growth rate of the dependent cells off therapy.

        w12_off:
            The rate at which adapted cells return to dependence off therapy.

        w22_off:
            The net growth rate of the adapted cells off therapy.

        w33_off:
            The net growth rate of the mutated cells off therapy.

        x1:
            The PSA per dependent cell.

        x2:
            The PSA per adapted cell.

        x3:
            The PSA per mutated cell.

    Examples:
        >>> from skcomplex.models.biology.oncology import PiecewiseLinearProstateElement
        >>> PiecewiseLinearProstateElement('tumour').u
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 17.84,
        AIR: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.778,
        AII: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.382,
        u: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        w11_on: Annotated[float, ParameterSpec()] = -0.043844,
        w21_on: Annotated[float, ParameterSpec()] = 0.00130716,
        w22_on: Annotated[float, ParameterSpec()] = -0.004282,
        w31_on: Annotated[float, ParameterSpec()] = 0.0,
        w32_on: Annotated[float, ParameterSpec()] = 0.0,
        w33_on: Annotated[float, ParameterSpec()] = 0.00181,
        w11_off: Annotated[float, ParameterSpec()] = 0.00223,
        w12_off: Annotated[float, ParameterSpec()] = 0.1,
        w22_off: Annotated[float, ParameterSpec()] = 0.00392,
        w33_off: Annotated[float, ParameterSpec()] = -0.2,
        x1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        x2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        x3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.AD = AD
        self.AIR = AIR
        self.AII = AII
        self.u = u
        self.w11_on = w11_on
        self.w21_on = w21_on
        self.w22_on = w22_on
        self.w31_on = w31_on
        self.w32_on = w32_on
        self.w33_on = w33_on
        self.w11_off = w11_off
        self.w12_off = w12_off
        self.w22_off = w22_off
        self.w33_off = w33_off
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the linear growth and transitions of each mode and the switching of the therapy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        on, off = s.u, 1 - s.u
        return [
            RateLaw(s.AD, on * s.w11_on * s.AD + off * (s.w11_off * s.AD + s.w12_off * s.AIR)),
            RateLaw(s.AIR, on * (s.w21_on * s.AD + s.w22_on * s.AIR) + off * s.w22_off * s.AIR),
            RateLaw(s.AII, on * (s.w31_on * s.AD + s.w32_on * s.AIR + s.w33_on * s.AII) + off * s.w33_off * s.AII),
        ]
