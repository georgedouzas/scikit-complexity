"""Yeast glycolysis in a flow reactor, the fifteen metabolites of the model of Nielsen and coworkers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import DeclaredElement
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec

CONCENTRATIONS = MappingProxyType(
    {
        'ATP': 4.49064,
        'ADP': 0.108367,
        'AMP': 0.00261149,
        'GLC': 0.0112817,
        'F6P': 0.65939,
        'FBP': 0.00770135,
        'GAP': 0.00190919,
        'NAD': 3.62057,
        'NADH': 0.616118,
        'DPG': 0.299109,
        'PEP': 0.0021125,
        'PYR': 0.00422702,
        'ACA': 0.0738334,
        'EtOH': 0.33981,
        'P': 0.0,
    },
)
CONSTANTS = MappingProxyType(
    {
        'V1': 0.5,
        'K1GLC': 0.1,
        'K1ATP': 0.063,
        'V2': 1.5,
        'K2': 0.0016,
        'k2': 0.017,
        'K2ATP': 0.01,
        'k3f': 1.0,
        'k3b': 50.0,
        'V4': 10.0,
        'K4GAP': 1.0,
        'K4NAD': 1.0,
        'k5f': 1.0,
        'k5b': 0.5,
        'V6': 10.0,
        'K6PEP': 0.2,
        'K6ADP': 0.3,
        'V7': 2.0,
        'K7PYR': 0.3,
        'k8f': 1.0,
        'k8b': 0.000143,
        'k9f': 10.0,
        'k9b': 10.0,
        'k10': 0.05,
        'flow': 0.011,
    },
)
FEEDS = MappingProxyType({'ATP': 3.5, 'ADP': 1.1, 'NADH': 0.24, 'NAD': 4.0, 'GLC': 50.0})


class GlycolysisElement(DeclaredElement):
    """Yeast glycolysis in a flow reactor, the fifteen metabolites of the model of Nielsen and coworkers.

    The model of Nielsen, Sørensen, Hynne, and Busse (1998), BioModels BIOMD0000000042, follows glucose `GLC` through
    hexokinase, phosphofructokinase with its inhibition by the ratio of `ATP` to `AMP`, aldolase, glyceraldehyde
    phosphate dehydrogenase, phosphoglycerate kinase, pyruvate kinase, pyruvate decarboxylase, alcohol dehydrogenase,
    adenylate kinase, and a glycerol branch to `P`. The reactor is fed with `ATP`, `ADP`, `NADH`, `NAD`, and glucose
    at fixed concentrations and washed out at the rate `flow`. The concentrations, in mM, and the rate constants, in mM
    and per minute, are declared variables named as in the source, and `variables` replaces any of them.

    Args:
        label:
            The label of the cell extract.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        variables:
            The declared states and parameters that replace those of the source, by name, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import GlycolysisElement
        >>> GlycolysisElement('extract').get_value('V4')
        10.0
    """

    def _declare_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the concentrations and the rate constants of the source, updated by `variables`."""
        nonnegative = RealSet.closed_open(0, oo)
        declared = {
            name: DeclaredVariableSpec(value, StateSpec(domain=nonnegative)) for name, value in CONCENTRATIONS.items()
        }
        declared |= {
            name: DeclaredVariableSpec(value, ParameterSpec(domain=nonnegative)) for name, value in CONSTANTS.items()
        }
        return {**declared, **(self.variables or {})}

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the metabolites, the reactions of glycolysis and the exchange with the feed.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        v1 = s.V1 * s.ATP * s.GLC / ((s.K1GLC + s.GLC) * (s.K1ATP + s.ATP))
        inhibition = s.K2 * (1 + s.k2 * (s.ATP / s.AMP) ** 2)
        v2 = s.V2 * s.ATP * s.F6P**2 / ((inhibition + s.F6P**2) * (s.K2ATP + s.ATP))
        v3 = s.k3f * s.FBP - s.k3b * s.GAP**2
        v4 = s.V4 * s.NAD * s.GAP / ((s.K4GAP + s.GAP) * (s.K4NAD + s.NAD))
        v5 = s.k5f * s.DPG * s.ADP - s.k5b * s.PEP * s.ATP
        v6 = s.V6 * s.ADP * s.PEP / ((s.K6PEP + s.PEP) * (s.K6ADP + s.ADP))
        v7 = s.V7 * s.PYR / (s.K7PYR + s.PYR)
        v8 = s.k8f * s.NADH * s.ACA - s.k8b * s.NAD * s.EtOH
        v9 = s.k9f * s.AMP * s.ATP - s.k9b * s.ADP**2
        v10 = s.k10 * s.F6P
        reactions = {
            'ATP': -v1 - v2 + v5 + v6 - v9,
            'ADP': v1 + v2 - v5 - v6 + 2 * v9,
            'AMP': -v9,
            'GLC': -v1,
            'F6P': v1 - v2 - v10,
            'FBP': v2 - v3,
            'GAP': 2 * v3 - v4,
            'NAD': -v4 + v8,
            'NADH': v4 - v8,
            'DPG': v4 - v5,
            'PEP': v5 - v6,
            'PYR': v6 - v7,
            'ACA': v7 - v8,
            'EtOH': v8,
            'P': v10,
        }
        laws: list[BaseLaw] = []
        for name, rate in reactions.items():
            state = getattr(s, name)
            exchange = s.flow * (FEEDS.get(name, 0.0) - state)
            laws.append(RateLaw(state, rate + exchange))
        return laws
