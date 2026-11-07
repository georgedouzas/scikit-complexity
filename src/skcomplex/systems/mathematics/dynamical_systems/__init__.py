"""Dynamical systems: chaotic and driven flows, maps, Koopman embeddings, and normal forms of bifurcations."""

from ._baker_map import BakerMapElement
from ._blow_up import BlowUpElement
from ._cat_map import CatMapElement
from ._chaotic_ensembles import cubic_chaotic_map, doubling_map, skew_tent_map
from ._circle_map import CircleMapElement
from ._conservative_henon_map import ConservativeHenonMapElement
from ._cubic_walk import CubicWalkElement
from ._cusp import CuspElement
from ._doubling_map import DoublingMapElement
from ._driven_flows import DrivenRosslerElement, ou_driven_rossler
from ._fermi_ulam_map import FermiUlamMapElement
from ._flows import (
    finite_time_blow_up,
    hyperchaotic_rossler,
    ikeda,
    langford,
    lorenz,
    nose_hoover,
    rossler,
    sprott_jerk,
    sprott_mixed_flow,
    thomas,
)
from ._gaussian_map import GaussianMapElement
from ._henon_map import HenonMapElement
from ._hopf import HopfElement
from ._hyperchaotic_rossler import HyperchaoticRosslerElement
from ._ikeda import IkedaElement
from ._intermittency_map import IntermittencyMapElement
from ._jerk import JerkElement
from ._koopman import SlowManifoldElement, koopman_slow_manifold
from ._langford import LangfordElement
from ._logistic_map import LogisticMapElement
from ._lozi_map import LoziMapElement
from ._maps import (
    baker_map,
    cat_map,
    cdn_map,
    circle_map,
    conservative_henon_map,
    fermi_ulam_map,
    henon_map,
    intermittency_map,
    logistic_map,
    lozi_map,
    newton_raphson_map,
    standard_map,
    tent_map,
    towel_map,
)
from ._mixed_flow import MixedFlowElement
from ._newton_map import NewtonMapElement
from ._normal_forms import cusp_form, hopf_form, saddle_node_form, takens_bogdanov_form
from ._nose_hoover import NoseHooverElement
from ._rossler import RosslerElement
from ._saddle_node import SaddleNodeElement
from ._skew_tent_map import SkewTentMapElement
from ._standard_map import StandardMapElement
from ._subcritical_forms import SubcriticalHopfElement, subcritical_hopf_form
from ._takens_bogdanov import TakensBogdanovElement
from ._tent_map import TentMapElement
from ._thomas import ThomasElement
from ._towel_map import TowelMapElement

__all__: list[str] = [
    'BakerMapElement',
    'BlowUpElement',
    'CatMapElement',
    'CircleMapElement',
    'ConservativeHenonMapElement',
    'CubicWalkElement',
    'CuspElement',
    'DoublingMapElement',
    'DrivenRosslerElement',
    'FermiUlamMapElement',
    'GaussianMapElement',
    'HenonMapElement',
    'HopfElement',
    'HyperchaoticRosslerElement',
    'IkedaElement',
    'IntermittencyMapElement',
    'JerkElement',
    'LangfordElement',
    'LogisticMapElement',
    'LoziMapElement',
    'MixedFlowElement',
    'NewtonMapElement',
    'NoseHooverElement',
    'RosslerElement',
    'SaddleNodeElement',
    'SkewTentMapElement',
    'SlowManifoldElement',
    'StandardMapElement',
    'SubcriticalHopfElement',
    'TakensBogdanovElement',
    'TentMapElement',
    'ThomasElement',
    'TowelMapElement',
    'baker_map',
    'cat_map',
    'cdn_map',
    'circle_map',
    'conservative_henon_map',
    'cubic_chaotic_map',
    'cusp_form',
    'doubling_map',
    'fermi_ulam_map',
    'finite_time_blow_up',
    'henon_map',
    'hopf_form',
    'hyperchaotic_rossler',
    'ikeda',
    'intermittency_map',
    'koopman_slow_manifold',
    'langford',
    'logistic_map',
    'lorenz',
    'lozi_map',
    'newton_raphson_map',
    'nose_hoover',
    'ou_driven_rossler',
    'rossler',
    'saddle_node_form',
    'skew_tent_map',
    'sprott_jerk',
    'sprott_mixed_flow',
    'standard_map',
    'subcritical_hopf_form',
    'takens_bogdanov_form',
    'tent_map',
    'thomas',
    'towel_map',
]
