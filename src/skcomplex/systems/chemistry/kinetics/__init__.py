"""Chemical kinetics: reactions, oscillating reactors, and excitable media."""

from ._cyclic import CyclicCellElement, CyclicCellularAutomatonSystem
from ._hodgepodge import HodgepodgeCellElement, HodgepodgeSystem
from ._nanoparticles import NanoparticleCoronaSystem, SpeciesElement
from ._oscillators import (
    BrusselatorODESystem,
    CubicAutocatalysisElement,
    CubicAutocatalysisSystem,
    OregonatorElement,
    OregonatorSystem,
)
from ._reactions import (
    ChemicalLangevinEquationSystem,
    ChemicalLangevinNetworkSystem,
    MichaelisMentenSystem,
    SchloglElement,
)
from ._reactors import (
    CatalystPelletElement,
    CatalystPelletSystem,
    PulsedElectrochemicalReactorSystem,
    PulsedElectrodeElement,
)

__all__: list[str] = [
    'BrusselatorODESystem',
    'CatalystPelletElement',
    'CatalystPelletSystem',
    'ChemicalLangevinEquationSystem',
    'ChemicalLangevinNetworkSystem',
    'CubicAutocatalysisElement',
    'CubicAutocatalysisSystem',
    'CyclicCellElement',
    'CyclicCellularAutomatonSystem',
    'HodgepodgeCellElement',
    'HodgepodgeSystem',
    'MichaelisMentenSystem',
    'NanoparticleCoronaSystem',
    'OregonatorElement',
    'OregonatorSystem',
    'PulsedElectrochemicalReactorSystem',
    'PulsedElectrodeElement',
    'SchloglElement',
    'SpeciesElement',
]
