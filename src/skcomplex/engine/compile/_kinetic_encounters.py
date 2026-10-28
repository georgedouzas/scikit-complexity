"""Lower encounter laws to the gain and loss rates of densities on activity grids, as sparse tensor contractions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...core.laws import DestructionLaw, EncounterLaw, ProliferationLaw

_KINETIC_LAWS = (EncounterLaw, ProliferationLaw, DestructionLaw)
type KineticLaw = EncounterLaw | ProliferationLaw | DestructionLaw


