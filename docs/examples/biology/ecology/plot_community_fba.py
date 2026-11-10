"""
# Community flux balance analysis

A community can do what its members cannot. One mutant of E. coli core lacks isocitrate dehydrogenase, and the other
lacks ammonium transport, so neither grows alone on glucose. Joined through a shared medium, each takes up what the
other secretes, 2-oxoglutarate one way and glutamate the other, and both grow.
"""

# %%
# The example builds `CommunityFBASystem`, which needs the extra `metabolism`. The programs of the members merge through
# the medium, whose law holds the growth weighted by the abundances. The primary view is the bar chart of the fluxes
# of the first member.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.common.metabolism import community_growth
from skcomplex.systems.biology.ecology import CommunityFBASystem

SYSTEM = 'community_fba'
GROWTH = 'v_Biomass_Ecoli_core'

system = CommunityFBASystem().prepare().system_
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(system, observers=[observer], until=0.0)

# %%
# Alone and together
# ------------------
# Each member alone should not grow, and together both should. Community FBA, where both grow at one rate, should grow
# no faster than joint FBA, where each has its own. SteadyCom should find a common rate with abundances that
# sum to one, and MICOM's trade-off should keep the growth of the community.

alone = [
    float(
        simulate(CommunityFBASystem(knockouts=(name,), abundances=(1.0,))).get_values('medium.objective_0')[0],
    )
    for name in ('ICDHyr', 'NH4t')
]
together = [float(trajectory.get_values(f'member_{index}.{GROWTH}')[0]) for index in range(2)]
community = float(trajectory.get_values('medium.objective_0')[0])
balanced = simulate(CommunityFBASystem(common_rate=True))
balanced_rate = float(balanced.get_values('medium.objective_0')[0])
steadycom = community_growth(system, medium={'EX_akg_e': 0.0, 'EX_glu__L_e': 0.0})
tradeoff = community_growth(CommunityFBASystem(knockouts=('', '')).prepare().system_, kind='tradeoff', fraction=1.0)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['ΔICDHyr alone', 'ΔNH4t alone', 'ΔICDHyr together', 'ΔNH4t together'], [*alone, *together])
axes[0].tick_params(axis='x', rotation=20)
axes[0].set_ylabel('growth (1/h)')
axes[1].bar(
    ['joint FBA', 'community FBA', 'SteadyCom', 'wild-type trade-off'],
    [community, balanced_rate, steadycom.growth, tradeoff.growth],
)
axes[1].set_ylabel('community growth (1/h)')
fig.suptitle(f'Alone {max(alone):.3f}, together {community:.3f} 1/h; SteadyCom common rate {steadycom.growth:.3f} 1/h')
fig.tight_layout()
plt.show()
