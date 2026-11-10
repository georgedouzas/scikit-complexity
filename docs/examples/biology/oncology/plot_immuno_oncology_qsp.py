"""
# Immuno-oncology QSP with virtual patients

A reduced quantitative systems pharmacology model of a tumour, effector and regulatory T cells, and an anti-PD-1
antibody. Virtual patients are sampled and kept when their tumours are plausible, matched to observed ratios of T
cells, and one patient is checked for identifiability and calibrated as a digital twin.
"""

# %%
# The default patient has a tumour of 2.7 cm and receives 240 mg of antibody every two weeks. The antibody enters the
# central compartment over one day, exchanges with a peripheral one, and occupies PD-1, which frees the effector T
# cells to kill. The run lasts 180 days, and the observer saves every day with the default blueprint.

import functools

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import calibrate_individuals, identifiability, virtual_population
from skcomplex.systems.biology.oncology import ImmunoOncologyQspSystem

SYSTEM = 'immuno_oncology_qsp'
BASELINE = 60.0
BOUNDS = {
    'growth': (0.005, 0.1),
    'kill': (1e-11, 5e-10),
    'recruitment': (0.002, 0.05),
    'treg_recruitment': (0.0005, 0.01),
}
FITTED = {'growth': (0.005, 0.1), 'kill': (1e-11, 5e-10)}

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(ImmunoOncologyQspSystem(), observers=[observer])
untreated = simulate(ImmunoOncologyQspSystem(dose=0.0))

# %%
# Treatment and the antibody
# --------------------------
# Without the antibody the tumour should keep growing. With it, the central antibody should rise with every dose and
# settle into a cycle, and the tumour should shrink.

fig, (tumour, antibody) = plt.subplots(1, 2, figsize=(10, 4))
tumour.plot(trajectory.times, trajectory.get_values('diameter'), label='240 mg every two weeks')
tumour.plot(untreated.times, untreated.get_values('diameter'), label='no treatment')
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour diameter (cm)')
tumour.set_title('Treatment shrinks the tumour')
tumour.legend()
antibody.plot(trajectory.times, trajectory.get_values('central.x') / 4.0, color='tab:purple')
antibody.set_xlabel('time (days)')
antibody.set_ylabel('central antibody (mg/L)')
antibody.set_title('Two-compartment kinetics of the doses')
fig.tight_layout()
plt.show()

# %%
# A virtual population
# --------------------
# Virtual patients start from a tumour of 0.8 cm, and treatment starts at day 60, the baseline. The growth, the
# killing, and the recruitment of both kinds of T cells are drawn uniformly. A patient is plausible when the baseline
# diameter lies between 1 and 10 cm. The observed patients have log ratios of effector to regulatory T cells around
# 1.5, and selection by the probability of inclusion should give a population whose log ratios follow them.


def log_ratio(run):
    """Return the log ratio of effector to regulatory T cells at the end of a run."""
    return float(np.log(run.get_values('effector_treg_ratio')[-1]))


patient = functools.partial(ImmunoOncologyQspSystem, cancer=1e8, effector=1e6, treg=1e6, start=BASELINE)
observed = np.random.default_rng(1).normal(1.5, 0.4, 100)
population = virtual_population(
    patient,
    BOUNDS,
    ['diameter', log_ratio],
    {'diameter': (1.0, 10.0)},
    n=80,
    target={'log_ratio': observed},
    until=BASELINE,
    random_state=0,
)
fig, (sizes, ratios) = plt.subplots(1, 2, figsize=(10, 4))
sizes.hist(population.plausible['diameter'], bins=20, color='tab:blue', alpha=0.7)
sizes.set_xlabel('baseline diameter (cm)')
sizes.set_ylabel('plausible patients')
sizes.set_title(f'Acceptance {population.acceptance:.0%}')
bins = np.linspace(-1.0, 5.0, 31)
ratios.hist(population.plausible['log_ratio'], bins=bins, density=True, alpha=0.4, label='plausible')
ratios.hist(population.selected['log_ratio'], bins=bins, density=True, alpha=0.6, label='selected')
ratios.hist(observed, bins=bins, density=True, histtype='step', color='black', label='observed')
ratios.set_xlabel('log ratio of effector to regulatory T cells')
ratios.set_title('Selection by the probability of inclusion')
ratios.legend()
fig.tight_layout()
plt.show()

# %%
# Identifiability of one patient
# ------------------------------
# The first selected patient is treated from day 60, and its tumour diameter is measured every week for 180 days with
# 2% noise. The profile of the killing, with the other parameters known, should close inside its bounds, since the
# treated weeks show how fast the T cells kill. A profile of several parameters fits the others again at every point,
# so it costs far more runs.

chosen = population.selected.iloc[0]
truth = {name: float(chosen[name]) for name in BOUNDS}
known = {name: value for name, value in truth.items() if name not in FITTED}
days = np.arange(0.0, 181.0, 7.0)
rng = np.random.default_rng(2)
diameters = simulate(patient(**truth), until=180.0, dt=1.0).get_values('diameter')
noisy = diameters[days.astype(int)] * (1 + 0.02 * rng.standard_normal(days.size))
data = {'diameter': (days, noisy)}
profile = identifiability(
    patient(growth=truth['growth'], **known),
    data,
    {'kill': FITTED['kill']},
    n_starts=1,
    random_state=0,
)
values, ratio = profile.profiles['kill']
order = np.argsort(values)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(values[order], ratio[order], marker='.')
ax.axhline(np.exp(-3.84 / 2), color='grey', linestyle=':', label='95% threshold')
ax.axvline(truth['kill'], color='black', linestyle='--', label='truth')
ax.set_xlabel('kill (per cell per day)')
ax.set_ylabel('likelihood ratio')
ax.set_title(f"The killing is {profile.verdicts['kill']}")
ax.legend()
plt.show()

# %%
# A digital twin
# --------------
# The twin is the patient's own calibration of the growth and the killing, with the recruitment of both kinds of T cells
# known and passed to the factory of patients. Its tumour should follow the measured diameters, and its estimates
# should lie near the truth.

twins = calibrate_individuals(
    patient,
    {'patient': data},
    FITTED,
    known={'patient': known},
    random_state=0,
)
fitted = simulate(twins.calibrations['patient'].system, until=180.0, dt=1.0)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(days, noisy, 'o', color='black', label='measured')
ax.plot(fitted.times, fitted.get_values('diameter'), color='tab:green', label='digital twin')
ax.axvline(BASELINE, color='grey', linestyle=':', label='first dose')
ax.set_xlabel('time (days)')
ax.set_ylabel('tumour diameter (cm)')
estimates = twins.estimates.loc['patient']
ax.set_title(f"Twin growth {estimates['growth']:.3f} (truth {truth['growth']:.3f})")
ax.legend()
plt.show()
