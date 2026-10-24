# Spec Delta

## MODIFIED Requirements

### Requirement: Examples show analysis plots from the simulation
Every curated example SHALL show at least one Matplotlib figure computed from its own run, from the trajectory, the
observables, or the agents of the simulation, that exhibits the behaviour its model is known for. Each figure SHALL
have labelled axes and a title, and the text before it SHALL state the agreement the reader should see. The analysis
SHALL use the functions of `skcomplex.engine.analysis` where they fit, and NumPy or SciPy otherwise.

#### Scenario: Ideal gas speeds
- **WHEN** a user opens the ideal gas example page
- **THEN** below the Rerun screenshot it shows the histogram of the simulated molecular speeds at the end of the run
  against the Maxwell-Boltzmann speed distribution, in the dimension of the model, at the temperature measured from the
  kinetic energy of the simulated molecules

#### Scenario: Thumbnail stays the Rerun screenshot
- **WHEN** an example draws analysis figures
- **THEN** its gallery thumbnail is still its Rerun screenshot, and the figures follow the screenshot on its page
