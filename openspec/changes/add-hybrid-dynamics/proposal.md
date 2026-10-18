# Proposal

## Why

Many models in the references depend on time in ways the current law kinds cannot express:
- **Delays:**
  - Mackey-Glass (Dilão 2023 §6.3).
  - Delayed Kuramoto and car following (Gros 2015 §2.5, §4.4, §9.3).
  - Sensory-delay robots and delayed flocking (Argun et al. 2021 ch. 8, 10).
  - System-dynamics material and information delays (Bala et al. 2017 §4.5, §4.10; Meadows 2008).
- **Events:** impacts in Newton's cradle and the ballistic pendulum (Ryzhov et al. 2021), billiards (Datseris and
  Parlitz 2022 ch. 8), stick-slip earthquakes (Charbonneau 2017 ch. 8), and neuron resets (Trzaska 2023 ch. 5).
- **Drivers:** STEP, RAMP, and PULSE test inputs (Bala et al. 2017 §4.9), periodic and impulsive forcing (Trzaska 2023
  ch. 2), and rate-induced tipping by parameter ramps (Datseris and Parlitz 2022 ch. 12).
- **Noise conventions:** multiplicative noise needs Stratonovich or anti-Itô handling to avoid biased results (Argun et
  al. 2021 ch. 7).
- **Poincaré sections:** these require locating crossings (Datseris and Parlitz 2022 §1.6).

## What Changes

- Add delays: laws read a state at `t - τ` for constant or state-dependent delays. Initial histories are given as
  functions or constants, and solvers integrate by the method of steps with dense interpolation.
- Add events: a guard function crossing zero in a given direction triggers a reset of states. Events can terminate the
  run. Solvers locate crossings by root finding, and events are recorded in trajectories.
- Add drivers: step, ramp, pulse, pulse train, sine, and interpolated data series as time functions inside laws, and
  scheduled parameter ramps.
- Add stochastic conventions: Itô, Stratonovich, and anti-Itô noise, with the drift correction derived symbolically,
  plus Milstein and stochastic Heun steppers.
- Add Poincaré sections and stroboscopic maps from events.
- Add library models: Mackey-Glass, Hutchinson's delayed logistic, delayed Kuramoto, a bouncing ball, Newton's cradle,
  a circular billiard, integrate-and-fire and Izhikevich neurons, the forced Duffing oscillator, stochastic resonance,
  and a Brownian particle near a wall with multiplicative noise.

## Capabilities

### New Capabilities

- `hybrid-dynamics`: delays, events, drivers, stochastic conventions, and Poincaré sections.

### Modified Capabilities

- `model-library`: delay, event, and forced models.

## Impact

- `skcomplex.model` gains `delayed`, `Event`, `step`, `ramp`, `pulse`, `pulse_train`, and `series`, and a `noise`
  convention on systems.
- Solvers gain a history buffer, event location, the method of steps, and the Milstein and Heun steppers.
- Depends on `add-general-model-interface`, and `add-analysis-toolkit` adds Poincaré sections to its dynamics tools.
