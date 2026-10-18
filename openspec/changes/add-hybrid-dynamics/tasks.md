# Tasks

## 1. Delays

- [ ] 1.1 Add `delayed`, histories, and the history buffer with the method of steps for the SciPy and fixed-step
  solvers, and verify the Mackey-Glass and delayed logistic scenarios

## 2. Events and drivers

- [ ] 2.1 Add `Event` with located crossings, resets, terminal events, and event records, and verify the bouncing ball
  and terminal event scenarios
- [ ] 2.2 Add the drivers with breakpoints and scheduled ramps, and verify the step response scenario

## 3. Noise

- [ ] 3.1 Add noise conventions with symbolic drift correction and the Milstein and Heun steppers, and verify the
  Stratonovich scenario

## 4. Sections and library

- [ ] 4.1 Add `poincare_section` and `stroboscopic`, and verify the Duffing scenario
- [ ] 4.2 Add the delay, event, and forced models as registered factories, and verify the Izhikevich scenario and the
  conformance suite

## 5. Documentation and gate

- [ ] 5.1 Document delays, events, drivers, and noise conventions in the user guide, and verify the documentation test
  passes
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
