# Tasks

## 1. Flows and conservation

- [ ] 1.1 Add quantity tags, `Flow`, `Source`, and `Sink` for rate, jump, and update forms, and verify the bathtub
  scenario
- [ ] 1.2 Add the symbolic conservation check, the run-time monitor, and `flow_matrix`, and verify the undeclared change
  scenario

## 2. System-dynamics blocks

- [ ] 2.1 Add `Auxiliary` and `lookup` on NumPy and Numba, and verify the table function scenario
- [ ] 2.2 Add `delay1`, `delay3`, `smooth`, and `smooth3`, and verify the third-order delay scenario

## 3. Loops

- [ ] 3.1 Add `loops` and `loop_dominance`, and verify the population loops scenario

## 4. Ledgers

- [ ] 4.1 Add `Transfer`, `Identity`, and settlement on removal for agents, and verify the identity violation and
  bankruptcy scenarios, and the closed economy scenario

## 5. Library

- [ ] 5.1 Add the systems zoo, the traps, Bass diffusion, the beer game, and the monetary economy as registered
  factories, and verify the inventory scenario and the conformance suite

## 6. Documentation and gate

- [ ] 6.1 Document flows, quantities, conservation, system-dynamics blocks, loops, and ledgers in the user guide, and
  verify the documentation test passes
- [ ] 6.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
