# Tasks

## 1. Flows and conservation

- [x] 1.1 Add quantity tags, `Flow`, `Source`, and `Sink` for rate, jump, and update forms, and verify the bathtub
  scenario
- [x] 1.2 Add the symbolic conservation check, the run-time monitor, and `flow_matrix`, and verify the undeclared change
  scenario

## 2. System-dynamics blocks

- [x] 2.1 Add `Auxiliary` and `lookup` on NumPy and Numba, and verify the table function scenario
- [x] 2.2 Add `delay1`, `delay3`, `smooth`, and `smooth3`, and verify the third-order delay scenario

## 3. Loops

- [x] 3.1 Add `loops` and `loop_dominance`, and verify the population loops scenario

## 4. Ledgers

- [x] 4.1 Add `transfer`, `Identity`, and settlement on removal for agents, and verify the identity violation and
  bankruptcy scenarios, and the closed economy scenario

## 5. Library

- [x] 5.1 Add the models that probe the primitives in distinct ways as registered factories with gallery examples,
  rebuild `bathtub` and `car_dealer` on flows, and verify the inventory scenario and the conformance suite
- [x] 5.2 Add `energy_return`, `cocoa`, `hilsa_fishery`, and `food_security`, which repeat patterns the implemented
  models already probe, and rebuild `oil_economy` and `fishery_capital`, Meadows' nonrenewable and renewable
  resources, on flows in place of the duplicate entries `renewable_resource` and `nonrenewable_resource`

## 6. Documentation and gate

- [x] 6.1 Document flows, quantities, conservation, system-dynamics blocks, loops, and ledgers in the user guide, and
  verify the documentation test passes
- [ ] 6.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
