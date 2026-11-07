# Spec Delta

## ADDED Requirements

### Requirement: Audited engineering and computer science systems
The library SHALL provide, as registered factories built from the general classes of the core, the engineering and
computer science systems of the reference audit that the core supports: `memristive_mixed_mode_oscillator` and
`coupled_colpitts` in `engineering/circuits`, `magnetic_levitation` and `vibrating_table` in
`engineering/mechatronics`, `minimum_time_control` in `engineering/control`, `grain_storage` in
`engineering/processes`, `light_emitting_robots` in `engineering/robotics`, and `fredkin_replicator`,
`turing_machine`, `small_world_life`, `dna_computation`, and `lenia` in `computer_science/automata`. The module of
every system SHALL define the elements, interactions, views, and helpers that only this system uses. Every system
SHALL have tests of a known result of its source, a gallery example, and an inventory entry with its sources and
sections.

#### Scenario: Published waveforms
- **WHEN** `memristive_mixed_mode_oscillator`, `magnetic_levitation`, or `vibrating_table` runs with its defaults
- **THEN** it settles on the mixed-mode oscillation `2^11` of period near 7.8, follows the closed-form periodic
  velocity, or vibrates steadily at the supply frequency close to its linearized model

#### Scenario: Control and synchronization
- **WHEN** `minimum_time_control` steers a free particle from `(3, 2)`, or `coupled_colpitts` runs with and without
  coupling
- **THEN** the particle reaches the origin near the time `2 + 2 sqrt(5)` and stays, or the phases drift apart without
  coupling and stay locked for `k = 0.4`

#### Scenario: Machines and automata
- **WHEN** `turing_machine`, `fredkin_replicator`, `small_world_life`, `dna_computation`, or `lenia` runs
- **THEN** the busy beaver halts after 13 steps with six ones, the pattern reappears four times at the powers of two,
  the unrewired network follows the Game of Life, the filtered strands are the only Hamiltonian path, and Orbium
  glides with a steady mass while one step follows equation 1

#### Scenario: Stocks and robots
- **WHEN** `grain_storage` runs, or `light_emitting_robots` runs with delays of 10 s and -10 s
- **THEN** the stock oscillates and settles at `6000 / (1 + 0.025 * 5)`, and the robots gather with the delay and
  spread with the prediction

### Requirement: Evolution by circumstance of RobbyGA
`robby_ga` SHALL take the regime `'circumstance'`, in which strategies live `lifespan` generations, the fraction
`starvation` with the lowest scores starves, and random survivors breed in the places of the dead, with an optional
period that switches the starvation on and off. The default regime SHALL keep the earlier behaviour.

#### Scenario: Starvation as selection
- **WHEN** `robby_ga` runs in the regime `'circumstance'` with a starvation of 0.3 and of 0
- **THEN** the mean score rises near zero with starvation and stays near the score of random strategies without it
