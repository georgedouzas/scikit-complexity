## ADDED Requirements

### Requirement: Immuno-oncology QSP system
The library SHALL provide `ImmunoOncologyQspSystem` in `biology/oncology`, a reduced model of the QSP-IO platform that
Wang et al. (2024, arXiv 2403.03335) review: cancer cells killed by effector T cells, regulatory T cells that suppress
them, PD-1 occupancy from an anti-PD-1 antibody with two-compartment kinetics, and doses given by a schedule. It SHALL
declare the tumour diameter and the ratio of effector to regulatory T cells as observables.

#### Scenario: Treatment shrinks the tumour
- **WHEN** the system runs 180 days with a dose every two weeks and again without doses, from the same state
- **THEN** the final tumour diameter with doses is smaller than without them

#### Scenario: Antibody kinetics
- **WHEN** one dose is given and the T cells are held at zero
- **THEN** the central antibody concentration decays with the two exponential phases of the two-compartment kinetics

### Requirement: The QSP system is recorded in the system inventory
The inventory SHALL record `immuno_oncology_qsp` as implemented with the source 2403.03335 and its primitives, and the
gallery SHALL hold its example, which builds a virtual population and calibrates one virtual patient.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for `immuno_oncology_qsp`
