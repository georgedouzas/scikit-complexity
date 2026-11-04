# Design

## Context

The source check of `add-constraint-based-models` recorded the numbers in the proposal. Raman's step (ii) diffuses metabolites
only and alternates with growth; Harcombe also diffuses biomass, at 3e-9 cm²/s, or 3e-10 for E. coli colonies, with
metabolites at 5e-6 cm²/s, steps of 0.1 h, and 3e-7 g at the centre of a 50 by 50 lattice. Only summaries of Harcombe were
read, so the first task reads the full paper if it can be had, and otherwise states that the values come through a summary.

## Goals / Non-Goals

**Goals:** the published scheme and values, within the gallery budget.

**Non-Goals:** a general operator splitting for every law, or changes to `dynamic_fba`.

## Decisions

### Reuse before adding

`IMEXEulerSolver`, a `SplitSolver`, is the alternation, and no solver is added. Its step `(I - h L) x' = x + h N(x)` is the
growth step of Harcombe et al. in `x + h N(x)`, with the program in the remainder `N`, and then one implicit diffusion solve.
`SplitSolver` needed `LinearProgramLaw` in its support. `StrangSplittingSolver` does not fit, because it splits the flux
differences of finite-volume fields and raises an error for a problem without flux.

The compiler treated `dx/dt = v x` as a linear term with the coefficient `v`, a state of another element, and the solver read
that coefficient as a parameter. The growth came out 100 times too small. `_linear_terms` now leaves out every coefficient
that a state gives, as its docstring states.

Harcombe et al. use an alternating direction implicit scheme. Implicit Euler replaces it, because at 4.5 boxes^2 per step the
amplification factor of Peaceman-Rachford is negative for the fine modes, so it does not keep glucose non-negative, while
`I - h L` is an M-matrix and does.

### Sources

Harcombe et al. (2014) was read in full from the open copy PMC4097880, the main text with Table 1 and the Experimental
Procedures. No value comes through a summary. The Supplemental Experimental Procedures were not read, because the PMC file sits
behind a proof-of-work page.

| Value | Default | Source |
| --- | --- | --- |
| Metabolite diffusion | 5e-6 cm^2/s | Harcombe, Table 1 |
| Biomass diffusion | 3e-10 cm^2/s | Harcombe, Table 1 footnote, the colony runs |
| Box, lattice | 0.02 cm, 50 by 50 | Harcombe, Colony expansion comparisons |
| Step | 0.1 h | Harcombe, In silico experiments |
| Inoculum | 3e-7 g at the centre | Harcombe, Colony expansion comparisons |
| Colony level | 1e-7 g per box | Harcombe, Colony expansion comparisons, used by tests and the example |
| Glucose | 5 g/L as 0.0088 mmol/cm^2 | Harcombe, In silico experiments |
| Vmax, Km | 10 mmol/g/h, 0.01 mM | Harcombe, Table 1 |
| Interval of the uptake bound | the step | Raman 2021, Section 9.6 |
| Volume under a box | 1.27e-7 L | No source: glucose per box over its concentration, the text defers the volume to its supplement |
| Molar mass of glucose | 180.16 g/mol | No source in these texts |
| Periodic edges | | No source: the text does not state the outer boundary |
| Length of the run | 14 h | No source: keeps a run within the gallery budget |

The death rate, the colony height, and the oxygen of Harcombe et al. are outside this change and are left out, and the network
is the E. coli core model instead of iJO1366.

### Sites without a solution

Without the old threshold a run reaches sites whose glucose cannot pay for the maintenance flux of the network, whose programs
have no solution. Those sites take zero fluxes through `infeasible='zero'`. The activity remains at a biomass of zero, which
is no parameter, so a site without biomass has zero fluxes.

### Biomass diffusion stays

The spec keeps biomass diffusion, which Harcombe has and Raman's short description leaves out. The docstring says so.

## Risks / Trade-offs

- [The front moves about 0.1 box/h at the published values, so a short run shows little] → The gallery example runs long
  enough to show it, or a smaller lattice, and states which; if the budget cannot hold, the change pauses for a decision.
  The front moves at about 0.18 box/h on the lattice, because the front is thinner than a box and the estimate of
  `2 sqrt(D mu)` holds on a continuum. The example keeps the published lattice for 14 h, in 141 frames of the run and 36 logged
  frames.
