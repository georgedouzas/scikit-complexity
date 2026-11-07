# Design

## Context

The audit judged each listed model supported by the current core. Discrete agent models are agent rules over groups
(`RuleLaw` interactions and `MapInteraction` in the agent form), lattice games are stochastic local rules, the
system traps are transfers between stocks, and the multilevel selection densities are finite-volume fields with an
integral transform, as in the existing `nonlinear_pairwise_group_conflict`.

## Goals / Non-Goals

**Goals:**
- Follow the rules, parameters, and defaults of the cited sections, and test a known result of each.
- Reuse existing modules, helpers, and systems, and add an option where the source presents a variant.

**Non-Goals:**
- Core changes. Models that need one, or that the sources do not specify, are left out and listed below.

## Decisions

- **Additive pairwise conflict is an option of the two-level replicator.** With an additively separable victory
  probability `rho = 1/2 + (G(z) - G(u)) / 2`, the pairwise group conflict of Cooney section 5 reduces to the
  two-level replicator equation of Eq. 3.3 (Eq. 3.11), so both audit rows are `two_level_replicator`, whose `victory`
  chooses `G(z) = z` or the normalized average payoff.
- **The Big Sort is an option of Schelling.** `schelling(choices=k)` moves a share `moving` of the residents every
  step to the best of `k` random empty cells, in the existing `RelocationInteraction`.
- **The three-state opinion model is two systems.** The network of agents and the mean-field flows between the
  numbers against, undecided, and for have different components, so `three_state_opinion` and `undecided_flows` are
  separate systems. The flows driven by the undecided are the option `driven` of the latter.
- **Mobility lives in `social/organizations`.** The audit proposed a new subfield `social/mobility`, which would be a
  new package, so `exploration_preferential_return` and `container_mobility` join the other movement systems of
  organizations until the user decides on the subfield.
- **Ajani's Choice follows the rules of the text.** The NetLogo listing accepts a low offer when yesterday's pay was
  above the minimum wage, against rule 3 of section 3.3, and the system follows the text. The two slums and the areas
  of the farms are states, since the rules only use which area a farm belongs to.
- **Namrata's Family reads the power condition as flows.** `V_ij < V_ji` with the same `w_ji` is
  `w_->i > w_j->`, so the firms hold their inflow and outflow and the adoption is a synchronous agent map on the
  directed supply network.
- **The kinetic crowd is a particle method.** As in Bellomo section 6.5, the kinetic equation is solved by a direct
  simulation Monte Carlo, with one particle per walker and densities counted on cells, the decision process of
  Table 6.2, and the wall rule of Eq. 6.25. The homogeneous case reproduces the fundamental diagram of Eq. 6.30.
- **The echo chamber integrates in steps of ten Euler steps.** The temporal network is redrawn every integration step
  of `dt = 0.01`, as in the paper, and a step of the system groups ten of them so default runs stay fast.
- **Narrative traps get documented rules.** Meadows describes policy resistance, drift to low performance, and the
  watchmakers in prose, so the systems write the structure she describes with documented corrective rules and
  parameters, and say so in their docstrings.

## Left out

- `hoping_for_arrest` (Chesney 5.2 to 5.8): the criminal network is the Caviar data set, which the references do not
  hold, and the text gives only the order of the weekly events and the logistic pressure. How police investigate and
  rescue, how pressure travels along the police network, and how criminals pass girls along their network are not
  specified, so the system would be invented rather than implemented.
- Louf and Barthelemy's congestion model of cities (arXiv 2404.02642 section 4): the review names the model and its
  premise only, and its equations are in the cited works, which the references do not hold.
