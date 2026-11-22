## Context

`social/cooperation` holds spatial games, hypergraph games, multilevel selection, and two-level birth-death processes.
The core offers `NetworkSpace` with layers given as `LayerSpec`, conditions with `where`, members on the edges of a
network with `Element(..., space=network, on='edges')`, lookup tables, reductions over neighbours, the bindings of
`Interaction`, such as `along='<layer>'` for the edges of one layer and `along='neighborhoods'` for every member with
the members around it, phases of agent rules, draws such as `categorical`, and `ensemble` for independent runs. A Markov
chain of a discrete value is an `UpdateLaw` with a `categorical` draw, or a `JumpLaw` in continuous time, since the
interactions that only wrapped one law are gone.

Every system follows the current API. An element holds its values, an interaction owns the parameters of the relation
and sets its laws in `laws_` from `_prepare`, a ready-made system sets `system_` from `_prepare`, and a run is
`system.prepare().simulate(until=...)`.

## Goals / Non-Goals

**Goals:**
- Probes that use new combinations of existing primitives, each checked against an analytic result of its paper.
- No change to the core unless a probe proves a primitive missing.

**Non-Goals:**
- The general replicator equation of any `n`-strategy game of 2401.11686 as an analysis. Its two cases used here are
  short formulas in the test of the system.
- Gossip groups with private views per individual, the institution variant, and the insular interactions of
  2204.10811. The group views cover its main result.
- The donor game of 2412.10270. It joins the language-model probes in a later change.

## Decisions

### Placement and the primitives each probe exercises

| Component | Placement step | Where | Primitives it exercises |
| --- | --- | --- | --- |
| `DynamicNetworkDilemmaSystem`, `critical_benefit_cost_ratio` | 3, one system | `social/cooperation/_dynamic_network_dilemma.py` | `networks` with one layer per structure, a structure value switched by `transitions`, neighbour reductions chosen by `where` on that value, `discrete-states`, and `ensemble` |
| `GossipGroupNormsSystem` | 3, one system | `social/cooperation/_gossip_group_norms.py` | `network-generators` (views as values of members on the edges from groups to individuals), `numeric-functions` (norms as lookup tables), `draws` (errors), `phases` (donation, assessment, migration) |
| `GraphPunishmentGameSystem` | 3, one system | `social/cooperation/_graph_punishment_game.py` | `neighbour-hyperedges` on random regular graphs from `network-generators`, `reductions` over every group a node belongs to, three `discrete-states`, `draws` |

A component that two of these modules share, such as the payoff of a donation round, stays in each module, as the rule
for one system asks. A family `social/common/games` would be a new package, which needs the maintainer's approval
first.

### Dynamic network dilemma

- `DynamicNetworkDilemmaSystem(networks=None, transitions=None, b=..., c=1.0, delta=0.01, random_state=None)`.
  `networks` of `None` builds the pair of Figure 2a of the paper with `N` nodes and community size `a`: a star and a
  complete graph joined by one edge, swapped between the two networks. `transitions` gives the probability per update of
  moving between networks, from the mean durations `t1 N` and `t2 N`.
- The network is one `NetworkSpace` with one `LayerSpec` per structure. The value `structure` of an element of the
  system is a discrete value switched by an interaction whose `_prepare` sets an `UpdateLaw` with a `categorical` draw
  from the row of the transition matrix. Strategies follow death-birth updates, a rule of the whole system, with
  fitness `1 + delta * payoff`, where the payoff sums the donation game over the neighbours of the current layer, read
  by reductions over each layer chosen by `where` on `structure`.
- `critical_benefit_cost_ratio(networks, transitions)` solves the linear system of Equation 3 of the paper for the
  coalescence quantities and returns `(b/c)*`, positive when cooperation can be favoured, negative when spite is.

### Gossip group norms

- Individuals are a group with a strategy (discerning, as in the paper) and a gossip group label. Gossip groups are a
  group of `K` members, and every edge from a gossip group to an individual holds the view, 0 or 1, as a value of
  `Element('views', values={'view': 1}, space=network, on='edges')`, which interactions read through `along=`.
- Each round every donor meets every recipient, gives with cost `c` and benefit `b` when its group views the recipient
  as good, and errs with probability `u_x`. Every gossip group observes one random interaction of each donor and
  assesses it with its norm table, erring with probability `u_a`. Group sizes follow the replicator equation of the
  paper on the mean payoffs of their members, applied by migration of individuals at a given rate.
- Norm tables, indexed by the action and the view of the recipient: Stern Judging (good for C with good and D with
  bad), Simple Standing (good unless D with good), Scoring (good after C), and Shunning (good only for C with good).

### Graph punishment game

- `GraphPunishmentGameSystem(n=1000, k=4, r=3.0, c=1.0, alpha=0.3, beta=1.0, intensity=0.01, random_state=None)`. The
  graph is `networkx.random_regular_graph(k, n)` in a `NetworkSpace`. Every node forms the group of itself and its `k`
  neighbours, bound by `along='neighborhoods'`, so each node plays `k + 1` games. Punishers cooperate and pay `alpha`
  per defector in a group, and defectors pay `beta` per
  punisher. Updates follow pairwise comparison with intensity of selection `intensity`.
- The scenarios compare simulated trends with the predictions of the paper, `r > k + 1` for cooperation and the
  threshold `beta*` of equation 6, over enough seeds to overcome noise, and are marked slow.

## Risks / Trade-offs

- [Weak selection needs many runs to see fixation differences] → The fixation scenario uses a small network and many
  seeds through `ensemble` with a parallel executor, and is marked slow.
- [Pair approximation predictions are approximate on finite graphs] → The scenarios test the direction of change on
  either side of a threshold, away from it, rather than the value at it.
- [A combination proves inexpressible] → The task records the system as blocked in the inventory and opens a separate
  change for the missing primitive.

## Migration Plan

None.

## Open Questions

- Should `social/common/games` be created for the payoff helpers if two modules share them? That is a new package and
  needs the maintainer's approval.
