## Why

Papers of the survey of complex adaptive systems study cooperation where the structure of the population changes
or is split. Su, McAvoy, and Plotkin show that a network that switches among structures can favour cooperation when
every structure alone favours spite (2301.11982). Kessinger, Tarnita, and Plotkin let gossip groups with their own norms
of moral judgment compete, and find that Stern Judging spreads (2204.10811). Wang, Perc, and Szolnoki derive the
replicator equation of any multiplayer game on regular graphs and an exact threshold of peer punishment beyond which
defection dies out (2401.11686). Each model exercises a combination of core primitives that no system of the library
uses yet, and each gives analytic reference values against which the core can be checked: a network whose layer is
chosen by a Markov state, states on the edges of a complete network read through lookup tables, and payoffs summed over
the groups a node forms with its neighbours on random regular graphs.

## What Changes

- New system `DynamicNetworkDilemmaSystem` in `social/cooperation`: the donation game under weak selection on a
  population whose network switches between given structures by a Markov chain, with the two networks of dense and
  sparse communities of the paper as the default, and the function `critical_benefit_cost_ratio` that computes
  `(b/c)*` of the paper for given networks and transition rates.
- New system `GossipGroupNormsSystem` in `social/cooperation`: individuals that donate by the views of their gossip
  group, groups that assess every donor by their norm (Stern Judging, Simple Standing, Scoring, or Shunning) with
  errors of action and assessment, and group sizes that follow the payoffs of their members.
- New system `GraphPunishmentGameSystem` in `social/cooperation`: the public goods game with cooperators, defectors,
  and peer punishers on random regular graphs, played in the group of every node and its neighbours, with pairwise
  comparison updates.
- No change to the core. If an implementation finds that a combination is not expressible, the system is recorded as
  blocked and a separate change adds the missing primitive, as the constitution asks.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-library`: the dynamic network dilemma, the gossip group norms, and the graph punishment game.

## Impact

- Systems: a new module per system in `skcomplex.systems.social.cooperation`, and the shared parts they need, if any, in
  `social/common/games`.
- No new dependency. NetworkX generates the random regular graphs, and SciPy solves the linear systems of the critical
  ratio.
- Docs: a gallery example and an inventory entry per system. They exercise the existing primitives `networks`,
  `transitions`, `switches`, `discrete-states`, `network-generators`, `numeric-functions`, `neighbour-hyperedges`,
  `draws`, `reductions`, and `phases`.
- Left out of this change, since they add no combination: the explosive cooperation of 2303.11475, which is a sweep of
  the existing `simplicial_social_dilemmas`, the unlearnable games of 2312.12252, which `sk_game` already implements,
  and the non-reciprocal consumer-resource model of 2308.15757, which is a variant of the existing `consumer_resource`.
