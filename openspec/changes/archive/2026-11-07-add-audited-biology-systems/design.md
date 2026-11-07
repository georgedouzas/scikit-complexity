# Design

## Context

Every model of the biology rows of the audit is expressible with the current core. The work is to write each one in
its subfield with the realizations it needs, and to choose the primitives that the source describes.

## Decisions

- **Variants are options.** MacArthur's consumer-resource model, the chemostat supply, essential resources, and Gros's
  competition for one resource are one system, `consumer_resource`, with a `supply` and a `kind` option. The
  essential-resource growth needs the uptake of a consumer inside a reduction over consumers, which the nested
  cross-group reductions of the core give.
- **Interactions read the other group through reductions.** Consumer growth and resource turnover are two interactions,
  since an interaction that reads two groups member by member needs groups of one size.
- **Delays of edge states.** Delays read states of single elements only, so the adaptive delay network binds single
  units and axons, one interaction per axon, instead of a group on edges.
- **Smooth spikes for plasticity.** The STDP network uses theta neurons with smooth pulses and spike traces, the
  continuous approximation of the event-based update that the source mentions, so plasticity is a rate law per edge.
- **Rules for algorithms.** The NKCS climbing, the competitive Jain-Krishna replacement, ant colony rounds, polymer
  joining, colocation infection, and shepherding are interaction rules over agent groups, with views for typed access.
- **Data-defined kinetics.** The glycolysis model of BioModels BIOMD0000000042 declares its fifteen concentrations and
  twenty-five constants as declared variables of one element, named as in the source.
- **Source gaps.** Where a source names a model without its rules, the standard published form is written and cited:
  Watts and May (1992), Kretzschmar and Morris pair formation, Strömbom et al. (2014), West, Brown, and Enquist
  (1997), Ma et al. (2009), Thomas's logical networks, and EpiSims as a synthetic city.

## Risks

- The printed coefficients of the penaeid shrimp model make its respiration exceed its ration, so the biomass declines.
  The factory exposes the respiration coefficient and documents it.
- Miki and Nakamura's shepherding is described in the review without rules, and a reconstruction from its prose did not
  herd reliably, so it is left out of this change.
