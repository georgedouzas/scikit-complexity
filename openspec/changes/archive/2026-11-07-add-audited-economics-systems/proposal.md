# Proposal

## Why

The reference audit lists economics models of Argun, Bala, Caiani, Carmichael, Chesney, Gooding, Forrest, Hamill,
Thurner and of five recent papers that the catalog lacks, all judged expressible with the current core. Some chapters
were cited by catalog systems that implement other models, and the recitation of `fix-catalog-citations` left the
chapter models themselves missing: Caiani's guru network, Chesney's Fair Food Program, Gooding's Toy Trader and his
adaptive double auction.

## What Changes

- New catalog systems in `economics/resources`: `fishery_measurement_errors` (Ar Problem 14.8),
  `rice_food_balance` (Ba 4.11.3), and `crop_livestock_household` (Ba 4.11.6).
- New catalog systems in `economics/finance`: `guru_order_book` (Ca 3.2), `kirman_exchange_rate` (Ca 1.3.2),
  `default_cascade` (Th 5.5.3.2 and the Gai-Kapadia cascade of 2103.05623 §3.1, one system with a threshold and a
  recovery rate), `eisenberg_noe_clearing` (2103.05623 §3.1), and `overlapping_portfolio_fire_sales`
  (2103.05623 §3.2).
- New catalog systems in `economics/macroeconomics`: `brock_mirman_growth` (Ca 1.1.4), `network_herding` (Ca 1.1.6),
  `categorized_purchasing_power` (Fo 11.4 and 22), and `mark0` (2006.08469 §2).
- New catalog systems in `economics/markets`: `fair_food_program` (Cy 7), `toy_trader` (Go 8.2, 10, 11 and the
  appendix, with the three-goods variant as an option), `gooding_double_auction` (Go 11.3), and `household_demand`
  (Ha 3.2 and 3.3, the utility and the elasticity models as one system with a `demand` option).
- New catalog systems in `economics/econophysics`: `el_farol_coalitions` (Cm, Collins, strategic group formation).
- New catalog system in `economics/innovation`: `ccc_model` (Th 5.6.4).
- `urn_with_triggering` of the audit is not added: `interacting_urns_network` on a graph of one node is the urn with
  triggering.
- Every system has its module, registration, tests, inventory entry, and gallery example. The existing systems
  `order_book`, `fair_food`, `goods_traders`, and `double_auction` are unchanged.

## Capabilities

### New Capabilities

### Modified Capabilities
- `model-library`: the audited economics systems.

## Impact

- Systems: `skcomplex.systems.economics` resources, finance, macroeconomics, markets, econophysics, and innovation.
  No core change, no new package, and no new dependency.
- Docs: the inventory and the gallery examples.
