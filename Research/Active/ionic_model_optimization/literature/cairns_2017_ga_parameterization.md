---
paper: cairns_2017_ga_parameterization
title: "Efficient parameterization of cardiac action potential models using a genetic algorithm"
authors: "Cairns DI, et al."
year: 2017
journal: "Chaos"
doi: "10.1063/1.5000354"
pmid: "28964158"
pdf: ../papers/genetic_algorithm_ap_parameterization_2017_cairns.pdf
questions: [Q8]
---

## Key Findings
- Demonstrates GA can efficiently parameterize cardiac AP models
- Obtains good parameter fits with appropriate population sizes and generation counts
- Serves as a baseline comparison for more advanced methods (Bayesian, gradient-based)

## Method
- **Genetic Algorithm** applied to cardiac action potential model parameterization
- Details from abstract only — full text is paywalled (no PMC version available)

## Connections to Our Models
- GA is the simplest optimization baseline for our TTP06/ORd tuning
- Could implement using Python's `deap` library or scipy's differential evolution
- **Actionable**: Use GA as a rough initial search (wide parameter bounds), then refine with gradient-based or Bayesian methods

## Limitations
- Paywalled — full methodology not available for detailed analysis
- GA provides no uncertainty quantification
- May be superseded by Groenendaal 2015's iterative GA approach
