"""
Optimizer V1 — BayesOpt Tuning Pipeline for PHAS13 hiPSC-CM Model

Phases:
  I.   config + metrics (parameter registry, AP biomarker extraction)
  II.  cell_runner (single-cell simulation wrapper)
  III. cell_fitter (multi-objective BayesOpt for ionic params)
  IV.  tissue_runner + tissue_fitter (CV measurement + D optimization)
  V.   joint_refiner (GP emulator + NSGA-II co-optimization)
  VI.  validator (automated validation suite)
"""
