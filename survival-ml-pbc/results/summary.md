# Results: survival ML on PBC

*Generated automatically by `src/run.py`. Runtime 1.3 min.*

Data: 312 randomised patients, 125 deaths, 19 transplants (censored).
Validation: 10 × repeated stratified 5-fold CV (50 test folds per model). Brackets give the 2.5–97.5 percentile range of the per-repeat mean, which reflects the variability due to the random split.

| Model | Harrell's C | Uno's C | Integrated Brier | 5-year AUC | ΔC vs Cox | folds better than Cox |
|---|---|---|---|---|---|---|
| Kaplan-Meier (no covariates) | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.194 [0.194, 0.195] | 0.500 [0.500, 0.500] | -0.345 | 0% |
| Cox PH | 0.845 [0.836, 0.851] | 0.805 [0.792, 0.818] | 0.121 [0.116, 0.128] | 0.922 [0.913, 0.928] | +0.000 | 0% |
| Ridge Cox | 0.857 [0.849, 0.865] | 0.816 [0.804, 0.829] | 0.142 [0.135, 0.153] | 0.929 [0.922, 0.934] | +0.012 | 76% |
| LASSO Cox | 0.850 [0.844, 0.855] | 0.812 [0.803, 0.819] | 0.117 [0.114, 0.121] | 0.926 [0.920, 0.932] | +0.005 | 68% |
| Elastic-net Cox | 0.852 [0.845, 0.857] | 0.814 [0.805, 0.822] | 0.118 [0.114, 0.122] | 0.927 [0.921, 0.933] | +0.008 | 76% |
| Componentwise boosting (CoxBoost-like) | 0.850 [0.843, 0.855] | 0.812 [0.803, 0.821] | 0.117 [0.114, 0.121] | 0.926 [0.920, 0.930] | +0.005 | 64% |
| Gradient-boosted trees | 0.826 [0.817, 0.834] | 0.785 [0.769, 0.796] | 0.129 [0.123, 0.135] | 0.905 [0.900, 0.911] | -0.019 | 12% |
| Random survival forest | 0.838 [0.829, 0.846] | 0.803 [0.796, 0.814] | 0.124 [0.122, 0.127] | 0.910 [0.904, 0.913] | -0.007 | 30% |

## Full-data Cox model (interpretation only)

Apparent (in-sample) C-index: 0.865, which is optimistic compared with the cross-validated value above.

| Variable | HR | 95% CI | p |
|---|---|---|---|
| age | 1.03 | 1.01–1.05 | 0.00218 |
| log(bili) | 1.88 | 1.36–2.59 | 0.000143 |
| log(chol) | 1.20 | 0.71–2.05 | 0.497 |
| albumin | 0.52 | 0.31–0.90 | 0.0186 |
| log(copper) | 1.34 | 0.98–1.84 | 0.0696 |
| log(alk.phos) | 0.94 | 0.71–1.23 | 0.646 |
| log(ast) | 1.66 | 0.95–2.87 | 0.0732 |
| log(trig) | 0.83 | 0.52–1.34 | 0.449 |
| platelet | 1.00 | 1.00–1.00 | 0.891 |
| log(protime) | 20.76 | 1.85–232.93 | 0.0139 |
| sex_female | 0.93 | 0.52–1.66 | 0.808 |
| ascites | 1.43 | 0.76–2.71 | 0.268 |
| hepato | 1.10 | 0.69–1.77 | 0.68 |
| spiders | 0.94 | 0.60–1.47 | 0.788 |
| trt_dpca | 0.96 | 0.66–1.41 | 0.841 |
| edema | 2.31 | 1.20–4.43 | 0.0122 |
| stage | 1.34 | 0.99–1.82 | 0.0593 |

## Proportional-hazards check (Schoenfeld residuals, rank time transform)

| Variable | χ² | p | Holm-adjusted p |
|---|---|---|---|
| age | 0.06 | 0.808 | 1 |
| albumin | 0.02 | 0.889 | 1 |
| alk.phos | 0.75 | 0.386 | 1 |
| ascites | 0.01 | 0.941 | 1 |
| ast | 0.10 | 0.747 | 1 |
| bili | 0.42 | 0.518 | 1 |
| chol | 1.90 | 0.168 | 1 |
| copper | 3.11 | 0.0779 | 1 |
| edema | 0.99 | 0.319 | 1 |
| hepato | 1.12 | 0.291 | 1 |
| platelet | 0.20 | 0.655 | 1 |
| protime | 3.13 | 0.0769 | 1 |
| sex_female | 0.62 | 0.429 | 1 |
| spiders | 0.74 | 0.388 | 1 |
| stage | 1.02 | 0.313 | 1 |
| trig | 0.22 | 0.636 | 1 |
| trt_dpca | 0.32 | 0.572 | 1 |

No covariate shows evidence against proportional hazards at the 5% level.
