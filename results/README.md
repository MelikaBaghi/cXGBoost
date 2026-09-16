# Result files behind the manuscript

Every table and figure of "Constrained Extreme Gradient Boosting for Adapting
Reduced-Order Models" (Baghi, Liu and Paynabar) is generated from the files in
this folder. Errors are relative reconstruction errors of a predicted basis on
the stored snapshots at that parameter, one row per held-out parameter.

| Folder | Manuscript | Contents |
|---|---|---|
| `benchmark/` | Table 2, Figures 2, 5, 7, 9 and 10 | five-fold cross-validated errors of cXGBoost, Grassmann interpolation and the projected Gaussian process (`*_errors.csv`, `*_pgp_errors.csv`), per-example summaries, and the paired Wilcoxon tests with Holm correction (`paired_tests_holm.json`) |
| `sample_size/` | Table 3 | mean error of the three predictors at every training budget, eight draws each (`sample_size.json`) |
| `nested_cv_jcp2/` | Table 4 | cXGBoost with the settings reselected inside each training fold: per-fold selections and outer errors (`<case>.json`, `<case>_errors.csv`), the comparison with Table 2 (`report.json`, `summary.json`) |
| `ablation_stability/` | Table 5, Figure 11 | raw-coordinate diagnostics of the constrained and unconstrained models (`<case>_stability.json`) and the stage-wise norms drawn in Figure 11 (`stage_trajectories.json`) |
| `ablation_native/` | Table 6 | the four versions of the boosting model and their paired tests (`compare.json`, `*_errors.csv`) |
| `sensitivity_jcp2/` | Section 4.1 | refits with the geodesic-centroid reference (`reference.json`) and with a lossless PCA cap on Kuramoto-Sivashinsky (`pca.json`) |
| `projection_cost_jcp2/` | Section 4.4 | solver counters of the five-fold fits (`cost.json`) |
| `audit/` | Section 4.3 | base-score and leaf-combination facts at held-out parameters (`heldout_gap.json`) |

Fold partitions are `kfold(n, 5, seed=42)` in dataset order, so rows with the
same `fold` and `mu` across files refer to the same held-out parameter.
