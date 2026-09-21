# Result files behind the manuscript

Every table and figure of "Constrained Extreme Gradient Boosting for Parametric
POD Subspace Prediction" (Baghi, Liu and Paynabar) is generated from the files in
this folder. Errors are relative reconstruction errors of a predicted basis on
the stored snapshots at that parameter, one row per held-out parameter.

| Folder | Manuscript | Contents |
|---|---|---|
| `benchmark/` | Table 2, Figures 2, 5, 7, 9 and 10 | five-fold cross-validated errors of cXGBoost, Grassmann interpolation and the projected Gaussian process (`*_errors.csv`, `*_pgp_errors.csv`), per-example summaries, and the paired Wilcoxon tests with Holm correction (`paired_tests_holm.json`) |
| `sample_size/` | Table 3 | mean error of the three predictors at every training size, eight draws each (`sample_size.json`) |
| `nested_cv_jcp2/` | Table 4 | cXGBoost with the settings reselected inside each training fold: per-fold selections and outer errors (`<case>.json`, `<case>_errors.csv`), the comparison with Table 2 (`report.json`, `summary.json`) |
| `ablation_stability/` | Table 5, Table 6 fit times, Figure 11 | raw-coordinate diagnostics of the constrained and unconstrained models and the wall-clock time of each five-fold fit, `wall_s`, which is the Fit time column of Table 6 (`<case>_stability.json`), and the stage-wise norms drawn in Figure 11 (`stage_trajectories.json`) |
| `ablation_native/` | Table 6 errors | the three versions of the boosting model in Table 6, a single constrained tree, and their paired tests (`compare.json`, `*_errors.csv`) |
| `sensitivity_jcp2/` | Sections 3 and 4.3 | refits with the geodesic-centroid reference (`reference.json`) and with a lossless PCA cap on Kuramoto-Sivashinsky (`pca.json`) and on Kolmogorov (`pca_kolmogorov.json`) |
| `projection_cost_jcp2/` | Section 4.4 | a separate refit of the Table 2 cXGBoost fits with counters in the leaf solver (`cost_fallbacks.json`): active solves, closed-form solves, Dykstra calls, sweeps, and the calls at which the scaled iterate replaced the projection. Its `wall_s` is the time of that refit, not the Table 6 fit time |
| `benchmark_r12/` | Section 3, rank check | Kolmogorov refitted at rank 12 (`kolmogorov_errors.csv`, `kolmogorov_summary.json`) |
| `rank_energy/` | Section 3, snapshot energy | share of snapshot energy held by the fixed POD rank at every parameter (`rank_energy.json`) |
| `tuning/` | Table 1 | the leave-one-out grid search on the cylinder and beam development splits and the settings it selected (`cylinder_grid.json`, `beam_grid.json`, `best.json`) |
| `audit/coordinate_energy_folds.json` | Section 3, coordinate energy | share of coordinate energy kept by the PCA reduction inside each training fold |
| `audit/leaf_optimality_kolmogorov_fold0.json` | Section 4.4 | fold 0 of Kolmogorov with every 40th active leaf solve re-solved by an interior-point method (`experiments/audit_leaf_optimality_real.py`): objective gap of the closed form, of converged Dykstra runs and of the scaled iterate against the constrained optimum; `leaf_optimality.json` is the same check on synthetic leaves |
| `audit/` | Sections 4.1 and 4.3 | the largest principal angle and Euclidean norm of every full-sweep target (`spectral_vs_frobenius.json`) and base-score and leaf-combination facts at held-out parameters (`heldout_gap.json`) |

Fold partitions are `kfold(n, 5, seed=42)` in dataset order, so rows with the
same `fold` and `mu` across files refer to the same held-out parameter.
