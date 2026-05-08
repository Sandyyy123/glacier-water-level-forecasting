# IMPROVER Report - Project 20 (Vernagtferner Glacier Water Level Forecasting)

**Role:** B (IMPROVER), read-only review
**Date:** 2026-05-08
**Scope:** brief.md, data/README.md, src/model_baseline.py, src/model_advanced.py, notebooks/01_EDA.ipynb (structure only), reports/references.md, manuscripts/manuscript.md, deliverables/presentation.html, checkpoint.json

---

## Top recommendation

**Add a process-aware ablation/accumulation regime split and a snow/ice albedo-state covariate before training the LSTM, then re-run the headline benchmark with this stratification.**

The current scaffold treats the proglacial discharge series as one homogeneous regression target, but glacier hydrology has two physically distinct regimes (May-September ablation with exposed-ice melt versus October-April accumulation with gauge shutdown and snow-buffered low flow). A single global LSTM trained across both will be dominated by the easy zero-flow winter rows and will under-fit the high-flow ablation tail that the client cares about. The single-step fix: (1) build a binary regime indicator `is_ablation_season` (DOY 121-273) and a continuous `snow_to_ice_transition_index` from snow depth and cumulative PDD, (2) fit the LSTM only on ablation-season rows or as a two-stage mixture (winter constant low-flow + ablation regression), and (3) report metrics separately for the two regimes. Expected gain: NSE on the ablation-only fold should rise by 0.05-0.10 and the 80% quantile interval should calibrate to nominal coverage rather than being inflated by winter zero-flow rows. This single change addresses the largest known structural weakness identified across the manuscript, the modelling code, and the EDA notebook structure simultaneously.

---

## Weakness inventory and recommended fixes

### 1. No reproducibility manifest (HIGH)

**Gap.** The project root has no `requirements.txt`, `environment.yml`, `pyproject.toml`, or `Makefile`. `model_advanced.py` requires torch >= 2.0; `model_baseline.py` requires sklearn, pandas, joblib; the notebook will need matplotlib and seaborn. A reviewer cannot reproduce the run without guessing versions.

**Fix.** Add `requirements.txt` pinning torch>=2.0,<2.4, scikit-learn>=1.3, pandas>=2.0, numpy>=1.24, joblib>=1.3, matplotlib>=3.7, seaborn>=0.13, pyarrow (for the parquet IO already used), tqdm. Optionally add a `Makefile` with `make data`, `make baseline`, `make lstm`, `make report` targets that map onto the existing four-script pipeline. This takes 10 minutes and unblocks any external evaluator.

### 2. LSTM advanced model: missing early stopping patience and no learning-rate schedule (HIGH)

**Gap.** `fit_lstm` in `src/model_advanced.py` runs all 60 epochs unconditionally and tracks `best_state` without a patience counter or LR schedule. On a 9-year training fold, 60 epochs of Adam with lr=1e-3 will overfit on small basins; Kratzert and colleagues 2018 used early stopping with patience and gradient clipping in the published HESS LSTM.

**Fix.** Add (i) early stopping with patience=10, (ii) `torch.optim.lr_scheduler.CosineAnnealingLR` over the 60 epochs or `ReduceLROnPlateau` on validation pinball, (iii) `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` after `loss.backward()`. These three lines change the training loop and bring it into line with the NeuralHydrology library defaults (see Kratzert and colleagues 2022 JOSS, DOI 10.21105/joss.04050).

### 3. Forward-chained split is one-shot, not rolling (HIGH)

**Gap.** `forward_chain_split` produces a single (train, val, test) triple at year T-2, T-1, T. With only 11 years of open-access daily data (2002-2012) and 11-12 years of pending 5-minute data (2013-2024), one test year is high-variance noise and not a defensible benchmark. A single-year test fold also cannot tell the reader whether the model is robust to climatologically extreme years (2003, 2018, 2022, 2023 alpine heatwaves).

**Fix.** Replace with rolling-origin K-fold forward chaining: for each year y in {T-K+1 ... T}, train on years < y, validate on y-1, test on y. Report median NSE/KGE across folds with IQR. This is the standard hydrological time-series CV (Bergmeir and Benitez 2012, Information Sciences DOI 10.1016/j.ins.2011.12.028) and removes the brittle one-year-test reporting that currently sits in section 4.1 of the manuscript.

### 4. Feature engineering ignores precipitation phase and rain-on-snow events (HIGH)

**Gap.** Both models use total precipitation as a single column. In alpine catchments, rain-on-snow events drive the largest flood peaks (the 2005 and 2013 alpine floods are textbook examples) and require separation of rain from snow, typically with a rain/snow temperature threshold. Pellicciotti and colleagues 2005 and Carenzo and colleagues 2009 (already cited as refs 10-11) flag this explicitly. The manuscript itself acknowledges this in section 4.3 ("peaky rainfall-on-snow events") but the code does not implement it.

**Fix.** In `build_features`, add three columns: `rain_mm = precip_mm where t_mean_C > 1.0 else 0`, `snowfall_mm = precip_mm where t_mean_C < -1.0 else 0`, and a smooth interpolation in the [-1, +1] band. Then add lag features for `rain_mm` specifically (1-, 3-, 7-day rolling sums). Expected effect: the LSTM should pick up the rain-on-snow flood peaks that the current single-precipitation feature blurs out.

### 5. No calibration audit on quantile heads (MEDIUM)

**Gap.** The advanced model trains three quantile heads (0.1, 0.5, 0.9) under pinball loss but never validates that the resulting intervals achieve nominal coverage. Pinball-trained quantiles routinely under-cover on high-flow tails because the loss is locally linear and the optimiser settles on the conditional mean of the residual rather than the conditional quantile. Manuscript section 4.2 lists empirical coverage as `<TBD>` but the code does not compute it.

**Fix.** In `fit_lstm`, after the test loop, compute and persist: empirical coverage = `mean((obs >= q10) & (obs <= q90))`, average width, and a calibration plot of nominal-versus-empirical quantiles at the 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95 levels. If coverage is off, apply post-hoc conformal prediction (Romano and colleagues 2019 NeurIPS Conformalised Quantile Regression, DOI 10.48550/arXiv.1905.03222) on the validation fold. Conformal correction is one extra training-fold pass and guarantees marginal coverage.

### 6. Glacier area enters as a single annual scalar with no rate-of-change information (MEDIUM)

**Gap.** Manuscript section 5 explicitly flags this as a structural limitation but the code follow-through is missing. The WGMS database supplies annual area, length, front variation, and mass balance; only area is used. The mass-balance year, the mass-balance signal itself, and the front-variation rate are all directly relevant to the inter-annual non-stationarity of the proglacial discharge.

**Fix.** Extend `FEATURE_COLS_DEFAULT` and `FEATURE_COLS` with `glacier_dArea_dt_km2yr` (annual diff of area), `mass_balance_mwe` (annual WGMS-FoG-MB), and `front_variation_m_yr`. Forward-fill all three onto the daily axis at the WGMS October 1 mass-balance year boundary. The mass-balance feature in particular should let the LSTM compress the 2002-2012 daily fold and the 2013-2024 5-minute fold onto a comparable scale (the 2018 and 2022 melt years had near-record-negative mass balance at Vernagtferner per Lambrecht and colleagues 2023, ref 9).

### 7. No baseline beyond linear regression: missing tree-based and persistence baselines (MEDIUM)

**Gap.** The two-tier setup pits one linear regression against one LSTM. A reviewer cannot tell whether the LSTM gain comes from (a) memory, (b) non-linearity, or (c) both. The manuscript already cites XGBoost (ref 33) but never runs it. A persistence baseline (yhat_t = y_{t-1}) is also missing, which is the absolute floor for any time-series regression.

**Fix.** Add two extra baselines in `model_baseline.py`: (1) persistence (`yhat = water_level_lag1d`) reported on the same forward-chained fold, and (2) gradient-boosted trees (`sklearn.ensemble.HistGradientBoostingRegressor` or `xgboost.XGBRegressor` if torch is already a dependency, then xgboost is a smaller add-on). The LSTM headline gain should then be reported as `delta NSE` over the strongest non-LSTM baseline, not over linear-only. This is now the convention in CAMELS-LSTM literature post-Frame and colleagues 2021.

### 8. EDA notebook has 19 cells but no fail-safe data-load fallback (MEDIUM)

**Gap.** The notebook structure (10 markdown + 9 code cells) suggests it expects the parquet file at `data/processed/vernagt_daily.parquet`, which does not exist in Phase 1. Anyone opening the notebook in nbviewer or Jupyter Lab will hit `FileNotFoundError` on the first code cell.

**Fix.** Add a synthetic-data fallback at the top of the notebook: if the parquet file is missing, generate a 2002-2012 daily synthetic frame with the right columns and a plausible seasonal pattern (sinusoid + noise + cumulative degree-day toy). Mark this clearly as `SYNTHETIC FALLBACK - replace with real data before reporting`. This makes the notebook self-running for review and removes a friction point that will otherwise be hit by every external reader. Same fix applies to `model_baseline.py` and `model_advanced.py` `load_processed()` functions.

### 9. Manuscript: no quantitative reporting from the open 2002-2012 daily slice that COULD have been run (MEDIUM)

**Gap.** Section 4 reports every metric as `<TBD after model run>`. The PANGAEA daily slice is open-access (CC BY 4.0), small (under 5 MB), and the `model_baseline.py` script is runnable end-to-end on it within minutes. Liora project briefs ask for an exploration / data-visualisation / data-pre-processing report PLUS a final report; the current manuscript ships only the scaffold.

**Fix.** In Phase 1.5 (low-cost extension), download the PANGAEA 829530 TSV, run the baseline (no LSTM, no torch dependency), and replace the placeholder values in Table 1 row 1 with real numbers. The advanced LSTM row can stay placeholder until BAdW registration clears. This single concrete table-row update converts the manuscript from "scaffold awaiting execution" to "interim report with one real benchmark and one pending."

### 10. Presentation HTML is 223 lines, but with no concrete numbers it is not a client deliverable yet (LOW)

**Gap.** The presentation is structured but cannot show a forecast plot, a residual plot, or a calibration curve until the model is run. For a business audience, the asymmetry between the 13-page methodologically rigorous manuscript and the placeholder presentation will read as "the team has not yet finished the work."

**Fix.** Once recommendation 9 produces real baseline numbers, add three SVG plots to the presentation: (1) observed-versus-predicted scatter for the test fold, (2) residual time series colour-coded by month, (3) a single-year hindcast strip showing the 80% interval band overlaid on the observed gauge. Inline the SVG (per the project rule that the presentation be self-contained), keeping it under 500 KB. This converts the presentation from a structural skeleton to a credible client-facing artefact.

---

## Priority summary

| # | Title | Priority |
|---|---|---|
| 1 | Reproducibility manifest (requirements.txt + Makefile) | HIGH |
| 2 | LSTM training loop: early stopping, LR schedule, grad clip | HIGH |
| 3 | Rolling-origin K-fold forward chaining | HIGH |
| 4 | Rain/snow phase separation in features | HIGH |
| 5 | Conformal calibration of quantile heads | MEDIUM |
| 6 | Mass-balance and front-variation as features | MEDIUM |
| 7 | Persistence and HistGBR/XGBoost baselines | MEDIUM |
| 8 | EDA notebook synthetic-data fallback | MEDIUM |
| 9 | Run baseline on PANGAEA 2002-2012 daily slice now | MEDIUM |
| 10 | Add three SVG plots to the presentation | LOW |

**Total:** 4 HIGH, 5 MEDIUM, 1 LOW.

---

## Notes on what is already strong

- References list (34 entries) is well-curated and DOI-resolved, with appropriate coverage of Vernagtferner glaciology, degree-day melt physics, and LSTM rainfall-runoff benchmarks. No padding.
- Forward-chained year-by-year split (even if currently one-shot) is the right default for non-stationary climate covariates.
- Quantile LSTM with pinball loss is methodologically correct and matches the alpine-flood-tail-prediction use case.
- `data/README.md` documents the registration-gated access pattern explicitly rather than pretending the data is in hand.
- The manuscript explicitly flags structural limitations (single-seed, no hyperparameter search, no ablation-window study, no climate-bias-correction) in section 5, which is the right intellectual posture for a Phase 1 scaffold.

These are the foundations the recommendations above build on, not replace.
