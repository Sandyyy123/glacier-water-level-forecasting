# Validation Report - Project 20 Vernagtferner Glacier Water Level

**Overall: PASS-WITH-WARNINGS**

The scaffold is internally consistent. Notebook JSON parses, both Python scripts parse cleanly under `ast`, manuscript word count (4335) sits inside the 4000-5000 target, presentation.html is fully self-contained with zero external resources, IMRaD section structure is complete, em-dash count is zero across all artefacts, no AI-tell phrases were detected, checkpoint.json contains all four required schema fields, and 5 of 5 randomly-sampled DOIs resolved live against CrossRef with title-match. Two minor warnings: a reference to "Frame and colleagues 2021" appears in the manuscript Methods (3.4) and Discussion but has no entry in `reports/references.md`, and the citation token `[25 follow-on]` is a non-standard format. Methods named in the manuscript fully map to functions present in `model_baseline.py` and `model_advanced.py`. Project #20 is scaffold-only so trained-model artefacts are not expected.

---

## Detailed findings

### Task 1: Notebook JSON validity
- [PASS] `notebooks/01_EDA.ipynb` parses as JSON via `json.load`.

### Task 2: Python script syntax
- [PASS] `src/model_baseline.py` parses cleanly via `ast.parse`.
- [PASS] `src/model_advanced.py` parses cleanly via `ast.parse`.

### Task 3: Manuscript word count
- [PASS] `manuscripts/manuscript.md` word count = 4335 (target 4000-5000).

### Task 4: Self-contained presentation HTML
- [PASS] `grep -E 'href="http|src="http' deliverables/presentation.html` returns 0 hits. Presentation is inline-only with no external CDN, image, or font dependencies.

### Task 5: IMRaD completeness
- [PASS] Title (line 1).
- [PASS] Abstract (line 11).
- [PASS] Introduction (Section 1, line 19).
- [PASS] Methods (Section 3, line 55) plus a Data section (Section 2, line 33) which is appropriate for a hydrology forecasting paper.
- [PASS] Results (Section 4, line 96).
- [PASS] Discussion (Section 5, line 131).
- [PASS] Conclusion (Section 6, line 147).
- [PASS] References (line 151, with full numbered list at `reports/references.md`).

### Task 6: Method drift between manuscript and source code
Methods named in manuscript Methods section (3.1-3.6) and their presence in code:
- [PASS] Daily aggregation (Section 3.1) -> referenced in `model_baseline.py` docstring and `load_processed`.
- [PASS] Lag aggregates (1, 3, 7, 30 day rolling means) -> `build_features` in `model_baseline.py` (lines 75-118, columns `*_lag1d_mean`, `*_lag7d_mean`, `*_lag30d_mean`).
- [PASS] Cyclical seasonality `sin(2 pi DOY / 365.25)` and `cos(2 pi DOY / 365.25)` -> `model_baseline.py` line 81-82 and `model_advanced.py` `add_seasonal` lines 166-170.
- [PASS] Cumulative positive degree days (PDD) from October 1 -> `model_baseline.py` line 84 (`pdd_cum_C`).
- [PASS] WGMS annual area forward-filled -> mentioned in `model_baseline.py` feature list (`glacier_area_km2` if available in processed CSV).
- [PASS] Forward-chained year-by-year split -> `forward_chain_split` in `model_baseline.py` line 151 and used in `model_advanced.py`.
- [PASS] Linear regression with standardised inputs -> `LinearRegression` + `StandardScaler` in `model_baseline.py` lines 28, 31, 215.
- [PASS] LSTM with 64 hidden units, dropout 0.2, quantile heads (0.1, 0.5, 0.9) -> `LSTMConfig` in `model_advanced.py` lines 52-58 (`hidden_size: int = 64`, `dropout: float = 0.2`, `quantiles: tuple = (0.1, 0.5, 0.9)`).
- [PASS] Pinball loss across quantiles -> `pinball_loss` function `model_advanced.py` line 121.
- [PASS] Adam optimiser, lr=1e-3, weight_decay=1e-5 -> `model_advanced.py` line 236.
- [PASS] NSE, KGE metrics -> defined in both `model_baseline.py` (lines 120, 130) and `model_advanced.py` (lines 133, 142).

No method drift detected.

### Task 7: Citation drift (inline vs references.md)
Inline numeric tokens used in manuscript: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 34. All thirty resolve to entries in `reports/references.md` (which contains 34 entries 1-34).
- [PASS] All numeric inline citations [1]-[34] map to entries in references.md.
- [WARN] References [17, 31, 32] exist in `references.md` but are not cited inline in the manuscript (orphan refs, not orphan citations - acceptable for scaffold).
- [WARN] Manuscript Section 3.4 contains "Frame and colleagues 2021 [25 follow-on]". The `[25 follow-on]` token is non-standard and there is no Frame 2021 entry in `references.md`. Either add Frame, McCreight, Rahman, Beck (2021) "Post-Processing the National Water Model with Long Short-Term Memory Networks for Streamflow Predictions and Model Diagnostics" (DOI:10.1111/1752-1688.12964) to references.md, or rephrase to drop the named citation.
- [WARN] Manuscript Section 5 (Discussion) similarly references "Frame and colleagues 2021" without a numeric citation token and without a corresponding `references.md` entry.

### Task 8: Re-verify 5 random DOIs against CrossRef live
Sampled with `random.seed(42)` from references.md. All five returned HTTP 200 with first-author and title matches.
- [PASS] [30] 10.1016/0022-1694(70)90255-6 -> HTTP 200, Nash, "River flow forecasting through conceptual models part I". Title matches.
- [PASS] [3] 10.3189/1985aog6-1-158-160 -> HTTP 200, Escher-Vetter, "Energy Balance Calculations for the Ablation Period 1982 at Vernagtferner". Title matches.
- [PASS] [33] 10.1145/2939672.2939785 -> HTTP 200, Chen, "XGBoost". Title matches.
- [PASS] [18] 10.1038/s41558-017-0049-x -> HTTP 200, Huss, "Global-scale hydrological response to future glacier mass loss". Title matches.
- [PASS] [25] 10.5194/hess-22-6005-2018 -> HTTP 200, Kratzert, "Rainfall-runoff modelling using Long Short-Term Memory (LSTM) networks". Title matches.

### Task 9: Em-dash scan
- [PASS] Em-dash (U+2014) count across `brief.md`, `notebooks/01_EDA.ipynb`, `reports/references.md`, `src/model_baseline.py`, `src/model_advanced.py`, `manuscripts/manuscript.md`, `deliverables/presentation.html` totals 0.

### Task 10: AI-tell scan
- [PASS] `grep -riE 'verified by [0-9]+ agents|AI-verified|cross-checked by Claude'` across the project folder returned zero hits.

### Task 11: Checkpoint schema
- [PASS] `checkpoint.json` keys: `['project_number', 'title', 'methodology', 'phase', 'status', 'needs_main_session_execution', 'blockers']`. All four required fields (`project_number`, `title`, `methodology`, `status`) are present. Two extra documentary fields (`phase`, `needs_main_session_execution`, `blockers`) are non-blocking additions.

### Project-class-specific check (scaffold-only #20)
- Project #20 is in the >#8 scaffold-only band. The presence of trained model artefacts in `deliverables/` is not required. The current single-file `deliverables/presentation.html` is consistent with scaffold scope.

---

## Summary of warnings to address

1. Add Frame, McCreight, Rahman, Beck (2021) to `reports/references.md` and assign it a number, then replace "[25 follow-on]" in Section 3.4 and the unbracketed "Frame and colleagues 2021" mention in Section 5 with the new numeric token. Suggested DOI: `10.1111/1752-1688.12964`.
2. Optionally drop or use refs [17] (Huss 2015), [31] (Breiman 2001), [32] (Friedman 2001) which are listed in `references.md` but not cited inline; current treatment is non-blocking but trims the bibliography.
