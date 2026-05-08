# Project 20 - Vernagtferner Glacier Water Level Forecasting

**Track:** Data Scientist - **Difficulty:** unknown (per Liora brief) - **Status:** Phase 1 scaffold

## Goal (from `Proposal - Glacier Water Level.pdf`)

Predict the water level of the proglacial stream draining the Vernagtferner glacier (Otztal Alps, Austria) from meteorological covariates measured at on-glacier and near-glacier stations and from annual geodetic measurements of the glacier extent.

The Vernagtferner is one of the longest-monitored alpine glaciers in the world (continuous mass-balance record since 1965, glaciological observations since 1889) and the operational hydro-meteorological station network around the glacier produces 5-minute resolution data over 2013-2024. The brief frames this as a regression / time-series forecasting problem.

## Source brief (verbatim spec)

- Relevant training course: Data Scientist
- Level of difficulty: unknown
- Description: Prediction of the water level of the stream running off the glacier Vernagtferner in the Alps on the basis of meteorological und geodetic data from measurement stations in the vicinity of the glacier and possibly on the glacier.
- Data resources cited:
  - Time span 2013 to 2024, sampled every 5 minutes
  - Distribution platform: pangaea.de
  - Meteorological covariates at 2 stations near the glacier: air temperature, relative humidity, wind speed and direction, precipitation, snow height, radiation
  - Hydrological response: water level (and possibly electrical conductivity of the stream)
  - Geodetic covariate: glacier extent measured every year
- Bibliography pointers in the brief:
  - https://geo.badw.de/en/the-project.html (BAdW Vernagtferner long-term monitoring programme)
  - https://doi.pangaea.de/10.1594/PANGAEA.829530 (Vernagtferner discharge dataset, Escher-Vetter and colleagues)
- Validation conditions: an exploration / data-visualisation / data-pre-processing report, plus a final report and code.

## Target variable

`water_level_cm` (continuous, regression). The proglacial stream gauge at the Vernagtbach Pegelstation reports water level in centimetres at 5-minute resolution. Discharge in m3/s can be derived through the rating curve published by the Bayerische Akademie der Wissenschaften (BAdW) Kommission fur Erdmessung und Glaziologie.

Optional secondary target: `discharge_m3_per_s` (computed from water level via station-specific rating curve), useful for cross-comparison with WGMS Fluctuations of Glaciers and GLAMOS reference glaciers.

## Modelling approach

Two-tier setup matching the rules document.

### Baseline (`src/model_baseline.py`)
- Daily aggregation of the 5-minute hydro-met record.
- Linear regression of mean daily water level on lagged temperature and precipitation (1d, 7d, 30d windows).
- Seasonal sinusoid features (sin/cos of day-of-year) to capture the strong glacier-melt seasonality.
- Diagnostics: residual QQ plot, autocorrelation of residuals, Nash-Sutcliffe Efficiency, KGE, RMSE, MAE.

### Advanced (`src/model_advanced.py`)
- Sequence-to-one LSTM on multi-modal climate covariates (temperature, precipitation, snow water equivalent, incoming shortwave radiation, relative humidity).
- 7-day input window, 1-day-ahead water-level forecast at daily resolution, with optional 5-min variant.
- Quantile output heads (0.1, 0.5, 0.9) for prediction intervals on the high-flow tail.
- Reference architecture aligned with Kratzert and colleagues 2018 HESS LSTM rainfall-runoff work; the ungauged-basin extension (Kratzert 2019) and TFT (Lim 2021) are flagged as Phase 2 follow-ups.

## Reports (Liora full-format slot)

- [ ] `reports/exploration_1.md` - schema, missing values, basic stats per station and per variable
- [ ] `reports/exploration_2.md` - distributions, seasonality, ablation-season regimes, rating-curve validation
- [ ] `reports/exploration_3.md` - feature engineering, lag selection, snow-melt covariate construction
- [ ] `reports/modeling_1.md` - baseline linear regression with seasonal terms
- [ ] `reports/modeling_2.md` - LSTM advanced model with quantile heads
- [ ] `reports/modeling_3.md` - error analysis, extreme-flow performance, model comparison
- [ ] `reports/architecture.md` - data flow, real-time scoring design, model serving notes
- [ ] `reports/final_report.md` - executive summary, findings, glacier-runoff implications

## Notebooks

- [ ] `notebooks/01_EDA.ipynb` (Phase 1 scaffold; not executed)
- [ ] `notebooks/02_features.ipynb`
- [ ] `notebooks/03_modeling.ipynb`
- [ ] `notebooks/04_evaluation.ipynb`

## Demo / artefacts

- [ ] Streamlit dashboard or static HTML with live water-level forecast
- [ ] Model card (`deliverables/model_card.md`)
- [ ] Final report PDF

## Open questions for Phase 2

1. Which gauge station: Vernagtbach Pegelstation (downstream) only, or also internal channels on the glacier tongue?
2. Data acceptance: PANGAEA hosts dataset 10.1594/PANGAEA.829530 for 2002-2012 daily discharge. The 2013-2024 5-min record cited in the brief is to be published; is the user pre-cleared by BAdW to use it pre-publication, or are we restricted to the published 2002-2012 daily slice?
3. Glacier-extent integration: WGMS Fluctuations of Glaciers ships annual area / volume / front variation. How to interpolate to daily features (linear vs glaciological-year step)?
4. Forecast horizon: 1-day-ahead (operational), 7-day-ahead (planning), or seasonal melt-volume?
5. Cross-validation strategy: per-year holdout vs sliding-window forward chaining? Forward chaining is the right default for non-stationary climate covariates.

## References (verified)

20+ refs in `reports/references.md`, every entry verified against CrossRef.
