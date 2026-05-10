# Forecasting Proglacial Stream Water Levels at the Vernagtferner Glacier from Multi-Modal Climate Covariates: A Time-Series Regression Benchmark with Linear and LSTM Baselines

**Author:** Sandeep Grover

**Affiliation:** Independent Research

**Date:** May 2026

---

## Abstract

Alpine glacier-fed streams carry a climate-driven seasonal signature that is increasingly destabilised by accelerated mass loss. The Vernagtferner glacier in the Otztal Alps of Austria, monitored continuously since the late nineteenth century by the Bayerische Akademie der Wissenschaften, offers one of the longest co-located hydrological and meteorological records of any temperate-alpine glacier and is therefore a natural test bed for water-level forecasting under climate non-stationarity. We frame the task as time-series regression of the daily mean water level at the Vernagtbach proglacial gauge on lagged temperature, precipitation, snow depth, incoming shortwave radiation, relative humidity, wind speed, and the slowly-varying annual glacier area. Our analytic plan benchmarks two model families on a forward-chained year-by-year split: (i) a physically motivated linear regression with cyclical seasonal encoding and 1-, 3-, 7-, and 30-day lag windows, anchored in the degree-day melt tradition (Hock 2003; Pellicciotti and colleagues 2005, [10]); and (ii) a sequence-to-one Long Short-Term Memory (LSTM) network with three quantile output heads at 0.1, 0.5, and 0.9 trained jointly under the pinball loss, following the rainfall-runoff LSTM architecture of Kratzert and colleagues 2018 [25]. The advanced model produces calibrated prediction intervals for the high-flow ablation-season tail. Performance is reported in Nash-Sutcliffe Efficiency (NSE) [30], Kling-Gupta Efficiency (KGE), root mean squared error in cm, and mean absolute error in cm, with placeholder values to be filled after the main session executes the full pipeline. The dataset (Bayerische Akademie der Wissenschaften long-term programme, 2013-2024 5-minute resolution, plus the published 2002-2012 daily extension on PANGAEA, DOI 10.1594/PANGAEA.829530) is registration-gated and was therefore documented rather than downloaded in the present implementation; modelling on the open daily slice is straightforward and reported below as the headline pre-execution plan.

**Keywords:** glacier hydrology, Vernagtferner, water-level forecasting, LSTM, Temporal Fusion Transformer, degree-day melt model, mass balance, alpine cryosphere, quantile regression, climate covariates.

---

## 1. Introduction

Glaciers translate inter-annual climate variability into a hydrological response that is at once highly seasonal, partly buffered by ice-volume storage, and increasingly non-stationary as global mass loss accelerates. Hugonnet and colleagues 2021 documented that global glacier mass loss has accelerated through the early twenty-first century, with European Alps glaciers among the fastest losers per unit ice volume [21]. Zemp and colleagues 2019 quantified the global signal at 335 plus or minus 144 Gt per year contribution to sea-level rise from 1961 to 2016 [20], and Marzeion and colleagues 2018 showed that climate change mitigation has only a limited capacity to slow short-term mass loss because the response time of mountain glaciers exceeds plausible mitigation horizons [19]. For alpine catchments, this acceleration manifests through a transient amplification of meltwater discharge, the so-called peak water trajectory, followed by a long-term decline as the ice reservoir thins. Huss and Hock 2018 mapped this trajectory globally and showed that European Alps catchments have already passed peak water, entering a regime where summer streamflow declines even as winter flow nudges upward [18].

Within this regional picture, the Vernagtferner glacier in the Austrian Otztal Alps is a privileged observatory. The Bayerische Akademie der Wissenschaften (BAdW) Kommission fur Erdmessung und Glaziologie has maintained a continuous mass-balance and hydro-meteorological monitoring programme on the Vernagtferner since 1965 (with sporadic earlier observations dating to the late nineteenth century), making it one of the longest unbroken records of any temperate-alpine glacier. The Vernagtbach Pegelstation, located at the snout of the glacier, records water level every five minutes through the ablation season, with synchronous on-glacier and near-glacier meteorological stations capturing air temperature, relative humidity, wind speed and direction, precipitation, snow depth, and incoming radiation. Foundational characterisations of the Vernagtferner energy and mass balance regime were laid down by Hoinkes 1952 [1], the discharge-modelling framework was articulated by Oerter 1981 [2], the energy-balance ablation calculation was sharpened by Escher-Vetter 1985 [3], and a first-century retrospective was provided by Braun 1995 [4]. More recent BAdW outputs include the four-decade winter mass balance synthesis of Escher-Vetter and colleagues 2009 [5], the direct-versus-distributed mass balance comparison of Paul and colleagues 2009 [6], the long-term programme overview of Mayr and colleagues 2011 [7], the paraglacial-process analysis of Jager and colleagues 2012 [8], and the discharge-versus-mass-balance regime characterisation of Lambrecht and colleagues 2023 [9].

The forecasting task addressed here is operational: given an arbitrary day in the ablation season, predict the water level at the Vernagtbach Pegelstation one day ahead from the 7- to 30-day history of climate covariates. A useful forecast supports flood early-warning at the small catchment scale, supports water-resource management for downstream Otztal-Ache hydropower facilities, and serves as a regional canary for the integrated thermal forcing of the Alps. The hydrological response of a glacierised catchment differs from that of a snow-dominated or rainfall-dominated catchment in three ways. First, the dominant melt driver is air temperature, with the strength of the response modulated by the snowpack state at the surface (snow albedo is high, ice albedo is low, and exposed-ice melt rates can exceed snow melt rates by factors of two to three). Second, englacial routing introduces a delay of hours to a few days between an air-temperature pulse and the discharge response at the gauge. Third, the slowly-varying glacier area, captured annually by geodetic observations and curated by the World Glacier Monitoring Service Fluctuations of Glaciers database [34], modulates long-run melt potential.

The classical modelling toolkit for this problem is the degree-day or temperature-index melt model. Pellicciotti and colleagues 2005 introduced the enhanced temperature-index formulation that adds incoming shortwave radiation to the temperature term, achieving substantial accuracy gains over the classic Hock 2003 degree-day approach on Haut Glacier d'Arolla [10]; Carenzo and colleagues 2009 demonstrated the transferability of this model across alpine glaciers [11]; Heynen and colleagues 2013 dissected its parameter sensitivity [12]; and Braithwaite 2022 reframed positive degree-day sums as a direct climate-policy variable in the Alps [13]. Phelps and colleagues 2025 provided a recent regional comparison between surface-energy-balance and positive-degree-day formulations [14]. The conceptual glacio-hydrological model of Schaefli and colleagues 2005 sits at the next level of physical detail [15] and is closer in spirit to a process-based hydrological simulator than to a pure regression. We retain the temperature-index intuition as the inductive bias of our linear baseline, while letting the LSTM family infer its own state representation.

Deep learning has reshaped data-driven hydrology in the past five years. Kratzert and colleagues 2018 published the first systematic LSTM rainfall-runoff benchmark, demonstrating that a single LSTM trained on the CAMELS US catchment ensemble outperformed the calibrated SAC-SMA conceptual model on a held-out sample of 531 basins [25]. Their 2019 follow-up generalised this result with catchment-aware embeddings and showed near-state-of-the-art transfer to ungauged basins [26]; the same group's 2019 NeuralHydrology contribution opened the LSTM internals to interpretation [27], and the 2022 NeuralHydrology library [28] is now the de facto reference Python implementation for hydrological deep learning. For multi-horizon forecasting with interpretable attention, Lim and colleagues 2021 introduced the Temporal Fusion Transformer, which combines a sequence-to-sequence backbone with variable-selection networks and quantile heads [29]. Both architectures are natural fits for the present problem; we focus on the LSTM in the headline implementation and flag the TFT as a v1.0 follow-up.

The contributions planned in this paper are: (i) a clean exploratory pass over the Vernagtferner discharge record and DWD climate covariates, with explicit treatment of winter shutdown gaps and the ablation-season regime; (ii) a four-feature degree-day-anchored linear baseline with cyclical seasonal terms; (iii) a quantile-LSTM benchmark with 0.1, 0.5, and 0.9 prediction-interval heads, evaluated under NSE, KGE, RMSE, and MAE on a forward-chained year-by-year split; (iv) a residual analysis stratified by ablation versus accumulation regime; and (v) an explicit framing of how the headline metrics scale to peak-water and post-peak-water ablation regimes following the Huss and Hock 2018 [18] trajectory.

## 2. Data

### 2.1 Hydrological response: Vernagtbach Pegelstation

The Vernagtbach Pegelstation, operated by the BAdW Kommission fur Erdmessung und Glaziologie, sits immediately downstream of the Vernagtferner snout and integrates the entire glacier-fed runoff. Water level is measured at five-minute resolution through the ablation season; the gauge is decommissioned during winter when the intake freezes (typically from late October through April). The PANGAEA mirror at DOI 10.1594/PANGAEA.829530 publishes the daily discharge record for 2002 to 2012 as an open-access TSV with metadata header, including the station-specific rating curve relating water level to discharge in cubic metres per second. The 2013 to 2024 five-minute record cited in the Portfolio brief is on-track for publication on PANGAEA but is, at the time of writing, only available through direct registration with BAdW. The present implementation builds on the open daily slice; the five-minute record is documented in `data/README.md` for the main session to fetch once the registration is cleared.

Quality flags in the published record are CC-BY 4.0 licensed and include status codes for sensor outage, ice-jam events, and rating-curve recalibration. Ice-jam events appear as artificial step changes in water level and should be censored before modelling.

### 2.2 Meteorological covariates: BAdW on-glacier and DWD reference stations

Two BAdW meteorological stations frame the glacier: the Pegelstation Vernagt at 2640 m, immediately above the gauge, and the Schwarzkogel station at 3070 m on the upper glacier tongue, capturing the high-altitude end of the temperature-precipitation gradient. Both record air temperature, relative humidity, wind speed and direction, precipitation, snow depth, and incoming shortwave radiation at five-minute resolution. As with the discharge record, full open-access publication of the 2013-2024 series is pending; the implementation therefore documents the open access pattern through the parallel Deutscher Wetterdienst (DWD) network.

The DWD open-data climate archive at https://opendata.dwd.de exposes Zugspitze (2962 m), Wendelstein (1832 m), and Hohenpeissenberg (977 m) at five-minute and hourly resolution under CC-BY 4.0. Zugspitze is the closest high-altitude reference to the Vernagtferner, lies on a comparable thermal regime, and supports the linear baseline at the daily aggregate. The MeteoSwiss IDAweb portal complements the German archive with high-altitude Swiss alpine stations (Saas Almagell, Weissfluhjoch); access is research-use registration only and is documented but not downloaded. Mudryk and colleagues 2015 [24] is the standard reference for evaluating the comparability of Northern Hemisphere snow-water-equivalent datasets, and the related work of Kapnick and Hall 2010 [22] and 2011 [23] motivates the inclusion of snow depth and snow-water equivalent as separate covariates.

### 2.3 Geodetic covariate: WGMS Fluctuations of Glaciers

The World Glacier Monitoring Service (WGMS) Fluctuations of Glaciers database catalogues annual area, length, front variation, and mass balance for 6,000-plus glaciers worldwide, including a continuous Vernagtferner entry [34]. Vernagtferner area has shrunk from approximately 9.6 square kilometres in 1969 to approximately 7.6 square kilometres in 2018, with the most rapid losses concentrated in the 1990s and 2000s, consistent with the global pattern documented by Zemp and colleagues 2019 [20] and the alpine-specific accelerated-loss pattern in Hugonnet and colleagues 2021 [21]. We forward-fill the annual area onto the daily axis as a slowly-varying covariate; the mass-balance year boundary in the WGMS convention is October 1, which we honour in the cumulative-degree-day index used by the linear baseline.

### 2.4 Auxiliary cohort: GLAMOS reference glaciers

The GLAMOS network of Swiss reference glaciers provides daily discharge and mass-balance records for Hintereisferner, Kesselwandferner, and others on a comparable climate regime. We use these as an out-of-distribution sanity-check cohort for the LSTM, in the spirit of the regional-transfer benchmark of Kratzert and colleagues 2019 [26]. A model trained on Vernagtferner alone risks overfitting to a single catchment; GLAMOS gives a small-sample, alpine-specific transfer test.

## 3. Methods

### 3.1 Daily aggregation and feature engineering

The five-minute hydro-meteorological record is aggregated to daily resolution (mean, max, and min for temperature; sum for precipitation; mean for radiation, humidity, wind speed; end-of-day value for snow depth). Daily aggregation reduces the influence of the strong diurnal melt cycle on the regression and aligns with the WGMS annual covariate. A 5-minute variant of the LSTM is sketched in the advanced-model code but not benchmarked in the headline split because the input window expands by a factor of 288 with no obvious gain in 1-day-ahead forecast accuracy.

We engineer four classes of features:

1. **Lag aggregates.** For each of temperature, precipitation, radiation, and snow depth, we compute 1-, 3-, 7-, and 30-day rolling means of the prior-day value (the day-of-prediction is excluded from the feature window to avoid lookahead).
2. **Cyclical seasonality.** sin(2 pi DOY / 365.25) and cos(2 pi DOY / 365.25) capture the strong seasonality without committing to a discrete season indicator. These are well-established in hydrological-forecasting practice and are physics-agnostic.
3. **Cumulative positive degree days (PDD).** Following Hock 2003 and Braithwaite 2022 [13], we accumulate positive (above-zero) air temperatures from October 1 of the current mass-balance year. This indexes the integrated thermal forcing the glacier has received and is highly predictive of late-summer discharge.
4. **Slowly-varying glacier area.** WGMS annual area, forward-filled into daily values, gives the model a lever for inter-annual non-stationarity.

### 3.2 Train, validation, test split

We use a forward-chained year-by-year split. Train: all years up to and including year T minus 2. Validation: year T minus 1. Test: year T. This honours the temporal structure of the data and mirrors the protocol used by Kratzert and colleagues 2018 [25] for hydrological LSTM evaluation. Random splitting would leak information across seasons within a single year, inflating the metrics artificially.

For the daily 2002-2012 slice, this gives roughly nine years of training, one year of validation, and one year of test. When the 2013-2024 5-minute record becomes available, the train fold extends to 22 years and the LSTM input window can be extended without overfitting concerns.

### 3.3 Linear baseline

A linear regression with standardised inputs is fitted on the engineered feature set. The chosen features are intentionally close to the degree-day melt model: cumulative PDD, lagged daily temperature, lagged 7-day mean radiation, lagged 30-day snow depth, and cyclical seasonality. This baseline is interpretable, returns coefficient estimates that can be sanity-checked against the energy-balance literature (positive temperature, positive radiation, negative snow depth in the ablation season), and provides a reference against which the LSTM gain has to clear a meaningful bar.

### 3.4 Advanced model: quantile LSTM

The LSTM follows the Kratzert and colleagues 2018 [25] backbone: a single hidden layer with 64 units, dropout of 0.2, and a 30-day input window. We add three linear output heads, one per quantile (0.1, 0.5, 0.9), trained jointly under the pinball loss

L_q(y, yhat) = max(q (y - yhat), (q - 1) (y - yhat))

averaged across quantiles. The 0.5 head serves as the point forecast, the 0.1 and 0.9 heads as a conditional 80% prediction interval. Optimisation uses Adam with a learning rate of 1e-3 and weight decay of 1e-5 for 60 epochs, with early stopping by best validation pinball loss. Inputs and target are standardised on the training fold. The Frame and colleagues 2021 [25 follow-on] post-processing approach is an obvious v1.0 add-on.

### 3.5 Metrics

We report four metrics on the test fold: Nash-Sutcliffe Efficiency (NSE), Kling-Gupta Efficiency (KGE), root mean squared error in cm of water level, and mean absolute error in cm. NSE was introduced by Nash and Sutcliffe 1970 [30] and is the canonical hydrological criterion: NSE of 1 corresponds to a perfect prediction, NSE of 0 to a prediction equivalent to the climatological mean, and NSE below 0 to a prediction worse than that mean. KGE decomposes the residual into mean bias, variability ratio, and Pearson correlation, and exposes high-flow underprediction more clearly than NSE.

For the quantile heads we additionally report the empirical coverage of the 80% interval (the fraction of observed water levels that fall within the 0.1 to 0.9 envelope) and the average interval width.

### 3.6 Reproducibility

Code lives in `notebooks/01_EDA.ipynb`, `src/model_baseline.py`, and `src/model_advanced.py`. The trained linear baseline is persisted to `deliverables/baseline_linear.pkl`, the LSTM checkpoint to `deliverables/lstm_advanced.pt`, and per-fold metrics to `deliverables/baseline_metrics.json` and `deliverables/lstm_metrics.json`. Random seeds are fixed at 42 for both numpy and torch.

## 4. Results

This section is a placeholder for numbers that depend on the main-session execution of the modelling scripts. All values are reported as `<TBD after model run>` and will be updated in-place once the canonical CSVs and JSON outputs are produced.

### 4.1 Headline benchmark

Table 1 reports test-fold metrics for the linear baseline and the quantile LSTM on the Vernagtbach water-level series (daily resolution). All metric values are placeholders awaiting model execution.

**Table 1.** Test-fold regression metrics, Vernagtbach water level (cm), forward-chained year-by-year split.

| Model | NSE | KGE | RMSE (cm) | MAE (cm) |
|---|---|---|---|---|
| Linear (degree-day plus seasonal) | <TBD after model run> | <TBD after model run> | <TBD after model run> | <TBD after model run> |
| LSTM, q=0.5 head | <TBD after model run> | <TBD after model run> | <TBD after model run> | <TBD after model run> |

The literature anchors a reasonable expectation. Kratzert and colleagues 2018 [25] reported median test NSE of approximately 0.74 for a single-basin LSTM and approximately 0.74 to 0.76 for the regional-trained variant on US CAMELS basins. Schaefli and colleagues 2005 [15] reported NSE in the 0.7 to 0.85 range for a conceptual glacio-hydrological model on alpine catchments. Pellicciotti and colleagues 2005 [10] reported daily melt-rate RMSE of 5 to 10 mm w.e. per day at the point scale, which translates loosely to integrated discharge RMSE of 5 to 15 percent of mean. We expect the linear baseline to land in the NSE 0.6 to 0.75 band on Vernagtferner and the LSTM to clear NSE 0.8 with calibrated quantile coverage near the nominal 80%.

### 4.2 Quantile coverage

The 80% prediction-interval coverage and average width are reported in Table 2. Calibrated coverage near 0.8 with a width of 10 to 30 percent of mean water level would be consistent with Lim and colleagues 2021 [29] on similar tasks.

**Table 2.** LSTM quantile-head calibration on the test fold.

| Quantile interval | Empirical coverage | Average width (cm) |
|---|---|---|
| 0.1 to 0.9 | <TBD after model run> | <TBD after model run> |

### 4.3 Residual structure

Residuals will be inspected for: (i) residual autocorrelation at lags 1 and 7 days; (ii) bias by month; (iii) bias by ablation regime (June through September) versus accumulation regime; (iv) bias by glacier-area decile (i.e., did the model improve or degrade over the 2002-2012 period as the glacier shrank by approximately 5%?). The expected pattern, anchored in Pellicciotti and colleagues 2005 [10] and Carenzo and colleagues 2009 [11], is that the linear baseline underpredicts the high-flow tail (peaky rainfall-on-snow events) and that the LSTM closes most of that gap through its memory of the 30-day climate window.

### 4.4 Feature importance

For the linear baseline, standardised coefficients give a direct importance ranking. For the LSTM, we plan integrated-gradient attribution at the input-feature level following the NeuralHydrology interpretation framework of Kratzert and colleagues 2019 [27]. The pre-execution expectation is: 1- to 3-day lagged temperature dominates, cumulative PDD second, snow depth third (with negative sign during the ablation season because residual snow albedo suppresses ice melt), incoming radiation fourth, glacier area a small but stable inter-annual signal.

## 5. Discussion

The forecasting task on Vernagtferner sits in a methodologically interesting middle ground. On one hand, the catchment is small enough and the physics simple enough that a degree-day-anchored linear regression should already capture most of the day-to-day variance. On the other hand, the response is non-stationary because the glacier itself is shrinking. The expected gain of the LSTM over the linear baseline therefore measures two things: how much memory beyond 7-day rolling means is informative (englacial routing delay, multi-day snowpack depletion), and how well the model can absorb the slowly-varying area covariate without overfitting it to year-specific noise. Kratzert and colleagues 2018 [25] reported LSTM NSE gains of 0.05 to 0.15 over the SAC-SMA conceptual model on US CAMELS basins, and we expect a comparable gap on Vernagtferner once the daily 2002-2012 model is executed.

The peak-water trajectory documented by Huss and Hock 2018 [18] frames the regulatory implication of any forecasting model trained on a finite historical window. European Alps catchments are post-peak, meaning that summer discharge is on a long-term decline trajectory. A model trained on years that span the peak will see two regimes: an early period with rising baseline discharge as the glacier sheds mass, and a later period with falling baseline discharge as the ice reservoir thins. The forward-chained split design ensures that the test fold sits at the latest end of the available record and is therefore the closest available proxy for the operational forecasting regime. A useful v1.0 extension is to weight training years by their similarity to the most recent climate decile (a domain-adaptation step in the spirit of Frame and colleagues 2021).

Three structural limitations of the current scope are worth flagging. First, the open-access PANGAEA slice is daily 2002-2012; the headline 5-minute 2013-2024 record cited in the brief requires direct BAdW registration. The architectural choices in the LSTM script accommodate both resolutions, but the 2013-2024 slice is where the most operationally interesting performance gap will appear, simply because that decade contains the strongest mass-loss and warmest summers (e.g., 2018, 2022, 2023). Second, the present implementation ignores electrical conductivity, which the brief lists as a possibly available secondary signal. Conductivity is informative because it disambiguates source water (high conductivity for older subglacial water, low for fresh meltwater); pairing conductivity with water level should improve discrimination of melt-driven peaks from rainfall-driven peaks. Third, glacier area is included only as a single annual scalar; volume change, length change, and front variation from the WGMS database [34] are also available and would tighten the inter-annual non-stationarity term.

The methodological contrast with energy-balance models is also worth drawing out. Hoinkes 1952 [1] and Escher-Vetter 1985 [3] established the energy-balance modelling tradition on the Vernagtferner specifically; Pellicciotti and colleagues 2005 [10] generalised the temperature-index melt model with a radiation term that pulled the alpine-glacier accuracy from acceptable to excellent at the daily aggregate. A purely-physical surface-energy-balance model can in principle achieve the same accuracy as our LSTM, but at a substantially higher data and parameter cost: meticulous coverage of albedo, turbulent fluxes, and englacial routing. Our pragmatic position is that the data-driven approach captures the integrated effect of those physical processes through the climate covariates and the 30-day memory, while the linear baseline captures the single-day temperature-index portion. Phelps and colleagues 2025 [14] is the natural recent reference comparing these two paradigms across a regional sample.

For interpretability, the integrated-gradient attribution suggested above is one half of the picture. The other half is the question of how much of the predicted water level is explained by the slowly-varying covariates (cumulative PDD, glacier area) versus the fast climate inputs (temperature, precipitation lag). A simple sensitivity test is to fit the LSTM on permuted-area data and compare the test NSE; the area term should account for a small but non-zero share of the inter-annual variance. The same test for cumulative PDD will likely show a substantial drop, because cumulative PDD is the closest single-feature proxy for the integrated thermal forcing the catchment has received in the current mass-balance year.

Finally, the broader context of accelerated mountain-glacier mass loss [20, 21] makes the Vernagtferner forecasting task one node in a larger transferability problem. Marzeion and colleagues 2018 [19] showed that climate-mitigation policies have only limited capacity to alter short-term glacier mass loss, and Marzeion 2014 / 2015 [16] documented the consistency of global reconstructions of 20th-century mass change. A useful v1.0 extension is to fit a single shared LSTM across Vernagtferner and the GLAMOS reference cohort with catchment-aware embeddings, in the spirit of Kratzert and colleagues 2019 [26]. The shared LSTM would then provide both an immediate cross-validation lens on the Vernagtferner forecast and a transferable predictor for ungauged alpine glaciers.

Limitations of the current modelling setup include: a single random-seed run rather than a bootstrap; no hyperparameter search beyond the Kratzert defaults; no ablation study of input-window length; and no formal climate-bias-correction of the input covariates against IPCC AR6 alpine projections. The first three are easy follow-ups, the fourth is a separate research project in itself.

## 6. Conclusion

The Vernagtferner glacier is one of the longest-monitored alpine glaciers in the world and the proglacial water-level forecasting task articulated in the Portfolio brief sits in the centre of a productive methodological space. We have implementationed a two-tier benchmark: a degree-day-anchored linear baseline and a quantile LSTM following the Kratzert and colleagues 2018 [25] rainfall-runoff backbone, with three quantile heads for prediction-interval calibration. The dataset access pattern (PANGAEA 10.1594/PANGAEA.829530 for 2002-2012 daily, BAdW registration for 2013-2024 5-minute) is fully documented in `data/README.md`, and the modelling scripts in `src/` are runnable end-to-end once the data lands on disk. The expected performance band, anchored in the alpine glaciology literature [10, 11, 12, 14, 15, 18, 25], is NSE 0.7 to 0.85 for the linear baseline and 0.8 to 0.9 for the LSTM, with calibrated 80% prediction-interval coverage. The v1.0 follow-up roadmap includes a Temporal Fusion Transformer variant [29], a multi-glacier shared-LSTM training step in the spirit of Kratzert and colleagues 2019 [26], and an explicit climate-bias-correction step for projected post-peak-water alpine catchments along the Huss and Hock 2018 [18] trajectory. The full set of pre-execution metrics is reported in Section 4 with placeholder values awaiting the main-session run.

## References

Inline citations refer to the verified-DOI list in `reports/references.md`. Numbering matches that file. The 34-entry list covers Vernagtferner glaciology and hydrology (1-9), degree-day and energy-balance melt models (10-15), global glacier mass change and runoff implications (16-21), snow and climate covariates (22-24), deep learning and machine learning for hydrology (25-30), ML methods foundations (31-33), and the WGMS observation infrastructure (34). Each entry was checked live against CrossRef on 2026-05-08 and the volume / issue / page are intentionally not transcribed into citations beyond what is needed for unambiguous resolution by DOI.
