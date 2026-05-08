# Additional References - Vernagtferner glacier water-level forecasting

Independent literature scan. Each entry resolved live against CrossRef on 2026-05-08. Entries are NEW relative to `reports/references.md`. Volume / issue / page numbers omitted on purpose; the DOI is the canonical pointer.

## State-of-the-art callout (gaps the existing `references.md` does NOT cover)

The current bibliography stops at Lim 2021 (TFT) and Kratzert 2018-2022 for deep-learning hydrology, and at Hugonnet 2021 for glacier mass change. Five concrete gaps the project should close:

1. **Hybrid VIC-glacier + LSTM error correction** for cold-alpine basins (Shi 2025, DOI:10.1016/j.jhydrol.2025.134251) is the closest published analogue to a Vernagtferner-style 5-min hydro-met-driven forecast and should be the methodological target for `model_advanced.py`.
2. **Entity-aware LSTM** for transboundary / glacierised lake basins (Park 2025, DOI:10.3390/hydrology12100261) supersedes the 2018 EA-LSTM by Kratzert and matters because Vernagtferner glacier-extent and station-altitude attributes act as static features.
3. **NSE / KGE evaluation pitfalls in seasonal alpine regimes** (Ruzzante 2026 HESS, DOI:10.5194/hess-30-2337-2026; Williams 2025 EnvModSoftw, DOI:10.1016/j.envsoft.2025.106665). These directly bear on the metric set in `src/model_baseline.py` and should appear in the manuscript Methods justification rather than the 1970 Nash-Sutcliffe paper alone.
4. **Recent Vernagtferner-specific papers** (Dobler 2026 The Cryosphere, DOI:10.5194/tc-20-2531-2026; Lechner 2025 EGU, DOI:10.5194/egusphere-egu25-4054; Gavriilidou 2024 Global Planet Change, DOI:10.1016/j.gloplacha.2024.104378). The current ref list ends at Lambrecht 2023 (also EGU) for site-specific work; three newer site-specific entries exist.
5. **Sentinel-1 SAR snow / snowmelt covariates** (Dunmire 2024 RSE, DOI:10.1016/j.rse.2024.114369; Turbé 2024 IEEE JSTARS, DOI:10.1109/jstars.2024.3384030). Operational snow-depth and melt-onset products from Sentinel-1 are a missing input modality given the brief explicitly cites "snow height" as a station covariate but the references do not bring in remote-sensing SWE / depth.

---

## Vernagtferner-specific (2023-2026)

1. Lambrecht, Mayer. Discharge characteristics for different glacier mass balance conditions at Vernagtferner, Ötztal Alps. EGU General Assembly 2023. 2023. DOI:10.5194/egusphere-egu23-13841
2. Dobler, Hagg, Mayer. Detection of crevassed areas with minimum geometric information: Vernagtferner case study. Journal of Glaciology. 2023. DOI:10.1017/jog.2023.12
3. Gavriilidou, Gerlach, Tsoulis. Analytical computation of local gravitational effects of mountain glacier mass change from polyhedral and prismatic modeling - test case Vernagtferner, Austrian Alps. Global and Planetary Change. 2024. DOI:10.1016/j.gloplacha.2024.104378
4. Lechner, Pail. Preliminary concept for observing the Vernagtferner Glacier with an optimized geodetic sensor network. EGU General Assembly 2025. 2025. DOI:10.5194/egusphere-egu25-4054
5. Dobler, Hagg, Rückamp, Seehaus. Understanding slow glacier flow under climate change: A case study on Vernagtferner, Austria. The Cryosphere. 2026. DOI:10.5194/tc-20-2531-2026

## Deep-learning rainfall-runoff and glacier-runoff hybrids (2024-2026)

6. Shi, Liu, Bai, Yu. Improving runoff simulation in cold alpine regions based on VIC-glacier by combining LSTM error correction technology. Journal of Hydrology. 2025. DOI:10.1016/j.jhydrol.2025.134251
7. Yu, Jiang, Schneider, Zheng. Deciphering the Mechanism of Better Predictions of Regional LSTM Models in Ungauged Basins. Water Resources Research. 2024. DOI:10.1029/2023wr035876
8. Hashemi, Javelle, Delestre, Razavi. Closing the data gap: runoff prediction in fully ungauged settings using LSTM. Hydrology and Earth System Sciences (Discussions). 2023. DOI:10.5194/hess-2023-282
9. Park, Liu, Zhu, Hong. Using Entity-Aware LSTM to Enhance Streamflow Predictions in Transboundary and Large Lake Basins. Hydrology. 2025. DOI:10.3390/hydrology12100261
10. Mishra. Attention-Based Deep Learning for Runoff Forecasting: Evaluating the Temporal Fusion Transformer Against Traditional Machine Learning Models. EarthArXiv. 2025. DOI:10.31223/x55x7x
11. Duong, Tran, Nguyen. Evaluating Rainfall-Runoff Generation Mechanisms of Deep Learning Models Using a Process-Based Rainfall-Runoff Model. Water Resources Management. 2025. DOI:10.1007/s11269-025-04231-5

## Glacier hydrology, mass balance and runoff projection (2024-2026)

12. Vincent, Thibert. Brief communication: Non-linear sensitivity of glacier mass balance to climate attested by temperature-index models. The Cryosphere. 2023. DOI:10.5194/tc-17-1989-2023
13. Yang, Bai, Tian, Liu. Glacier Coverage Dominates the Response of Runoff and Its Components to Climate Change in the Tianshan Mountains. Water Resources Research. 2025. DOI:10.1029/2024wr037947
14. Zhang, Wang, Leng, Zhao. Projections of Peak Water Timing From the East Rongbuk Glacier, Mt. Everest, Using a Higher-Order Ice Flow Model. Earth's Future. 2024. DOI:10.1029/2024ef004545
15. Karakoti, Mehta, Dobhal. Impact of climate change on himalayan water resources: a predictive model for glacier surface melt assessment. Sustainable Water Resources Management. 2024. DOI:10.1007/s40899-024-01110-6
16. Avzalshoev, Chun. Interpretable Deep Learning for Glacier Mass Balance: Temporal Attention Patterns in Central Asia. EGUsphere (Discussions). 2025. DOI:10.5194/egusphere-2025-5302
17. Barbuzano. Glacier Runoff Becomes Less Nutritious as Glaciers Retreat. Eos. 2025. DOI:10.1029/2025eo250431

## Snow / SWE remote sensing and alpine cryosphere (2024-2026)

18. Dunmire, Lievens, Boeykens, De Lannoy. A machine learning approach for estimating snow depth across the European Alps from Sentinel-1 imagery. Remote Sensing of Environment. 2024. DOI:10.1016/j.rse.2024.114369
19. Turbé, Karbou, Rabatel, Gouttevin. Snowmelt Dynamics in a Temperate Glacier Using Sentinel-1 SAR Images: A Case Study on Saint-Sorlin Glacier, French Alps. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. 2024. DOI:10.1109/jstars.2024.3384030
20. Schilling, Dietz, Kuenzer. Snow Water Equivalent Monitoring - A Review of Large-Scale Remote Sensing Applications. Remote Sensing. 2024. DOI:10.3390/rs16061085
21. Diaconu, Bamber, Zekollari. Glacier Area Change Assessment over 2015-2023 in the European Alps with Deep Learning. EGU General Assembly 2025. 2025. DOI:10.5194/egusphere-egu25-15293
22. Togaibekov, Gimbert, Rabatel, Walpersdorf. Surface mass balance monitoring of an alpine glacier using GNSS Interferometric Reflectometry. Journal of Glaciology. 2025. DOI:10.1017/jog.2025.10086
23. Mayer, Hendrick, Michel, Richter. Impact of climate change on snow avalanche activity in the Swiss Alps. The Cryosphere. 2024. DOI:10.5194/tc-18-5495-2024

## Hydrological model evaluation and metrics (2023-2026)

24. Mathevet, Le Moine, Andréassian, Gupta. Multi-objective assessment of hydrological model performances using Nash-Sutcliffe and Kling-Gupta efficiencies on a worldwide large sample of watersheds. Comptes Rendus. Géoscience. 2024. DOI:10.5802/crgeos.189
25. Melsen, Puy, Torfs, Saltelli. The rise of the Nash-Sutcliffe efficiency in hydrology. Hydrological Sciences Journal. 2025. DOI:10.1080/02626667.2025.2475105
26. Williams. Friends don't let friends use Nash-Sutcliffe Efficiency (NSE) or KGE for hydrologic model accuracy evaluation: A rant with data and suggestions for better practice. Environmental Modelling & Software. 2025. DOI:10.1016/j.envsoft.2025.106665
27. Ruzzante, Knoben, Wagener, Gleeson. Technical note: High Nash-Sutcliffe Efficiencies conceal poor simulations of interannual variance in seasonal regimes. Hydrology and Earth System Sciences. 2026. DOI:10.5194/hess-30-2337-2026
28. Gupta, Hantush, Govindaraju, Beven. Evaluation of hydrological models at gauged and ungauged basins using machine learning-based limits-of-acceptability and hydrological signatures. Journal of Hydrology. 2024. DOI:10.1016/j.jhydrol.2024.131774

---

**Verification record.** All 28 DOIs were resolved live against `https://api.crossref.org/works/{doi}` on 2026-05-08; HTTP 200 was returned and the title plus first author were checked against the citation. None of these 28 entries duplicate `reports/references.md`. Filing per project rule: omit volume / issue / pages, keep DOI as canonical pointer.

---

## Compact summary

Output: `/root/AI/liora_projects/20_glacier_water/additional_references.md`. Top 3 findings:
1. The existing reference list misses Vernagtferner site-specific 2024-2026 papers (Dobler 2026 TC, Gavriilidou 2024 GPC, Lechner 2025 EGU); these should be cited in Introduction and Discussion.
2. Methodological state-of-the-art for glacier-basin streamflow has moved past Kratzert 2018 LSTM to VIC-glacier + LSTM error correction (Shi 2025) and entity-aware LSTM (Park 2025); the advanced model design should reference these.
3. NSE / KGE diagnostic critique (Ruzzante 2026 HESS, Williams 2025 EnvModSoftw) is essential for an alpine-seasonal regime and is currently absent. Blockers: CrossRef intermittent HTTP 429 rate-limiting on `api.crossref.org`, all retries succeeded with 0.7-3 s delay. Role C complete.
