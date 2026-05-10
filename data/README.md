# Data sources - Vernagtferner glacier water-level forecasting

The brief points at three classes of public datasets. None of them ships as a single bulk download; each requires per-source registration or per-record fetching. **No bulk download performed in v1.0.** This file documents the URLs, the formats, and the exact access pattern so the main session can pull what is needed at execution time.

## Folder layout (created at fetch time)

```
data/
  raw/
    pangaea/                  # PANGAEA NetCDF / TSV downloads
    wgms/                     # WGMS Fluctuations of Glaciers tables
    glamos/                   # GLAMOS Swiss reference glaciers (cross-validation)
    pegelportal_tirol/        # Tirol hydrography portal extracts
    badw/                     # BAdW Vernagtferner programme extracts
    dwd/                      # Deutscher Wetterdienst climate stations
    meteoswiss/               # MeteoSwiss IDAweb (cross-border alpine context)
  interim/                    # Cleaned per-station daily / 5-min CSVs
  processed/                  # Modelling-ready table (one row per timestamp)
```

## Primary source - PANGAEA (Vernagtferner discharge)

- DOI: `10.1594/PANGAEA.829530`
- Direct landing page: https://doi.pangaea.de/10.1594/PANGAEA.829530
- Coverage: Vernagtbach Pegelstation discharge, daily resolution. The brief notes that the 2013-2024 5-minute record is "to be published" on pangaea.de; check for new DOIs under the Escher-Vetter and BAdW group profile.
- Format: TSV with PANGAEA standard header. Columns include date/time, water level (cm), discharge (m3/s), and station metadata.
- Programmatic fetch:
  ```bash
  # Direct CSV/TSV is exposed by pangaea.de when the dataset is open access.
  curl -L -o data/raw/pangaea/PANGAEA.829530.tab \
       "https://doi.pangaea.de/10.1594/PANGAEA.829530?format=textfile"
  ```
- License: CC BY for most BAdW releases; confirm per-DOI before redistribution.

## BAdW Vernagtferner long-term monitoring programme

- Programme home: https://geo.badw.de/en/the-project.html
- Data products of interest: 5-minute hydro-meteorological records 2013-2024 (cited in the brief), annual mass-balance summaries, glacier-extent shapefiles.
- Access: most products require a research-use registration form to BAdW Kommission fur Erdmessung und Glaziologie. The PANGAEA mirror is the open-access entry point; pre-publication 5-minute data will need a direct request.
- Reference contacts and documentation are linked from the programme page.

## WGMS Fluctuations of Glaciers (annual extent and mass balance)

- Portal: https://wgms.ch/data_databaseversions/
- Login: free academic registration, then bulk CSV/JSON download per database release.
- Vernagtferner WGMS ID: VERNAGTFERNER (Austria, RGI region 11). Pull the FoG-MB (mass balance), FoG-FV (front variation), and FoG-LDA (length-diameter-area) tables for the years that overlap with the hydro record.
- Shipped as zipped CSVs; total < 200 MB.
- Programmatic fetch (after login):
  ```bash
  # WGMS does not expose an open API; the typical pattern is to download
  # the ZIP from the portal once (e.g. DOI:10.5904/wgms-fog-2024-01) and
  # unpack into data/raw/wgms/.
  ```

## GLAMOS - Swiss reference glaciers (cross-validation cohort)

- Portal: https://www.glamos.ch/en/data/
- Coverage: Hintereisferner, Kesselwandferner, and other Swiss / Austrian alpine reference glaciers with overlapping climate regime.
- Use case: out-of-distribution sanity check for the LSTM. A model trained on Vernagtferner alone risks overfitting to one catchment; GLAMOS gives a cohort to test transferability.
- Format: per-glacier daily CSVs.

## Pegelportal Tirol (Austrian hydrography service)

- Portal: https://hydro.tirol.gv.at/
- Coverage: live water levels and discharge across Tirol, including stations downstream of Vernagtbach (Rofenache, Otztal Ache).
- Use case: extending the upstream signal to a downstream context for catchment-level validation.
- Format: HTML widgets and CSV exports. Programmatic fetch is per-station and rate-limited.

## DWD (Deutscher Wetterdienst) climate covariates

- Portal: https://www.dwd.de/EN/climate_environment/cdc/cdc_node.html
- Open Data API: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/
- Stations of interest: Zugspitze (extreme-altitude reference, 2962 m), Wendelstein, Hohenpeissenberg.
- Variables: air temperature (5-min and hourly), precipitation, snow depth, relative humidity, global radiation, wind speed and direction.
- Programmatic fetch:
  ```bash
  # 5-min air temperature, station-by-station ZIP archive
  curl -L -o data/raw/dwd/zugspitze_air_temp_5min.zip \
       "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/5_minutes/air_temperature/historical/5minutenwerte_TU_05792_19920101_20221231_hist.zip"
  ```
  (URL pattern is `5minutenwerte_<param>_<station-id>_<from>_<to>_<state>.zip`; check the index page for the latest station IDs.)
- License: CC BY 4.0.

## MeteoSwiss IDAweb (cross-border alpine context, optional)

- Portal: https://gate.meteoswiss.ch/idaweb/
- Coverage: Swiss high-altitude stations near the Vernagt catchment (e.g. Saas Almagell, Weissfluhjoch).
- Access: free academic account, then per-request CSV by email or download. Not bulk-fetchable.
- Use case: cross-border meteorological covariates for ablation-season modelling.

## Size and download policy (per Portfolio rules)

| Source | Estimated size | Download in v1.0? |
|--------|---------------|----------------------|
| PANGAEA 829530 (daily 2002-2012) | < 5 MB | document only - main session pulls when modelling |
| BAdW 5-min 2013-2024 | ~ 1-2 GB raw | document only - registration-gated |
| WGMS FoG bundle | < 200 MB | document only - login-gated |
| GLAMOS daily | < 100 MB | document only |
| DWD 5-min stations | ~ 500 MB per station per variable | document only |
| MeteoSwiss IDAweb | request-based | document only |

Per the rules, datasets behind login or > 500 MB are documented rather than downloaded. The user runs the actual fetches in the main session.

## Provenance and licensing notes

- BAdW programme outputs (PANGAEA-mirrored portion): typically CC BY 4.0; per-DOI check still required.
- DWD: CC BY 4.0.
- WGMS: per-version licence statement, generally CC BY 4.0.
- MeteoSwiss: research-use only, citation required.
- GLAMOS: CC BY-SA 4.0.
- Cite BAdW Kommission fur Erdmessung und Glaziologie for the Vernagtferner programme regardless of which mirror is used.
