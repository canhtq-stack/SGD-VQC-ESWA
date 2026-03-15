# Data Download Instructions

All data used in this study are sourced from publicly available repositories. **No proprietary data were used.**

---

## Required Dataset Format

The pipeline expects a single CSV file named `economic_data_collected.csv` in the project root directory (or specify a custom path with `--data`).

### Required columns

| Column | Description | Source | Indicator Code |
|--------|-------------|--------|----------------|
| `country` | Country name or ISO code | — | — |
| `year` | Year (integer, e.g. 1990–2023) | — | — |
| `gdp_growth` | GDP growth rate (annual %) | World Bank WDI | `NY.GDP.MKTP.KD.ZG` |
| `inflation` | CPI inflation (annual %) | World Bank WDI | `FP.CPI.TOTL.ZG` |
| `unemployment` | Unemployment, total (% of labor force) | World Bank WDI | `SL.UEM.TOTL.ZS` |
| `reserves_import_cover` | Total reserves in months of imports | IMF IFS | — |
| `trade_openness_pct_gdp` | Trade (% of GDP) | World Bank WDI | `NE.TRD.GNFS.ZS` |
| `government_expenditure_pct_gdp` | General gov. final consumption expenditure (% of GDP) | World Bank WDI | `NE.CON.GOVT.ZS` |

### Minimal example of CSV format

```
country,year,gdp_growth,inflation,unemployment,reserves_import_cover,trade_openness_pct_gdp,government_expenditure_pct_gdp
Vietnam,2000,6.79,1.71,2.27,3.2,96.53,6.11
Vietnam,2001,6.19,0.08,2.79,3.5,97.71,6.25
...
```

---

## Source 1 — World Bank World Development Indicators (WDI)

**URL:** https://databank.worldbank.org/source/world-development-indicators

### Steps to download:

1. Go to https://databank.worldbank.org/source/world-development-indicators
2. Click **"Open"** (or **"Create New Query"**)
3. **Country:** Select all developing countries in your sample
4. **Series:** Search and add the following indicator codes:
   - `NY.GDP.MKTP.KD.ZG` — GDP growth (annual %)
   - `FP.CPI.TOTL.ZG` — Inflation, consumer prices (annual %)
   - `SL.UEM.TOTL.ZS` — Unemployment, total (% of total labor force)
   - `NE.TRD.GNFS.ZS` — Trade (% of GDP)
   - `NE.CON.GOVT.ZS` — General government final consumption expenditure (% of GDP)
5. **Time:** Select years 1990–2023 (or the full available range)
6. Click **"Download"** → Select **CSV**
7. Reshape the downloaded file to the long format shown above (one row per country-year)

---

## Source 2 — IMF International Financial Statistics (IFS)

**URL:** https://data.imf.org/

### Steps to download `reserves_import_cover`:

1. Go to https://data.imf.org/
2. Select **"International Financial Statistics (IFS)"**
3. Search for **"Reserve Assets"** → look for **"Total Reserves in Months of Imports"**
   - Alternatively: compute as `Total Reserves (excl. gold, USD) / (Annual Imports / 12)`
   - WDI alternative: `FI.RES.TOTL.MO` (Total reserves in months of imports)
4. Select countries and years matching your WDI download
5. Export as CSV and merge on `country` + `year`

> **Tip:** The indicator `FI.RES.TOTL.MO` is also available in WDI, which simplifies the download to a single source.

---

## Merging the Data

After downloading, merge the two sources on `country` and `year`:

```python
import pandas as pd

wdi = pd.read_csv("wdi_raw.csv")       # After reshaping to long format
ifs = pd.read_csv("ifs_reserves.csv")  # reserves_import_cover

df = pd.merge(wdi, ifs, on=["country", "year"], how="left")
df.to_csv("economic_data_collected.csv", index=False)
```

---

## Country Sample

The paper covers **developing countries** as classified by the World Bank (low-income and middle-income). The time span is **1990–2022**, with the temporal split:

- **Training set:** ≤ 2018
- **Test set:** ≥ 2019

---

## Missing Data Handling

The pipeline handles missing values automatically:

1. Country-level median imputation (within each country's time series)
2. Global median fallback for countries with insufficient data

No pre-processing of missing values is required before running the pipeline.
