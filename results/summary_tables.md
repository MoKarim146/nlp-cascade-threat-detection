# Result Summary Tables

These tables are generated from saved repository outputs. Missing metrics are left blank; no values are invented.

## Main Cascade

| Dataset | System | Macro F1 | FNR | Tier 1 % | RoBERTa % | Tier 3 % |
| --- | --- | --- | --- | --- | --- | --- |
| HateXplain | TF-IDF standalone | 0.7739 | 0.2224 | 100.00 | 0.0000 |  |
| HateXplain | RoBERTa standalone | 0.7911 | 0.1594 | 0.0000 | 100.00 |  |
| HateXplain | Best-F1 TF-IDF -> RoBERTa cascade | 0.7942 | 0.1734 | 62.84 | 37.16 |  |
| HateXplain | Safety-constrained TF-IDF -> RoBERTa cascade | 0.7913 | 0.1497 | 59.98 | 40.02 |  |
| OLID | TF-IDF standalone | 0.7282 | 0.3583 | 100.00 | 0.0000 |  |
| OLID | RoBERTa standalone | 0.8105 | 0.2208 | 0.0000 | 100.00 |  |
| OLID | Best-F1 TF-IDF -> RoBERTa cascade | 0.8090 | 0.2667 | 51.74 | 48.26 |  |
| OLID | Safety-constrained TF-IDF -> RoBERTa cascade | 0.8059 | 0.2083 | 38.37 | 61.63 |  |

## Source Files

- Curated key numbers: `results/key_metrics_source.csv`
