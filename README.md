# Anomaly Detection in E-Commerce Order Pipelines

Forecasting order import volumes at 30-minute granularity using a 9-model ensemble comparison across statistical, tree-based, and neural architectures -- built for a Microsoft-scale anomaly detection system.

The idea: accurately forecast "normal" order volume, then flag deviations as anomalies. This project is the forecasting backbone.

<p align="center">
  <img src="docs/images/pipeline_architecture.png" width="750"/>
</p>

## Skills & Frameworks

**Languages**: Python

**Time Series**: AutoARIMA, MSTL, AutoTBATS, N-HiTS, LSTM, Temporal Fusion Transformer (via Nixtla `statsforecast` / `neuralforecast`)

**ML / Boosting**: LightGBM, XGBoost, CatBoost

**Signal Processing**: Hampel filter, Fourier feature engineering, Welch PSD, ACF/PACF analysis, MSTL decomposition

**Evaluation**: Rolling origin cross-validation (101 folds, full refit per fold)

**Execution**: GPU-sequential neural training, CPU-parallel statistical/tree models via `multiprocessing.Process`

## Summary

| | |
|---|---|
| **Domain** | E-commerce order pipeline monitoring (Order Import -> Fraud Check -> Authorized Order -> Pick Ticket) |
| **Data** | Half-hourly order volumes with daily (period=48) and weekly (period=336) seasonality, plus Black Friday spikes |
| **Preprocessing** | Hampel filter outlier removal, log transform, first-order differencing, Fourier exogenous features, holiday mask |
| **Models** | 9 models across 3 families -- statistical (AutoARIMA, MSTL+ARIMA, AutoTBATS), gradient-boosted trees (LightGBM, XGBoost, CatBoost), deep learning (N-HiTS, LSTM, TFT) |
| **Evaluation** | Rolling CV with 101 windows, 1680-obs training window, step size 53, horizon h=1 |
| **Execution** | GPU-bound neural models run sequentially; CPU-bound models run in parallel via `multiprocessing.Process` |

## How to Use

**EDA** -- `explore_order_import.ipynb` covers data loading, rolling statistics, ACF/PACF, periodogram/Welch PSD, MSTL decomposition, Hampel outlier removal, and feature engineering. Place raw data at `./dataset/order_imported.csv` with columns `ds` (datetime) and `y` (order count).

**Baseline forecasting** -- `python oi_baseline.py` runs all 9 models with rolling CV, writing results to `./baseline/cv_<model>.csv`. Neural models run first on GPU, then statistical and tree models run in parallel on CPU.

**Evaluation** -- `python eval.py` reverses log-differencing transforms, computes MSE/MAPE, and saves forecast-vs-actual plots to `./baseline/forecasts/`.

Key parameters:
```
WINDOW_SIZE = 1680        # 35 days at 30-min intervals
h = 1                     # single-step forecast
step_size = 53            # ~26.5 hours between CV folds
n_windows = 101           # number of CV folds
level = [50, 80, 95]      # confidence interval levels
```

## Preprocessing Pipeline

<p align="center">
  <img src="docs/images/preprocessing_flow.png" width="700"/>
</p>

1. **Outlier removal** -- Hampel filter at $5\sigma$, window=50. Conservative enough to preserve Black Friday spikes while removing system artifacts.
2. **Log transform** -- $y_{\text{log}} = \ln(y)$ stabilizes variance and converts multiplicative seasonality to additive.
3. **Differencing** -- $\Delta y_{\text{log}}$ removes the trend, producing a stationary series.
4. **Fourier exogenous features** -- Sine/cosine pairs at periods 48 (daily) and 336 (weekly), order 2 each (8 features). Encodes seasonal structure without requiring the model to discover it.
5. **Holiday mask** -- Binary feature for Black Friday weekend. Hand-crafted because the spikes (3-4x normal) distort model fits if unaccounted for.

**Per-model transform routing:**

| Model Family | Target Variable | Exogenous Features |
|---|---|---|
| MSTL + ARIMA | `y_log` | Holiday mask |
| AutoARIMA (SARIMAX) | `y_detrended` | Holiday mask, Fourier terms |
| AutoTBATS | Raw `y` | None (handles seasonality internally) |
| LightGBM / XGBoost / CatBoost | `y_log` | Holiday mask, Fourier (48, 336), lags [1, 2, 48, 336] |
| N-HiTS / LSTM / TFT | `y_detrended` | Holiday mask, `y_log`, Fourier (48, 336) |

## Models

**Statistical (CPU, parallelized)** -- AutoARIMA with seasonal period 48 on Fourier-augmented detrended series; MSTL with dual seasonality (48, 336) and AutoARIMA trend forecaster; AutoTBATS with trigonometric seasonality on raw series.

**Tree-Based (CPU, parallelized)** -- LightGBM, XGBoost, CatBoost with lag features at [1, 2, 48, 336] and Fourier exogenous features. Explicit lags are necessary because gradient-boosted trees cannot learn temporal dependencies natively.

**Neural (GPU, sequential)** -- N-HiTS with multi-rate downsampling [24, 12, 1] and MQLoss; LSTM as recurrent baseline; TFT with `hidden_size=16`, `n_head=2`, kept small to avoid overfitting on this dataset size.

<p align="center">
  <img src="docs/images/model_comparison.png" width="700"/>
</p>

## Cross-Validation Strategy

Every model is evaluated with **rolling origin cross-validation** (101 windows, `refit=True`), simulating production behavior where the model retrains periodically and forecasts the next observation.

<p align="center">
  <img src="docs/images/cv_strategy.png" width="700"/>
</p>

- **Training window**: 1680 observations (35 days) -- 5 full weekly cycles for reliable MSTL decomposition
- **Step size**: 53 (~26.5 hours between folds)
- **Horizon**: $h=1$ (single next-step forecast)
- **Total coverage**: ~111 days of test data across 101 folds

## Key Design Decisions

1. **Dual seasonality drove model selection.** ARIMA's single seasonal component can't capture both daily and weekly cycles -- MSTL was chosen specifically to decompose each seasonal component before forecasting.

2. **Hampel filter at $5\sigma$ for outlier handling.** Needed to remove system glitches without clipping Black Friday spikes (real signal). Verified by confirming holiday observations survived the filter.

3. **Log-differencing over simple scaling.** Variance scales with level (multiplicative seasonality). After log + diff, ACF residuals confirmed stationarity.

4. **GPU/CPU workload separation.** Neural models can't share GPU effectively, so they run sequentially. Statistical and tree models run in parallel via `multiprocessing.Process`, bypassing the GIL entirely.

5. **Conservative neural hyperparameters.** TFT at `hidden_size=16`, `n_head=2` -- anything larger overfit on this dataset. 101-fold CV already takes hours per model, so aggressive tuning was deprioritized.

6. **35-day window size chosen empirically.** Captures 5 full weekly cycles, giving MSTL and weekly Fourier terms enough data. Shorter windows degraded decomposition quality.
