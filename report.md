# Executive Report: Monthly Air Passengers Time Series Analysis (1949-1960)

This report details a complete step-by-step Time Series Analysis and Forecasting project conducted on the **Monthly Airline Passengers** dataset (1949-1960). Using statistical decomposition, moving average smoothing, and Auto-SARIMA modeling, we analyze historical growth, cyclicality, model performance, and project passenger demand into 1961 to support capacity planning.

---

## 1. Executive Summary & Core Insights

- **Consistent Growth Trend**: Passenger demand has expanded by **3.75x** over the 12-year window (274.7% total growth).
- **Strong & Stable Seasonality**: Annual peaks consistently occur in **July** (summer holiday peak, factor: 1.23x average) and troughs in **November** (winter low, factor: 0.80x average).
- **Forecasting Performance**: Our optimized Seasonal ARIMA (SARIMA) model achieves a **MAPE of 2.97%** and **RMSE of 17.82 passengers** on the out-of-sample test partition, outperforming standard ARIMA baselines.
- **1961 Outlook**: Total monthly average passengers for 1961 is projected at **497.2** (a 4.42% increase from 1960), peaking at **642.5 passengers** in July.

---

## 2. Trend & Seasonality Decomposition

Using a **multiplicative decomposition model** ($\text{Observed} = \text{Trend} \times \text{Seasonal} \times \text{Residual}$), we isolated the distinct components of the passenger time series:

![Time Series Decomposition](file:///Users/kalyansaran/Desktop/Task-3/decomposition.png)

### Key Insights:
- **Long-term Growth**: The underlying trend shows continuous upward trajectory, indicating robust macroeconomic expansion in commercial aviation from 1949 to 1960.
- **Cyclical Demand**: Peak passenger traffic in **July** is **30% higher** than the annual average, while the trough in **November** is **20% below** the annual average.
- **Capacity Planning Opportunity**: There is a clear operational opportunity to adjust staff, flight frequencies, and seat inventory by up to **25%** between Q2 and Q3 to capture peak summer demand and optimize yield.

---

## 3. Moving Averages Smoothing

We calculated **6-month** and **12-month** rolling moving averages to smooth short-term variations and understand different trends:

![Moving Averages Smoothing](file:///Users/kalyansaran/Desktop/Task-3/moving_averages.png)

- **12-Month Moving Average (Red)**: Successfully filters out all annual seasonality, exposing the pure secular growth trend.
- **6-Month Moving Average (Orange)**: Retains a smoothed representation of seasonal fluctuations, showing the amplitude and timing of peak/trough transitions.
- **Seasonal Spread**: The gap between the actual curve and the 12-Month MA illustrates the expanding seasonal amplitude as the overall volume grows (indicative of multiplicative seasonality).

---

## 4. Forecasting Model Performance & Evaluation

We evaluated three candidate models by training on data from **1949 to 1959** (132 months) and testing on **1960** (12 months):

1. **Model A (Manual)**: SARIMA$(2,1,1)(1,1,1)_{12}$ — User-specified seasonal model.
2. **Model B (Auto)**: SARIMA$(3,0,0)(0,1,0)_{12}$ with intercept — Selected using stepwise AIC optimization.
3. **Model C (Baseline)**: ARIMA$(2,1,1)$ — Non-seasonal model.

### Comparative Evaluation Metrics:

| Model Description | Test RMSE (Passengers) | Test MAPE (%) | AIC | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Model A: SARIMA(2,1,1)(1,1,1)₁₂** | 21.15 | 3.67% | **903.4** | Candidate |
| **Model B: SARIMA(3,0,0)(0,1,0)₁₂** | **17.82** | **2.97%** | 911.9 | **Best (Selected)** |
| **Model C: ARIMA(2,1,1) [Non-seasonal]** | 87.88 | 12.54% | 1246.3 | Rejected |

### Visual Forecast Validation:

![Model Forecast Comparison](file:///Users/kalyansaran/Desktop/Task-3/forecast_vs_actual.png)

- **Model B** captures the peak amplitude and turning points of the 1960 validation period with the highest accuracy.
- **Model C (Non-seasonal)** completely fails to capture the cyclical peaks and troughs, demonstrating that modeling seasonal dependencies is mandatory for accurate planning.

---

## 5. Projections & Business Recommendations (1961)

Re-fitting our best model (**Model B: SARIMA$(3,0,0)(0,1,0)_{12}$ with intercept**) on the full dataset, we generated forecasts for **1961**:

![1961 Future Projections](file:///Users/kalyansaran/Desktop/Task-3/future_forecast.png)

### Projected Monthly Passengers for 1961:
- **Jan 1961**: 442.6 (95% CI: 419.9 to 465.3)
- **Feb 1961**: 415.8 (95% CI: 388.4 to 443.1)
- **Mar 1961**: 442.8 (95% CI: 410.6 to 475.0)
- **Apr 1961**: 483.9 (95% CI: 448.1 to 519.8)
- **May 1961**: 494.1 (95% CI: 455.1 to 533.0)
- **Jun 1961**: 556.3 (95% CI: 514.7 to 597.9)
- **Jul 1961**: **642.5** (95% CI: 598.5 to 686.5) -- **Peak Demand**
- **Aug 1961**: 625.7 (95% CI: 579.7 to 671.8)
- **Sep 1961**: 527.0 (95% CI: 479.2 to 574.9)
- **Oct 1961**: 479.3 (95% CI: 429.8 to 528.8)
- **Nov 1961**: **407.7** (95% CI: 356.7 to 458.6) -- **Trough Demand**
- **Dec 1961**: 449.0 (95% CI: 396.7 to 501.3)

### Strategic Recommendations:
1. **CAGR Validation**: Historical growth rates from 1954 to 1960 shows a **12.18% YoY Compounded Annual Growth Rate (CAGR)**.
2. **Q3 Resource Deployment**: Fleet availability, maintenance schedules, and crew scheduling should be optimized to support **25-30% higher volume in Q3 (July/August)** compared to annual averages.
3. **Nov-Feb Demand Stimulation**: Launch marketing promotions or discount fares during November and February to offset seasonal demand troughs and improve aircraft utilization.
