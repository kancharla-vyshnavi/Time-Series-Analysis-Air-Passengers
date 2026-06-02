import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style for premium look
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#fdfdfd'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['grid.color'] = '#eaeaea'
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

def load_data():
    print("--- 1. Loading and Preparing Data ---")
    filepath = "airline-passengers.csv"
    df = pd.read_csv(filepath, parse_dates=['Month'], index_col='Month')
    df.rename(columns={'Passengers': 'Passengers'}, inplace=True)
    print("Data Head:\n", df.head())
    print("Data Info:")
    df.info()
    return df

def resample_data(df):
    print("\n--- 2. Resampling to Quarterly ---")
    quarterly = df.resample('Q').mean()
    print("Quarterly Data Head:\n", quarterly.head())
    return quarterly

def decompose_series(df):
    print("\n--- 3. Trend/Seasonality Decomposition ---")
    # Multiplicative seasonal decomposition
    decomposition = seasonal_decompose(df['Passengers'], model='multiplicative', period=12)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Custom styling for premium look
    axes[0].plot(df.index, decomposition.observed, color='#1f77b4', linewidth=1.8, label='Observed')
    axes[0].set_title('Observed Series', fontsize=12, fontweight='bold', loc='left', pad=6)
    
    axes[1].plot(df.index, decomposition.trend, color='#e377c2', linewidth=1.8, label='Trend')
    axes[1].set_title('Trend Component (Long-term growth)', fontsize=12, fontweight='bold', loc='left', pad=6)
    
    axes[2].plot(df.index, decomposition.seasonal, color='#2ca02c', linewidth=1.5, label='Seasonal')
    axes[2].set_title('Seasonal Component (Yearly cycle)', fontsize=12, fontweight='bold', loc='left', pad=6)
    
    axes[3].scatter(df.index, decomposition.resid, color='#d62728', s=10, alpha=0.6, label='Residuals')
    axes[3].axhline(1, color='black', linestyle='--', linewidth=0.8)
    axes[3].set_title('Residuals (Noise/Irregularities)', fontsize=12, fontweight='bold', loc='left', pad=6)
    
    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('decomposition.png', dpi=300)
    plt.close()
    print("Decomposition plot saved as 'decomposition.png'.")
    
    # Calculate Insights
    trend = decomposition.trend.dropna()
    growth_ratio = trend.iloc[-1] / trend.iloc[0]
    print(f"Growth from start to end of trend: {growth_ratio:.2f}x ({((growth_ratio-1)*100):.1f}% growth)")
    
    # Analyze seasonal factor
    seasonal = decomposition.seasonal
    monthly_avg_seasonal = seasonal.groupby(seasonal.index.month).mean()
    peak_month = monthly_avg_seasonal.idxmax()
    trough_month = monthly_avg_seasonal.idxmin()
    print(f"Peak seasonal factor is in month {peak_month} (factor: {monthly_avg_seasonal[peak_month]:.3f})")
    print(f"Trough seasonal factor is in month {trough_month} (factor: {monthly_avg_seasonal[trough_month]:.3f})")
    return decomposition

def moving_averages(df):
    print("\n--- 4. Moving Averages Smoothing ---")
    df_ma = df.copy()
    df_ma['MA_6'] = df_ma['Passengers'].rolling(window=6).mean()
    df_ma['MA_12'] = df_ma['Passengers'].rolling(window=12).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_ma.index, df_ma['Passengers'], label='Actual Passengers', color='#1f77b4', alpha=0.4, linewidth=1.5)
    plt.plot(df_ma.index, df_ma['MA_6'], label='6-Month Moving Average (Seasonal Trend)', color='#ff7f0e', linestyle='--', linewidth=1.8)
    plt.plot(df_ma.index, df_ma['MA_12'], label='12-Month Moving Average (Cycles/Long-term Trend)', color='#d62728', linewidth=2.0)
    
    plt.title('Moving Averages Smoothing', fontsize=14, fontweight='bold', loc='left', pad=15)
    plt.xlabel('Year', fontsize=11, fontweight='bold')
    plt.ylabel('Passengers (in thousands)', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig('moving_averages.png', dpi=300)
    plt.close()
    print("Moving averages plot saved as 'moving_averages.png'.")
    return df_ma

def evaluate_models(df):
    print("\n--- 5. ARIMA/SARIMA Model Evaluation & Comparison ---")
    # Split data (last 12 months for testing)
    train = df.iloc[:-12]
    test = df.iloc[-12:]
    
    # 1. Fit Model A: Manual SARIMA(2, 1, 1)x(1, 1, 1, 12)
    print("Fitting Model A: Manual SARIMA(2, 1, 1)x(1, 1, 1, 12)...")
    model_a = SARIMAX(train['Passengers'], order=(2, 1, 1), seasonal_order=(1, 1, 1, 12))
    results_a = model_a.fit(disp=False)
    forecast_a = results_a.get_forecast(steps=12)
    mean_a = forecast_a.predicted_mean
    rmse_a = np.sqrt(mean_squared_error(test['Passengers'], mean_a))
    mape_a = mean_absolute_percentage_error(test['Passengers'], mean_a)
    
    # 2. Fit Model B: Auto-SARIMA search
    print("Running auto_arima to select Model B...")
    stepwise_fit = auto_arima(train['Passengers'], m=12, seasonal=True, trace=False)
    order_b = stepwise_fit.order
    seasonal_order_b = stepwise_fit.seasonal_order
    print(f"Auto-selected order: {order_b}, Seasonal order: {seasonal_order_b}")
    model_b = SARIMAX(train['Passengers'], order=order_b, seasonal_order=seasonal_order_b)
    results_b = model_b.fit(disp=False)
    forecast_b = results_b.get_forecast(steps=12)
    mean_b = forecast_b.predicted_mean
    rmse_b = np.sqrt(mean_squared_error(test['Passengers'], mean_b))
    mape_b = mean_absolute_percentage_error(test['Passengers'], mean_b)
    
    # 3. Fit Model C: Non-seasonal ARIMA(2, 1, 1)
    print("Fitting Model C: Non-seasonal ARIMA(2, 1, 1)...")
    model_c = SARIMAX(train['Passengers'], order=(2, 1, 1), seasonal_order=(0, 0, 0, 0))
    results_c = model_c.fit(disp=False)
    forecast_c = results_c.get_forecast(steps=12)
    mean_c = forecast_c.predicted_mean
    rmse_c = np.sqrt(mean_squared_error(test['Passengers'], mean_c))
    mape_c = mean_absolute_percentage_error(test['Passengers'], mean_c)
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Model Description':<40} | {'RMSE':<8} | {'MAPE':<8} | {'AIC':<8}")
    print("="*80)
    print(f"{'Model A: SARIMA(2,1,1)(1,1,1)12 [Manual]':<40} | {rmse_a:<8.2f} | {mape_a*100:<7.2f}% | {results_a.aic:<8.1f}")
    print(f"{f'Model B: SARIMA{order_b}{seasonal_order_b} [Auto]':<40} | {rmse_b:<8.2f} | {mape_b*100:<7.2f}% | {results_b.aic:<8.1f}")
    print(f"{'Model C: ARIMA(2,1,1) [Non-seasonal]':<40} | {rmse_c:<8.2f} | {mape_c*100:<7.2f}% | {results_c.aic:<8.1f}")
    print("="*80)
    
    # Plot forecast comparison vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(train.index[-24:], train['Passengers'].iloc[-24:], label='Training Data (Last 2 Years)', color='#7f7f7f', linewidth=1.5)
    plt.plot(test.index, test['Passengers'], label='Actual Passenger Count', color='#1f77b4', linewidth=2.5)
    plt.plot(test.index, mean_a, label=f'Model A: SARIMA(2,1,1)(1,1,1)12 (RMSE={rmse_a:.1f})', color='#ff7f0e', linestyle='--', linewidth=1.8)
    plt.plot(test.index, mean_b, label=f'Model B (Auto): SARIMA{order_b}{seasonal_order_b} (RMSE={rmse_b:.1f})', color='#2ca02c', linestyle='-.', linewidth=2.0)
    plt.plot(test.index, mean_c, label=f'Model C: Non-seasonal ARIMA(2,1,1) (RMSE={rmse_c:.1f})', color='#d62728', linestyle=':', linewidth=1.8)
    
    # Highlight the test partition
    plt.axvline(test.index[0], color='#888888', linestyle='-', alpha=0.5)
    plt.text(test.index[0], plt.ylim()[1]*0.95, ' Forecast Period Start', color='#666666', fontsize=9)
    
    plt.title('SARIMA Model Performance Comparison', fontsize=14, fontweight='bold', loc='left', pad=15)
    plt.xlabel('Date', fontsize=11, fontweight='bold')
    plt.ylabel('Passengers (in thousands)', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper left')
    plt.tight_layout()
    plt.savefig('forecast_vs_actual.png', dpi=300)
    plt.close()
    print("Forecast comparison plot saved as 'forecast_vs_actual.png'.")
    
    # Select best model for final projections
    if rmse_b < rmse_a:
        print("\nModel B (Auto-selected model) performs best and is chosen for future projections.")
        best_results = results_b
        best_order = order_b
        best_seasonal_order = seasonal_order_b
    else:
        print("\nModel A (Manual model) performs best and is chosen for future projections.")
        best_results = results_a
        best_order = (2, 1, 1)
        best_seasonal_order = (1, 1, 1, 12)
        
    return best_results, best_order, best_seasonal_order

def future_projections(df, best_order, best_seasonal_order):
    print("\n--- 6. Future Projections (1961) ---")
    print(f"Fitting final model SARIMA{best_order}x{best_seasonal_order} on the FULL dataset...")
    
    # Fit the best model on full data
    final_model = SARIMAX(df['Passengers'], order=best_order, seasonal_order=best_seasonal_order)
    final_results = final_model.fit(disp=False)
    
    # Forecast next 12 months (1961)
    future_forecast = final_results.get_forecast(steps=12)
    future_mean = future_forecast.predicted_mean
    future_ci = future_forecast.conf_int(alpha=0.05) # 95% CI
    
    # Print predictions
    print("\nProjected Monthly Passengers for 1961:")
    for date, val, ci_low, ci_high in zip(future_mean.index, future_mean, future_ci.iloc[:, 0], future_ci.iloc[:, 1]):
        print(f"{date.strftime('%Y-%m')}: {val:.1f} (95% CI: {ci_low:.1f} to {ci_high:.1f})")
    
    # Compounded Annual Growth Rate (CAGR) (1954 to 1960)
    # Check pandas resample rule for year-end to handle pandas versions cleanly
    try:
        yearly_avg = df['Passengers'].resample('YE').mean()
    except ValueError:
        yearly_avg = df['Passengers'].resample('Y').mean()
        
    cagr_54_60 = (yearly_avg.loc['1960'].values[0] / yearly_avg.loc['1954'].values[0]) ** (1/6) - 1
    print(f"\nCompounded Annual Growth Rate (CAGR) (1954-1960): {cagr_54_60 * 100:.2f}% YoY")
    
    # 1961 projected summary
    projected_mean_1961 = future_mean.mean()
    actual_mean_1960 = df.loc['1960', 'Passengers'].mean()
    projected_growth_1961 = (projected_mean_1961 / actual_mean_1960 - 1) * 100
    print(f"Projected 1961 Average Passengers: {projected_mean_1961:.1f}")
    print(f"Projected Growth Rate for 1961 vs 1960: {projected_growth_1961:.2f}%")
    
    # Plot history + 1961 projection
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Passengers'], label='Historical Data (1949-1960)', color='#1f77b4', linewidth=2.0)
    plt.plot(future_mean.index, future_mean, label='Projected Forecast (1961)', color='#ff7f0e', linestyle='-', linewidth=2.2)
    plt.fill_between(future_mean.index, future_ci.iloc[:, 0], future_ci.iloc[:, 1], color='#ff7f0e', alpha=0.15, label='95% Confidence Interval')
    
    plt.title('Air Passengers Forecast & Projections (1961)', fontsize=14, fontweight='bold', loc='left', pad=15)
    plt.xlabel('Year', fontsize=11, fontweight='bold')
    plt.ylabel('Passengers (in thousands)', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper left')
    plt.tight_layout()
    plt.savefig('future_forecast.png', dpi=300)
    plt.close()
    print("Future forecast plot saved as 'future_forecast.png'.")

def main():
    df = load_data()
    quarterly = resample_data(df)
    decompose_series(df)
    moving_averages(df)
    best_results, best_order, best_seasonal_order = evaluate_models(df)
    future_projections(df, best_order, best_seasonal_order)
    print("\nTime Series Analysis completed successfully!")

if __name__ == '__main__':
    main()
