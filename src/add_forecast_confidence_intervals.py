# =============================================================================
# FORECAST CONFIDENCE INTERVALS - Statistical Rigor Enhancement
# Adds uncertainty bands to Module 6 ARIMA predictions
# 
# GOVERNMENT REQUIREMENT: "Show us the risk, not just the prediction"
# 
# ADDS:
# - 95% confidence intervals (±1.96 standard errors)
# - 80% confidence intervals (narrower band)
# - Probability of exceeding thresholds
# - Forecast reliability metrics
# 
# OUTPUT: Enhanced forecast visualization + uncertainty report
# =============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

print("📊 FORECAST CONFIDENCE INTERVALS - Adding Statistical Uncertainty")
print("="*80)
print("GOVERNMENT REQUIREMENT: Predictions must include uncertainty estimates")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================
class ConfidenceConfig:
    OUTPUT_DIR = Path('outputs')
    DATA_FILE = Path('data/processed/fused_aadhar_final.csv')
    
    # Output files
    FORECAST_CI_PLOT = OUTPUT_DIR / 'FORECAST_WITH_CONFIDENCE_INTERVALS.html'
    FORECAST_CI_DATA = OUTPUT_DIR / 'forecast_with_confidence_intervals.csv'
    UNCERTAINTY_REPORT = OUTPUT_DIR / 'FORECAST_UNCERTAINTY_REPORT.json'
    
    # ARIMA parameters (from Module 6)
    ARIMA_ORDER = (2, 1, 2)
    FORECAST_HORIZON = 90  # days
    
    # Confidence levels
    CONFIDENCE_LEVELS = [0.80, 0.95]  # 80% and 95%
    
    VERSION = '1.0.0'

CONFIG = ConfidenceConfig()

print(f"\n⚙️  Configuration:")
print(f"   • ARIMA Order: {CONFIG.ARIMA_ORDER}")
print(f"   • Forecast Horizon: {CONFIG.FORECAST_HORIZON} days")
print(f"   • Confidence Levels: {[f'{int(c*100)}%' for c in CONFIG.CONFIDENCE_LEVELS]}")
print(f"   • Version: {CONFIG.VERSION}")

# =============================================================================
# STAGE 1: LOAD AND PREPARE DATA
# =============================================================================
print("\n📂 STAGE 1: Loading Data for Forecasting...")

try:
    df = pd.read_csv(CONFIG.DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    print(f"✅ Loaded {len(df):,} records")
except FileNotFoundError:
    print("❌ Error: Data file not found")
    exit(1)

# Aggregate to daily level
daily_agg = df.groupby('date')['enrol_total'].sum().reset_index()
daily_agg = daily_agg.sort_values('date').reset_index(drop=True)

print(f"✅ Aggregated to {len(daily_agg)} daily periods")
print(f"   • Date Range: {daily_agg['date'].min().date()} to {daily_agg['date'].max().date()}")

# Prepare time series
ts = daily_agg.set_index('date')['enrol_total']

# =============================================================================
# STAGE 2: FIT ARIMA MODEL
# =============================================================================
print("\n📊 STAGE 2: Fitting ARIMA Model with Confidence Intervals...")

try:
    # Fit ARIMA
    model = ARIMA(ts, order=CONFIG.ARIMA_ORDER)
    fitted_model = model.fit()
    
    print(f"✅ ARIMA{CONFIG.ARIMA_ORDER} fitted")
    print(f"   • AIC: {fitted_model.aic:.1f}")
    print(f"   • BIC: {fitted_model.bic:.1f}")
    
    # Get forecast with confidence intervals
    forecast_result = fitted_model.get_forecast(steps=CONFIG.FORECAST_HORIZON)
    
    # Extract forecast and confidence intervals
    forecast_mean = forecast_result.predicted_mean
    forecast_ci_95 = forecast_result.conf_int(alpha=0.05)  # 95% CI
    forecast_ci_80 = forecast_result.conf_int(alpha=0.20)  # 80% CI
    
    # Get standard errors
    forecast_se = forecast_result.se_mean
    
    print(f"✅ Forecast generated with confidence intervals")
    print(f"   • Forecast Horizon: {len(forecast_mean)} days")
    print(f"   • 95% CI Width (avg): {(forecast_ci_95.iloc[:, 1] - forecast_ci_95.iloc[:, 0]).mean():,.0f}")
    print(f"   • 80% CI Width (avg): {(forecast_ci_80.iloc[:, 1] - forecast_ci_80.iloc[:, 0]).mean():,.0f}")
    
except Exception as e:
    print(f"❌ ARIMA fitting failed: {e}")
    exit(1)

# =============================================================================
# STAGE 3: CALCULATE UNCERTAINTY METRICS
# =============================================================================
print("\n📊 STAGE 3: Calculating Uncertainty Metrics...")

# Create forecast dataframe
forecast_dates = pd.date_range(
    start=ts.index[-1] + pd.Timedelta(days=1),
    periods=CONFIG.FORECAST_HORIZON,
    freq='D'
)

forecast_df = pd.DataFrame({
    'date': forecast_dates,
    'forecast': forecast_mean.values,
    'std_error': forecast_se.values,
    'lower_95': forecast_ci_95.iloc[:, 0].values,
    'upper_95': forecast_ci_95.iloc[:, 1].values,
    'lower_80': forecast_ci_80.iloc[:, 0].values,
    'upper_80': forecast_ci_80.iloc[:, 1].values
})

# Calculate additional metrics
forecast_df['coefficient_of_variation'] = (forecast_df['std_error'] / forecast_df['forecast']) * 100
forecast_df['uncertainty_ratio_95'] = (forecast_df['upper_95'] - forecast_df['lower_95']) / forecast_df['forecast']

# Calculate probability of exceeding historical max
historical_max = ts.max()
forecast_df['prob_exceed_historical_max'] = 1 - (
    (forecast_df['forecast'] - historical_max) / forecast_df['std_error']
).apply(lambda x: 0.5 * (1 + np.tanh(x / np.sqrt(2))))

print(f"✅ Uncertainty metrics calculated")
print(f"\n📊 Forecast Statistics:")
print(f"   • Mean Forecast: {forecast_df['forecast'].mean():,.0f}")
print(f"   • Avg Std Error: {forecast_df['std_error'].mean():,.0f}")
print(f"   • Avg CV: {forecast_df['coefficient_of_variation'].mean():.1f}%")
print(f"   • Historical Max: {historical_max:,.0f}")

# Identify high-uncertainty periods
high_uncertainty = forecast_df[forecast_df['coefficient_of_variation'] > 50]
print(f"   • High Uncertainty Days (CV > 50%): {len(high_uncertainty)}")

# =============================================================================
# STAGE 4: CREATE VISUALIZATION
# =============================================================================
print("\n🎨 STAGE 4: Creating Confidence Interval Visualization...")

# Create figure
fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(
    x=ts.index,
    y=ts.values,
    mode='lines',
    name='Historical Data',
    line=dict(color='steelblue', width=2)
))

# Forecast mean
fig.add_trace(go.Scatter(
    x=forecast_df['date'],
    y=forecast_df['forecast'],
    mode='lines',
    name='Forecast (Mean)',
    line=dict(color='darkred', width=3, dash='dash')
))

# 95% Confidence Interval
fig.add_trace(go.Scatter(
    x=forecast_df['date'],
    y=forecast_df['upper_95'],
    mode='lines',
    name='95% CI Upper',
    line=dict(width=0),
    showlegend=False,
    hovertemplate='Upper 95%: %{y:,.0f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=forecast_df['date'],
    y=forecast_df['lower_95'],
    mode='lines',
    name='95% Confidence Interval',
    fill='tonexty',
    fillcolor='rgba(255, 0, 0, 0.15)',
    line=dict(width=0),
    hovertemplate='Lower 95%: %{y:,.0f}<extra></extra>'
))

# 80% Confidence Interval
fig.add_trace(go.Scatter(
    x=forecast_df['date'],
    y=forecast_df['upper_80'],
    mode='lines',
    name='80% CI Upper',
    line=dict(width=0),
    showlegend=False,
    hovertemplate='Upper 80%: %{y:,.0f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=forecast_df['date'],
    y=forecast_df['lower_80'],
    mode='lines',
    name='80% Confidence Interval',
    fill='tonexty',
    fillcolor='rgba(255, 0, 0, 0.25)',
    line=dict(width=0),
    hovertemplate='Lower 80%: %{y:,.0f}<extra></extra>'
))

# Add historical max reference
fig.add_hline(
    y=historical_max,
    line_dash="dot",
    line_color="green",
    annotation_text=f"Historical Max: {historical_max:,.0f}",
    annotation_position="right"
)

# Update layout
fig.update_layout(
    title={
        'text': f'📊 ARIMA{CONFIG.ARIMA_ORDER} FORECAST WITH CONFIDENCE INTERVALS<br><sub>{CONFIG.FORECAST_HORIZON}-Day Forecast with 80% and 95% Uncertainty Bands</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    xaxis_title='Date',
    yaxis_title='Daily Enrollments',
    height=700,
    template='plotly_white',
    hovermode='x unified',
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.8)"
    )
)

# Save
fig.write_html(CONFIG.FORECAST_CI_PLOT)
print(f"✅ Saved: {CONFIG.FORECAST_CI_PLOT}")

# =============================================================================
# STAGE 5: SAVE OUTPUTS
# =============================================================================
print("\n💾 STAGE 5: Saving Forecast Data & Report...")

# Save forecast data
forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')
forecast_df.to_csv(CONFIG.FORECAST_CI_DATA, index=False)
print(f"✅ Saved: {CONFIG.FORECAST_CI_DATA}")

# Create uncertainty report
uncertainty_report = {
    'timestamp': datetime.now().isoformat(),
    'version': CONFIG.VERSION,
    'model': {
        'type': 'ARIMA',
        'order': CONFIG.ARIMA_ORDER,
        'aic': float(fitted_model.aic),
        'bic': float(fitted_model.bic)
    },
    'forecast_horizon': CONFIG.FORECAST_HORIZON,
    'confidence_intervals': {
        '95_percent': {
            'avg_width': float((forecast_ci_95.iloc[:, 1] - forecast_ci_95.iloc[:, 0]).mean()),
            'min_width': float((forecast_ci_95.iloc[:, 1] - forecast_ci_95.iloc[:, 0]).min()),
            'max_width': float((forecast_ci_95.iloc[:, 1] - forecast_ci_95.iloc[:, 0]).max())
        },
        '80_percent': {
            'avg_width': float((forecast_ci_80.iloc[:, 1] - forecast_ci_80.iloc[:, 0]).mean()),
            'min_width': float((forecast_ci_80.iloc[:, 1] - forecast_ci_80.iloc[:, 0]).min()),
            'max_width': float((forecast_ci_80.iloc[:, 1] - forecast_ci_80.iloc[:, 0]).max())
        }
    },
    'uncertainty_metrics': {
        'avg_std_error': float(forecast_df['std_error'].mean()),
        'avg_coefficient_of_variation_pct': float(forecast_df['coefficient_of_variation'].mean()),
        'high_uncertainty_days': len(high_uncertainty),
        'uncertainty_trend': 'Increasing' if forecast_df['std_error'].iloc[-1] > forecast_df['std_error'].iloc[0] else 'Stable'
    },
    'forecast_summary': {
        'mean_daily_forecast': float(forecast_df['forecast'].mean()),
        'min_forecast': float(forecast_df['forecast'].min()),
        'max_forecast': float(forecast_df['forecast'].max()),
        'historical_max': float(historical_max),
        'days_likely_exceeding_historical_max': int((forecast_df['prob_exceed_historical_max'] > 0.5).sum())
    },
    'reliability_assessment': {
        'short_term_reliability': 'HIGH' if forecast_df['coefficient_of_variation'].iloc[:30].mean() < 20 else 'MEDIUM',
        'long_term_reliability': 'MEDIUM' if forecast_df['coefficient_of_variation'].iloc[60:].mean() < 50 else 'LOW',
        'recommendation': 'Use forecast with caution beyond 60 days' if forecast_df['coefficient_of_variation'].iloc[60:].mean() > 50 else 'Forecast reliable for full horizon'
    },
    'key_insights': [
        f"95% confidence interval avg width: {(forecast_ci_95.iloc[:, 1] - forecast_ci_95.iloc[:, 0]).mean():,.0f} enrollments",
        f"Average forecast uncertainty (CV): {forecast_df['coefficient_of_variation'].mean():.1f}%",
        f"High uncertainty days (CV > 50%): {len(high_uncertainty)} of {CONFIG.FORECAST_HORIZON}",
        f"Forecast range: {forecast_df['forecast'].min():,.0f} to {forecast_df['forecast'].max():,.0f} daily enrollments",
        f"Historical max ({historical_max:,.0f}) likely exceeded on {(forecast_df['prob_exceed_historical_max'] > 0.5).sum()} days"
    ]
}

with open(CONFIG.UNCERTAINTY_REPORT, 'w') as f:
    json.dump(uncertainty_report, f, indent=2)
print(f"✅ Saved: {CONFIG.UNCERTAINTY_REPORT}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("🎉 FORECAST CONFIDENCE INTERVALS COMPLETE!")
print("="*80)

print(f"\n📊 FORECAST SUMMARY:")
print(f"   • Horizon: {CONFIG.FORECAST_HORIZON} days")
print(f"   • Mean Daily Forecast: {forecast_df['forecast'].mean():,.0f}")
print(f"   • 95% CI Width: {(forecast_ci_95.iloc[:, 1] - forecast_ci_95.iloc[:, 0]).mean():,.0f}")
print(f"   • 80% CI Width: {(forecast_ci_80.iloc[:, 1] - forecast_ci_80.iloc[:, 0]).mean():,.0f}")

print(f"\n📊 UNCERTAINTY METRICS:")
print(f"   • Avg Std Error: {forecast_df['std_error'].mean():,.0f}")
print(f"   • Avg CV: {forecast_df['coefficient_of_variation'].mean():.1f}%")
print(f"   • High Uncertainty Days: {len(high_uncertainty)} ({len(high_uncertainty)/CONFIG.FORECAST_HORIZON*100:.1f}%)")

print(f"\n✅ STATISTICAL RIGOR SATISFIED:")
print("   ✅ 95% and 80% confidence intervals")
print("   ✅ Standard error estimates")
print("   ✅ Coefficient of variation (uncertainty metric)")
print("   ✅ Probability of exceeding thresholds")
print("   ✅ Reliability assessment (short vs long term)")

print(f"\n📁 OUTPUT FILES:")
print(f"   • {CONFIG.FORECAST_CI_PLOT}")
print(f"   • {CONFIG.FORECAST_CI_DATA}")
print(f"   • {CONFIG.UNCERTAINTY_REPORT}")

print("\n🚀 Ready for Next Polish: Model Generalization Report!")
print("="*80)
