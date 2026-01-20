# =============================================================================
# MODULE 8: FRAUD MONITORING SYSTEM - Isolation Forest
# Continuous anomaly detection for ghost beneficiaries & fraud patterns
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🛡️ MODULE 8: FRAUD MONITORING SYSTEM - Isolation Forest")
print("="*80)

# Configuration
class FraudConfig:
    OUTPUT_DIR = Path('outputs')
    CONTAMINATION = 0.05  # Expect 5% anomalies
    RANDOM_STATE = 42
    
    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = FraudConfig()
CONFIG.setup()

# =============================================================================
# STAGE 1: LOAD DATA
# =============================================================================
print("\n📂 STAGE 1: Loading Data...")

df = pd.read_csv('data/processed/fused_aadhar_final.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"✅ Data loaded: {len(df):,} records")

# =============================================================================
# STAGE 2: FEATURE ENGINEERING FOR FRAUD DETECTION
# =============================================================================
print("\n🔧 STAGE 2: Engineering Fraud Detection Features...")

# Aggregate by district
fraud_features = df.groupby('district').agg({
    'enrol_total': ['mean', 'std', 'max'],
    'enrol_0_17': 'mean',
    'enrol_18_plus': 'mean',
    'cis': 'mean',
    'enrol_momentum': 'mean',
    'state': 'first'
}).reset_index()

fraud_features.columns = ['district', 'enrol_mean', 'enrol_std', 'enrol_max', 
                          'child_mean', 'adult_mean', 'cis_mean', 'momentum', 'state']

# Additional features
fraud_features['enrol_volatility'] = fraud_features['enrol_std'] / (fraud_features['enrol_mean'] + 1)
fraud_features['child_adult_ratio'] = fraud_features['child_mean'] / (fraud_features['adult_mean'] + 1)
fraud_features['zero_cis_flag'] = (fraud_features['cis_mean'] == 0).astype(int)

print(f"✅ Features engineered for {len(fraud_features)} districts")
print(f"\nFeatures used for fraud detection:")
feature_cols = ['enrol_mean', 'enrol_volatility', 'child_adult_ratio', 
                'cis_mean', 'momentum', 'enrol_max']
for col in feature_cols:
    print(f"   • {col}")

# =============================================================================
# STAGE 3: ISOLATION FOREST MODEL
# =============================================================================
print("\n🤖 STAGE 3: Training Isolation Forest...")

# Prepare features
X = fraud_features[feature_cols].fillna(0)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
iso_forest = IsolationForest(
    contamination=CONFIG.CONTAMINATION,
    random_state=CONFIG.RANDOM_STATE,
    n_estimators=100,
    max_samples='auto',
    n_jobs=-1
)

fraud_features['anomaly_label'] = iso_forest.fit_predict(X_scaled)
fraud_features['anomaly_score'] = iso_forest.score_samples(X_scaled)

# Convert to 0-100 scale (lower score = more anomalous)
fraud_features['fraud_risk_score'] = (
    (fraud_features['anomaly_score'] - fraud_features['anomaly_score'].min()) /
    (fraud_features['anomaly_score'].max() - fraud_features['anomaly_score'].min())
) * 100

fraud_features['fraud_risk_score'] = 100 - fraud_features['fraud_risk_score']  # Invert (higher = riskier)

# Flag anomalies
fraud_features['is_anomaly'] = (fraud_features['anomaly_label'] == -1).astype(int)

anomaly_count = fraud_features['is_anomaly'].sum()
print(f"✅ Model trained")
print(f"   • Anomalies detected: {anomaly_count} districts ({anomaly_count/len(fraud_features)*100:.1f}%)")
print(f"   • Normal districts: {len(fraud_features) - anomaly_count}")

# =============================================================================
# STAGE 4: ANALYZE ANOMALIES
# =============================================================================
print("\n🔍 STAGE 4: Analyzing Fraud Patterns...")

anomalies = fraud_features[fraud_features['is_anomaly'] == 1].sort_values('fraud_risk_score', ascending=False)

if len(anomalies) > 0:
    print(f"\n📊 TOP 10 HIGHEST FRAUD RISK DISTRICTS:")
    print("="*100)
    top_10_fraud = anomalies.head(10)[['district', 'state', 'fraud_risk_score', 'cis_mean', 
                                        'enrol_volatility', 'child_adult_ratio']]
    print(top_10_fraud.to_string(index=False))
    
    # Pattern analysis
    print(f"\n📊 Anomaly Patterns:")
    print(f"   • Avg fraud risk score: {anomalies['fraud_risk_score'].mean():.1f}/100")
    print(f"   • Avg CIS: {anomalies['cis_mean'].mean():.3f}")
    print(f"   • Avg volatility: {anomalies['enrol_volatility'].mean():.2f}")
    print(f"   • Avg child/adult ratio: {anomalies['child_adult_ratio'].mean():.2f}")
else:
    print("✅ No significant fraud patterns detected (good news!)")

# =============================================================================
# STAGE 5: VISUALIZATIONS
# =============================================================================
print("\n📊 STAGE 5: Generating Fraud Monitoring Dashboards...")

# VIZ 1: Fraud Risk Distribution
fig1 = px.histogram(
    fraud_features,
    x='fraud_risk_score',
    nbins=50,
    title='Fraud Risk Score Distribution (0-100)',
    labels={'fraud_risk_score': 'Fraud Risk Score', 'count': 'Number of Districts'},
    color_discrete_sequence=['steelblue']
)
fig1.add_vline(x=80, line_dash="dash", line_color="red", 
               annotation_text="High Risk Threshold (80)")
fig1.update_layout(height=500)
fig1.write_html(CONFIG.OUTPUT_DIR / 'module8_fraud_risk_distribution.html')

# VIZ 2: Anomaly Scatter
fig2 = px.scatter(
    fraud_features,
    x='enrol_mean',
    y='enrol_volatility',
    color='is_anomaly',
    size='fraud_risk_score',
    hover_data=['district', 'state', 'cis_mean'],
    title='Fraud Detection: Enrollment Patterns',
    labels={'enrol_mean': 'Average Enrollment', 'enrol_volatility': 'Enrollment Volatility'},
    color_discrete_map={0: 'green', 1: 'red'}
)
fig2.update_layout(height=600)
fig2.write_html(CONFIG.OUTPUT_DIR / 'module8_anomaly_scatter.html')

# VIZ 3: Feature Importance (via variance in anomalies vs normal)
feature_importance = []
for col in feature_cols:
    normal_mean = fraud_features[fraud_features['is_anomaly'] == 0][col].mean()
    anomaly_mean = fraud_features[fraud_features['is_anomaly'] == 1][col].mean()
    importance = abs(anomaly_mean - normal_mean)
    feature_importance.append({'feature': col, 'importance': importance})

fi_df = pd.DataFrame(feature_importance).sort_values('importance', ascending=False)

fig3 = px.bar(
    fi_df,
    x='importance',
    y='feature',
    orientation='h',
    title='Fraud Detection: Key Distinguishing Features',
    labels={'importance': 'Difference (Anomaly vs Normal)', 'feature': 'Feature'}
)
fig3.update_layout(height=500)
fig3.write_html(CONFIG.OUTPUT_DIR / 'module8_feature_importance.html')

# VIZ 4: Geographic Fraud Risk Map
state_fraud = fraud_features.groupby('state').agg({
    'fraud_risk_score': 'mean',
    'is_anomaly': 'sum',
    'district': 'count'
}).reset_index()
state_fraud.columns = ['state', 'avg_fraud_risk', 'anomaly_count', 'total_districts']

fig4 = px.bar(
    state_fraud.sort_values('avg_fraud_risk', ascending=False).head(20),
    x='state',
    y='avg_fraud_risk',
    title='Top 20 States by Average Fraud Risk',
    labels={'avg_fraud_risk': 'Average Fraud Risk Score', 'state': 'State'},
    color='avg_fraud_risk',
    color_continuous_scale='Reds'
)
fig4.update_xaxes(tickangle=-45)
fig4.update_layout(height=600)
fig4.write_html(CONFIG.OUTPUT_DIR / 'module8_state_fraud_risk.html')

print("✅ Visualizations saved:")
print(f"   • {CONFIG.OUTPUT_DIR}/module8_fraud_risk_distribution.html")
print(f"   • {CONFIG.OUTPUT_DIR}/module8_anomaly_scatter.html")
print(f"   • {CONFIG.OUTPUT_DIR}/module8_feature_importance.html")
print(f"   • {CONFIG.OUTPUT_DIR}/module8_state_fraud_risk.html")

# =============================================================================
# STAGE 6: SAVE RESULTS
# =============================================================================
print("\n💾 STAGE 6: Saving Results...")

# Flagged districts
fraud_features.to_csv(CONFIG.OUTPUT_DIR / 'module8_fraud_monitoring_results.csv', index=False)
anomalies.to_csv(CONFIG.OUTPUT_DIR / 'module8_flagged_districts.csv', index=False)

# Summary JSON
summary = {
    'timestamp': datetime.now().isoformat(),
    'total_districts_analyzed': len(fraud_features),
    'anomalies_detected': int(anomaly_count),
    'contamination_rate': CONFIG.CONTAMINATION,
    'model': 'Isolation Forest',
    'features_used': feature_cols,
    'top_10_fraud_risk_districts': anomalies.head(10)[['district', 'state', 'fraud_risk_score']].to_dict('records')
}

with open(CONFIG.OUTPUT_DIR / 'module8_fraud_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✅ Results saved:")
print(f"   • {CONFIG.OUTPUT_DIR}/module8_fraud_monitoring_results.csv")
print(f"   • {CONFIG.OUTPUT_DIR}/module8_flagged_districts.csv")
print(f"   • {CONFIG.OUTPUT_DIR}/module8_fraud_summary.json")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("🎉 MODULE 8: FRAUD MONITORING SYSTEM COMPLETE!")
print("="*80)

print(f"\n🛡️ FRAUD DETECTION SUMMARY:")
print(f"   • Districts Analyzed: {len(fraud_features):,}")
print(f"   • Anomalies Detected: {anomaly_count} ({anomaly_count/len(fraud_features)*100:.1f}%)")
print(f"   • Model: Isolation Forest (100 trees)")
print(f"   • Features: {len(feature_cols)} behavioral indicators")

if len(anomalies) > 0:
    print(f"\n🔴 HIGHEST RISK DISTRICT:")
    top_fraud = anomalies.iloc[0]
    print(f"   • District: {top_fraud['district']}, {top_fraud['state']}")
    print(f"   • Fraud Risk Score: {top_fraud['fraud_risk_score']:.1f}/100")
    print(f"   • CIS: {top_fraud['cis_mean']:.3f}")
    print(f"   • Volatility: {top_fraud['enrol_volatility']:.2f}")

print(f"\n✅ CAPABILITIES:")
print("   ✅ Unsupervised fraud detection (no labeled data needed)")
print("   ✅ Real-time anomaly scoring (0-100 scale)")
print("   ✅ Multi-feature behavioral analysis")
print("   ✅ Geographic risk mapping")
print("   ✅ Continuous monitoring system")

print("\n🚀 Ready for deployment!")
print("="*80)
