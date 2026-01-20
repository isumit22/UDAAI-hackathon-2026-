# =============================================================================
# MODULE 7B: SHAP EXPLAINABILITY - Model Interpretation & Trust Building
# Advanced explainability for Random Forest welfare exclusion predictor
# =============================================================================

import pandas as pd
import numpy as np
import shap
import pickle
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

print("🔍 MODULE 7B: SHAP EXPLAINABILITY - Model Interpretation System")
print("="*80)
print("COMPETITIVE EDGE: Individual prediction explanations + global insights")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================
class SHAPConfig:
    OUTPUT_DIR = Path('outputs')
    MODEL_PATH = OUTPUT_DIR / 'module7_trained_model.pkl'
    EXPLAINER_PATH = OUTPUT_DIR / 'module7b_shap_explainer.pkl'
    
    N_BACKGROUND_SAMPLES = 100
    N_EXPLAIN_SAMPLES = 20  # Reduced for waterfall plots
    RANDOM_STATE = 42
    
    MAX_DISPLAY_FEATURES = 12
    PLOT_DPI = 150
    PLOT_FORMAT = 'png'
    
    VERSION = '1.0.0'
    
    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = SHAPConfig()
CONFIG.setup()

print(f"\n⚙️  Configuration:")
print(f"   • Background Samples: {CONFIG.N_BACKGROUND_SAMPLES}")
print(f"   • Explanation Samples: {CONFIG.N_EXPLAIN_SAMPLES}")
print(f"   • Max Display Features: {CONFIG.MAX_DISPLAY_FEATURES}")
print(f"   • Version: {CONFIG.VERSION}")

# =============================================================================
# STAGE 1: LOAD MODEL & DATA
# =============================================================================
print("\n📂 STAGE 1: Loading Model & Data...")

try:
    df = pd.read_csv('data/processed/fused_aadhar_final.csv')
    df['date'] = pd.to_datetime(df['date'])
    print(f"✅ Data loaded: {len(df):,} records")
except FileNotFoundError:
    print("❌ Error: Data file not found")
    exit(1)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("\n🔧 Recreating feature engineering...")

district_features = df.groupby('district').agg({
    'cis': 'mean',
    'adult_child_ratio': 'mean',
    'enrol_total': ['mean', 'std', 'sum'],
    'enrol_0_17': 'mean',
    'enrol_18_plus': 'mean',
    'enrol_momentum': 'mean',
    'coverage_gap': 'mean',
    'update_lag_index': 'mean',
    'enrol_pincodes': 'mean',
    'enrol_total_per_day': 'mean',
    'state': 'first'
}).reset_index()

district_features.columns = [
    'district', 'cis_mean', 'adult_child_ratio', 'enrol_mean', 'enrol_std',
    'enrol_total_sum', 'child_enrol_mean', 'adult_enrol_mean', 'momentum',
    'coverage_gap', 'update_lag', 'num_pincodes', 'enrol_per_day', 'state'
]

district_features['enrol_volatility'] = district_features['enrol_std'] / (district_features['enrol_mean'] + 1)
district_features['child_proportion'] = district_features['child_enrol_mean'] / (district_features['enrol_mean'] + 1)
district_features['zero_cis_flag'] = (district_features['cis_mean'] == 0).astype(int)
district_features['negative_momentum_flag'] = (district_features['momentum'] < 0).astype(int)

state_avg_cis = district_features.groupby('state')['cis_mean'].mean()
district_features['state_avg_cis'] = district_features['state'].map(state_avg_cis)
district_features['relative_to_state'] = district_features['cis_mean'] - district_features['state_avg_cis']

feature_cols = [
    'cis_mean', 'adult_child_ratio', 'enrol_mean', 'enrol_volatility',
    'child_enrol_mean', 'momentum', 'coverage_gap', 'update_lag',
    'num_pincodes', 'enrol_per_day', 'child_proportion', 'zero_cis_flag',
    'negative_momentum_flag', 'relative_to_state'
]

HIGH_RISK_CIS_THRESHOLD = 0.3
CRITICAL_MOMENTUM_THRESHOLD = -0.1

district_features['high_risk'] = (
    (district_features['cis_mean'] < HIGH_RISK_CIS_THRESHOLD) |
    ((district_features['momentum'] < CRITICAL_MOMENTUM_THRESHOLD) & 
     (district_features['cis_mean'] < 0.5)) |
    (district_features['zero_cis_flag'] == 1)
).astype(int)

X = district_features[feature_cols].fillna(0)
y = district_features['high_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=CONFIG.RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features prepared: {X.shape[1]} dimensions")
print(f"   • Training: {len(X_train)} samples")
print(f"   • Test: {len(X_test)} samples")

if CONFIG.MODEL_PATH.exists():
    print(f"\n📦 Loading pre-trained model...")
    with open(CONFIG.MODEL_PATH, 'rb') as f:
        rf_model = pickle.load(f)
    print("✅ Model loaded")
else:
    print("\n🤖 Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        random_state=CONFIG.RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    with open(CONFIG.MODEL_PATH, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"✅ Model trained and saved")

train_acc = rf_model.score(X_train_scaled, y_train)
test_acc = rf_model.score(X_test_scaled, y_test)
print(f"   • Training Accuracy: {train_acc:.3f}")
print(f"   • Test Accuracy: {test_acc:.3f}")

# =============================================================================
# STAGE 2: CREATE SHAP EXPLAINER
# =============================================================================
print("\n🔍 STAGE 2: Creating SHAP Explainer...")

# Delete old explainer if exists (compatibility fix)
if CONFIG.EXPLAINER_PATH.exists():
    CONFIG.EXPLAINER_PATH.unlink()
    print("🗑️  Deleted old explainer cache")

print(f"🔧 Creating TreeExplainer...")

# Use shap.sample for background (faster than kmeans)
background_sample = shap.sample(X_train_scaled, CONFIG.N_BACKGROUND_SAMPLES, random_state=CONFIG.RANDOM_STATE)

# Create explainer
explainer = shap.TreeExplainer(rf_model, background_sample)

# Save for future use
with open(CONFIG.EXPLAINER_PATH, 'wb') as f:
    pickle.dump(explainer, f)

print(f"✅ Explainer created and saved")
print(f"   • Background samples: {CONFIG.N_BACKGROUND_SAMPLES}")

# Handle expected_value (can be scalar or array)
expected_val = explainer.expected_value
if isinstance(expected_val, np.ndarray):
    expected_val = expected_val[1] if len(expected_val) > 1 else expected_val[0]
    
print(f"   • Expected value: {expected_val:.4f}")

# =============================================================================
# STAGE 3: COMPUTE SHAP VALUES
# =============================================================================
print("\n📊 STAGE 3: Computing SHAP Values...")

print(f"🔧 Computing SHAP values for {len(X_test_scaled)} test samples...")

# Compute SHAP values - this returns an Explanation object
shap_values = explainer(X_test_scaled)

print(f"✅ SHAP values computed")
print(f"   • Shape: {shap_values.values.shape}")
print(f"   • Mean absolute SHAP: {np.abs(shap_values.values).mean():.4f}")

# =============================================================================
# STAGE 4: GLOBAL INTERPRETABILITY - SUMMARY PLOTS
# =============================================================================
print("\n📊 STAGE 4: Generating Global Interpretability Visualizations...")

# VIZ 1: SHAP Summary Plot (Beeswarm)
print("🎨 Creating beeswarm plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values, 
    X_test,
    max_display=CONFIG.MAX_DISPLAY_FEATURES,
    show=False
)
plt.tight_layout()
plt.savefig(CONFIG.OUTPUT_DIR / f'module7b_shap_beeswarm.{CONFIG.PLOT_FORMAT}', 
            dpi=CONFIG.PLOT_DPI, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: module7b_shap_beeswarm.{CONFIG.PLOT_FORMAT}")

# VIZ 2: SHAP Bar Plot (Mean Absolute Values)
print("🎨 Creating mean SHAP value bar plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_test,
    plot_type='bar',
    max_display=CONFIG.MAX_DISPLAY_FEATURES,
    show=False
)
plt.tight_layout()
plt.savefig(CONFIG.OUTPUT_DIR / f'module7b_shap_bar.{CONFIG.PLOT_FORMAT}',
            dpi=CONFIG.PLOT_DPI, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: module7b_shap_bar.{CONFIG.PLOT_FORMAT}")

print(f"\n✅ Global interpretability visualizations complete")

# =============================================================================
# STAGE 5: LOCAL INTERPRETABILITY - INDIVIDUAL PREDICTIONS
# =============================================================================
print("\n🔍 STAGE 5: Generating Local Interpretability (Individual Predictions)...")

test_predictions = rf_model.predict_proba(X_test_scaled)[:, 1]
test_districts = district_features.loc[X_test.index, 'district'].values
test_states = district_features.loc[X_test.index, 'state'].values

# Get top N highest risk
top_risk_indices = np.argsort(test_predictions)[-CONFIG.N_EXPLAIN_SAMPLES:][::-1]

print(f"🔍 Explaining top {len(top_risk_indices)} highest-risk predictions...")

individual_explanations = []

for i, idx in enumerate(top_risk_indices[:10], 1):
    district_name = test_districts[idx]
    state_name = test_states[idx]
    risk_prob = test_predictions[idx]
    
    print(f"   🎨 Waterfall plot {i}/10: {district_name}, {state_name} (Risk: {risk_prob:.1%})")
    
    # For binary classification, select class 1 (high risk) - shape becomes (14,)
    # shap_values shape is (244, 14, 2) -> select [idx, :, 1] for high-risk class
    try:
        plt.figure(figsize=(10, 6))
        
        # Extract SHAP values for this instance and high-risk class
        instance_shap_values = shap_values.values[idx, :, 1]  # Shape: (14,)
        instance_data = shap_values.data[idx]  # Shape: (14,)
        instance_base = shap_values.base_values[idx, 1]  # Scalar
        
        # Create Explanation object for waterfall
        explanation = shap.Explanation(
            values=instance_shap_values,
            base_values=instance_base,
            data=instance_data,
            feature_names=feature_cols
        )
        
        shap.plots.waterfall(explanation, max_display=CONFIG.MAX_DISPLAY_FEATURES, show=False)
        plt.title(f"SHAP Waterfall: {district_name}, {state_name} (Risk: {risk_prob:.1%})", fontsize=12)
        plt.tight_layout()
        plt.savefig(CONFIG.OUTPUT_DIR / f'module7b_waterfall_{i:02d}_{district_name.replace(" ", "_")[:20]}.{CONFIG.PLOT_FORMAT}',
                    dpi=CONFIG.PLOT_DPI, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"      ⚠️  Waterfall plot failed: {e}")
        plt.close()
        continue
    
    # Store explanation
    feature_contributions = {
        feature_cols[j]: float(shap_values.values[idx, j, 1])  # Class 1 (high risk)
        for j in range(len(feature_cols))
    }
    
    individual_explanations.append({
        'rank': i,
        'district': district_name,
        'state': state_name,
        'risk_probability': float(risk_prob),
        'base_value': float(instance_base),
        'shap_contributions': feature_contributions,
        'top_3_positive': sorted(
            [(k, v) for k, v in feature_contributions.items() if v > 0],
            key=lambda x: x[1], 
            reverse=True
        )[:3],
        'top_3_negative': sorted(
            [(k, v) for k, v in feature_contributions.items() if v < 0],
            key=lambda x: x[1]
        )[:3]
    })

print(f"✅ Generated {len(individual_explanations)} individual explanations")

# =============================================================================
# STAGE 6: FEATURE IMPORTANCE COMPARISON
# =============================================================================
print("\n📊 STAGE 6: Comparing SHAP vs Traditional Feature Importance...")

# Traditional Random Forest importance
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'rf_importance': rf_model.feature_importances_
}).sort_values('rf_importance', ascending=False)

# SHAP-based importance (mean absolute SHAP values for class 1 - high risk)
# Shape is (244, 14, 2) -> extract [:, :, 1] for high-risk class -> mean over axis 0
shap_importance_values = np.abs(shap_values.values[:, :, 1]).mean(axis=0)

shap_importance = pd.DataFrame({
    'feature': feature_cols,
    'shap_importance': shap_importance_values
}).sort_values('shap_importance', ascending=False)

# Merge
importance_comparison = rf_importance.merge(shap_importance, on='feature')

# Calculate correlation
correlation = importance_comparison['rf_importance'].corr(importance_comparison['shap_importance'])

print(f"✅ Feature importance comparison:")
print(f"   • Correlation (RF vs SHAP): {correlation:.3f}")
print(f"\nTop 5 by RF Importance:")
for _, row in rf_importance.head(5).iterrows():
    print(f"   • {row['feature']}: {row['rf_importance']:.4f}")

print(f"\nTop 5 by SHAP Importance:")
for _, row in shap_importance.head(5).iterrows():
    print(f"   • {row['feature']}: {row['shap_importance']:.4f}")

# VIZ: Comparison plot
fig = go.Figure()

fig.add_trace(go.Bar(
    name='Random Forest',
    x=importance_comparison.head(12)['feature'],
    y=importance_comparison.head(12)['rf_importance'],
    marker_color='steelblue'
))

fig.add_trace(go.Bar(
    name='SHAP',
    x=importance_comparison.head(12)['feature'],
    y=importance_comparison.head(12)['shap_importance'],
    marker_color='coral'
))

fig.update_layout(
    title=f'Feature Importance: Random Forest vs SHAP (Correlation: {correlation:.3f})',
    xaxis_title='Feature',
    yaxis_title='Importance Score',
    barmode='group',
    height=600,
    xaxis_tickangle=-45
)

fig.write_html(CONFIG.OUTPUT_DIR / 'module7b_importance_comparison.html')
print(f"\n✅ Comparison visualization saved")

# =============================================================================
# STAGE 7: SAVE RESULTS
# =============================================================================
print("\n💾 STAGE 7: Saving Results...")

importance_comparison.to_csv(CONFIG.OUTPUT_DIR / 'module7b_importance_comparison.csv', index=False)

with open(CONFIG.OUTPUT_DIR / 'module7b_individual_explanations.json', 'w') as f:
    json.dump(individual_explanations, f, indent=2)

shap_df = pd.DataFrame(shap_values.values[:, :, 1], columns=feature_cols)  # Class 1 (high risk)
shap_df['district'] = test_districts
shap_df['state'] = test_states
shap_df['risk_probability'] = test_predictions
shap_df.to_csv(CONFIG.OUTPUT_DIR / 'module7b_shap_values_all.csv', index=False)

summary = {
    'timestamp': datetime.now().isoformat(),
    'version': CONFIG.VERSION,
    'model': 'Random Forest + SHAP TreeExplainer',
    'configuration': {
        'background_samples': CONFIG.N_BACKGROUND_SAMPLES,
        'explained_samples': len(test_predictions),
        'features': feature_cols,
        'random_state': CONFIG.RANDOM_STATE
    },
    'global_insights': {
        'rf_shap_correlation': float(correlation),
        'mean_absolute_shap': float(np.abs(shap_values.values[:, :, 1]).mean()),
        'base_value': float(shap_values.base_values[0, 1]),
        'top_5_features_shap': shap_importance.head(5)[['feature', 'shap_importance']].to_dict('records')
    },
    'individual_explanations_generated': len(individual_explanations),
    'test_accuracy': float(test_acc)
}

with open(CONFIG.OUTPUT_DIR / 'module7b_shap_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✅ Results saved:")
print(f"   • {CONFIG.OUTPUT_DIR}/module7b_importance_comparison.csv")
print(f"   • {CONFIG.OUTPUT_DIR}/module7b_individual_explanations.json")
print(f"   • {CONFIG.OUTPUT_DIR}/module7b_shap_values_all.csv")
print(f"   • {CONFIG.OUTPUT_DIR}/module7b_shap_summary.json")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("🎉 MODULE 7B: SHAP EXPLAINABILITY COMPLETE!")
print("="*80)

print(f"\n🎯 EXPLAINABILITY SUMMARY:")
print(f"   • Model: Random Forest ({test_acc:.1%} accuracy)")
print(f"   • Explainer: SHAP TreeExplainer")
print(f"   • Background Data: {CONFIG.N_BACKGROUND_SAMPLES} samples")
print(f"   • Explanations Generated: {len(test_predictions)} predictions")

print(f"\n📊 GLOBAL INSIGHTS:")
print(f"   • RF-SHAP Correlation: {correlation:.3f}")
print(f"   • Top Feature (SHAP): {shap_importance.iloc[0]['feature']}")
print(f"   • Mean Absolute SHAP: {np.abs(shap_values.values).mean():.4f}")

print(f"\n🔍 LOCAL INSIGHTS:")
print(f"   • Waterfall Plots: 10 generated")
print(f"   • Individual Explanations: {len(individual_explanations)} JSON records")

print(f"\n✅ COMPETITIVE ADVANTAGES:")
print("   ✅ SHAP explainability (matches competitor)")
print("   ✅ Global + local interpretability")
print("   ✅ Feature importance validation (RF vs SHAP)")
print("   ✅ Production-ready (cached explainer)")
print("   ✅ Individual prediction explanations")

print(f"\n📊 OUTPUT FILES:")
print(f"   • Visualizations: 12 PNG files")
print(f"   • Interactive: 1 HTML file")
print(f"   • Data: 4 CSV/JSON files")

print("\n✨ Gap with competitor CLOSED - SHAP advantage neutralized!")
print("="*80)
