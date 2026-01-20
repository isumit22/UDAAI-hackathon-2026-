# =============================================================================
# BASELINE COMPARISON - Adding National & State Benchmarks
# Critical for government decision-making: "Is this district worse than average?"
# 
# ADDS:
# - National median CIS, risk scores, enrollment rates
# - State-level benchmarks for comparison
# - Percentile rankings (Top 10%, Bottom 25%, etc.)
# - "Above/Below Average" labels for every metric
# 
# OUTPUT: Updated CSV files with baseline comparisons
# =============================================================================

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("📊 BASELINE COMPARISON - Adding National & State Benchmarks")
print("="*80)
print("GOVERNMENT REQUIREMENT: Context is critical for policy decisions")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================
class BaselineConfig:
    OUTPUT_DIR = Path('outputs')
    
    # Input files
    MODULE7_PREDICTIONS = OUTPUT_DIR / 'module7_district_risk_predictions.csv'
    MODULE9_CLUSTERS = OUTPUT_DIR / 'module9_district_clusters.csv'
    MODULE2_RISK_SCORES = OUTPUT_DIR / 'module2_district_risk_scores.csv'
    
    # Output files
    BASELINE_REPORT = OUTPUT_DIR / 'BASELINE_COMPARISON_REPORT.json'
    ENHANCED_PREDICTIONS = OUTPUT_DIR / 'module7_predictions_WITH_BASELINES.csv'
    
    VERSION = '1.0.0'

CONFIG = BaselineConfig()

print(f"\n⚙️  Configuration:")
print(f"   • Input: Module 7, 9, 2 predictions")
print(f"   • Output: Baseline-enhanced CSV + JSON report")
print(f"   • Version: {CONFIG.VERSION}")

# =============================================================================
# STAGE 1: LOAD ALL PREDICTION DATA
# =============================================================================
print("\n📂 STAGE 1: Loading Prediction Data...")

try:
    predictions_df = pd.read_csv(CONFIG.MODULE7_PREDICTIONS)
    print(f"✅ Loaded Module 7 predictions: {len(predictions_df)} districts")
except FileNotFoundError:
    print("❌ Error: Module 7 predictions not found. Run module7 first.")
    exit(1)

# Load additional data if available
try:
    clusters_df = pd.read_csv(CONFIG.MODULE9_CLUSTERS)
    predictions_df = predictions_df.merge(
        clusters_df[['district', 'persona_name', 'cost_burden_index']], 
        on='district', 
        how='left'
    )
    print(f"✅ Merged cluster data")
except FileNotFoundError:
    print(f"⚠️  Module 9 clusters not found - skipping cluster merge")

# =============================================================================
# STAGE 2: CALCULATE NATIONAL BASELINES
# =============================================================================
print("\n📊 STAGE 2: Calculating National Baselines...")

# National benchmarks
national_baselines = {
    'cis_mean': {
        'median': float(predictions_df['cis_mean'].median()),
        'mean': float(predictions_df['cis_mean'].mean()),
        'p25': float(predictions_df['cis_mean'].quantile(0.25)),
        'p75': float(predictions_df['cis_mean'].quantile(0.75)),
        'min': float(predictions_df['cis_mean'].min()),
        'max': float(predictions_df['cis_mean'].max())
    },
    'risk_score': {
        'median': float(predictions_df['risk_score'].median()),
        'mean': float(predictions_df['risk_score'].mean()),
        'p25': float(predictions_df['risk_score'].quantile(0.25)),
        'p75': float(predictions_df['risk_score'].quantile(0.75)),
        'critical_threshold': 80.0,
        'high_threshold': 60.0
    },
    'enrol_mean': {
        'median': float(predictions_df['enrol_mean'].median()),
        'mean': float(predictions_df['enrol_mean'].mean()),
        'p25': float(predictions_df['enrol_mean'].quantile(0.25)),
        'p75': float(predictions_df['enrol_mean'].quantile(0.75))
    },
    'momentum': {
        'median': float(predictions_df['momentum'].median()),
        'mean': float(predictions_df['momentum'].mean()),
        'declining_threshold': -0.1
    }
}

print(f"✅ National Baselines Calculated:")
print(f"\n📊 CIS (Child Inclusion Score):")
print(f"   • National Median: {national_baselines['cis_mean']['median']:.3f}")
print(f"   • National Mean: {national_baselines['cis_mean']['mean']:.3f}")
print(f"   • 25th Percentile: {national_baselines['cis_mean']['p25']:.3f}")
print(f"   • 75th Percentile: {national_baselines['cis_mean']['p75']:.3f}")

print(f"\n📊 Risk Score:")
print(f"   • National Median: {national_baselines['risk_score']['median']:.1f}")
print(f"   • National Mean: {national_baselines['risk_score']['mean']:.1f}")

print(f"\n📊 Enrollment:")
print(f"   • National Median: {national_baselines['enrol_mean']['median']:.0f}/month")
print(f"   • National Mean: {national_baselines['enrol_mean']['mean']:.0f}/month")

# =============================================================================
# STAGE 3: CALCULATE STATE-LEVEL BASELINES
# =============================================================================
print("\n📊 STAGE 3: Calculating State-Level Baselines...")

state_baselines = predictions_df.groupby('state').agg({
    'cis_mean': ['median', 'mean', 'count'],
    'risk_score': ['median', 'mean'],
    'enrol_mean': ['median', 'mean'],
    'predicted_risk': 'sum'  # Count of high-risk districts
}).round(3)

state_baselines.columns = [
    'cis_median', 'cis_mean', 'district_count',
    'risk_median', 'risk_mean',
    'enrol_median', 'enrol_mean',
    'high_risk_count'
]

# Calculate state performance rank
state_baselines['cis_rank'] = state_baselines['cis_median'].rank(ascending=False).astype(int)
state_baselines = state_baselines.sort_values('cis_rank')

print(f"✅ State baselines calculated for {len(state_baselines)} states")
print(f"\nTop 5 States by CIS:")
for state in state_baselines.head(5).index:
    cis = state_baselines.loc[state, 'cis_median']
    rank = state_baselines.loc[state, 'cis_rank']
    print(f"   {rank}. {state}: {cis:.3f}")

print(f"\nBottom 5 States by CIS:")
for state in state_baselines.tail(5).index:
    cis = state_baselines.loc[state, 'cis_median']
    rank = state_baselines.loc[state, 'cis_rank']
    print(f"   {rank}. {state}: {cis:.3f}")

# =============================================================================
# STAGE 4: ADD BASELINE COMPARISONS TO DISTRICT DATA
# =============================================================================
print("\n📊 STAGE 4: Adding Baseline Comparisons to District Data...")

# Merge state baselines
predictions_df = predictions_df.merge(
    state_baselines[['cis_median', 'risk_median']],
    left_on='state',
    right_index=True,
    how='left',
    suffixes=('', '_state')
)

# Calculate deviations from baselines
predictions_df['cis_vs_national'] = predictions_df['cis_mean'] - national_baselines['cis_mean']['median']
predictions_df['cis_vs_state'] = predictions_df['cis_mean'] - predictions_df['cis_median']
predictions_df['risk_vs_national'] = predictions_df['risk_score'] - national_baselines['risk_score']['median']
predictions_df['risk_vs_state'] = predictions_df['risk_score'] - predictions_df['risk_median']

# Calculate percentile rankings
predictions_df['cis_percentile'] = predictions_df['cis_mean'].rank(pct=True) * 100
predictions_df['risk_percentile'] = predictions_df['risk_score'].rank(pct=True) * 100

# Classify performance vs national
def classify_vs_baseline(value, baseline, metric_type='higher_is_better'):
    """Classify performance relative to baseline"""
    deviation = value - baseline
    threshold = baseline * 0.1  # 10% deviation threshold
    
    if metric_type == 'higher_is_better':
        if deviation > threshold:
            return 'Above Average'
        elif deviation < -threshold:
            return 'Below Average'
        else:
            return 'Average'
    else:  # lower_is_better (for risk scores)
        if deviation > threshold:
            return 'Below Average'  # High risk is bad
        elif deviation < -threshold:
            return 'Above Average'  # Low risk is good
        else:
            return 'Average'

predictions_df['cis_classification'] = predictions_df.apply(
    lambda row: classify_vs_baseline(
        row['cis_mean'], 
        national_baselines['cis_mean']['median'],
        'higher_is_better'
    ), axis=1
)

predictions_df['risk_classification'] = predictions_df.apply(
    lambda row: classify_vs_baseline(
        row['risk_score'], 
        national_baselines['risk_score']['median'],
        'lower_is_better'
    ), axis=1
)

# Add severity flags
predictions_df['severity'] = 'Normal'
predictions_df.loc[predictions_df['risk_score'] >= 80, 'severity'] = '🔴 CRITICAL'
predictions_df.loc[(predictions_df['risk_score'] >= 60) & (predictions_df['risk_score'] < 80), 'severity'] = '🟠 HIGH'
predictions_df.loc[(predictions_df['risk_score'] >= 40) & (predictions_df['risk_score'] < 60), 'severity'] = '🟡 MEDIUM'
predictions_df.loc[predictions_df['risk_score'] < 40, 'severity'] = '🟢 LOW'

print(f"✅ Baseline comparisons added")

# Distribution by classification
print(f"\n📊 CIS Classification Distribution:")
for cls in ['Above Average', 'Average', 'Below Average']:
    count = (predictions_df['cis_classification'] == cls).sum()
    pct = count / len(predictions_df) * 100
    print(f"   • {cls}: {count} districts ({pct:.1f}%)")

print(f"\n📊 Severity Distribution:")
severity_counts = predictions_df['severity'].value_counts()
for severity, count in severity_counts.items():
    pct = count / len(predictions_df) * 100
    print(f"   • {severity}: {count} districts ({pct:.1f}%)")

# =============================================================================
# STAGE 5: IDENTIFY CRITICAL OUTLIERS
# =============================================================================
print("\n📊 STAGE 5: Identifying Critical Outliers...")

# Districts that are BOTH below average CIS AND above average risk
critical_outliers = predictions_df[
    (predictions_df['cis_classification'] == 'Below Average') &
    (predictions_df['risk_classification'] == 'Below Average') &  # Below avg for risk = high risk
    (predictions_df['risk_score'] >= 70)
].sort_values('risk_score', ascending=False)

print(f"✅ Identified {len(critical_outliers)} critical outliers")
print(f"\nTop 10 Critical Outliers (Low CIS + High Risk):")
for idx, row in critical_outliers.head(10).iterrows():
    print(f"   • {row['district']}, {row['state']}")
    print(f"     - Risk: {row['risk_score']:.1f} ({row['risk_vs_national']:+.1f} vs national)")
    print(f"     - CIS: {row['cis_mean']:.3f} ({row['cis_vs_national']:+.3f} vs national)")

# =============================================================================
# STAGE 6: SAVE ENHANCED DATA
# =============================================================================
print("\n💾 STAGE 6: Saving Baseline-Enhanced Data...")

# Save enhanced predictions
predictions_df.to_csv(CONFIG.ENHANCED_PREDICTIONS, index=False)
print(f"✅ Saved: {CONFIG.ENHANCED_PREDICTIONS}")

# Save state baselines
state_baselines.to_csv(CONFIG.OUTPUT_DIR / 'STATE_BASELINES.csv')
print(f"✅ Saved: STATE_BASELINES.csv")

# Create comprehensive baseline report
baseline_report = {
    'timestamp': datetime.now().isoformat(),
    'version': CONFIG.VERSION,
    'national_baselines': national_baselines,
    'state_count': len(state_baselines),
    'district_count': len(predictions_df),
    'classification_distribution': {
        'cis': predictions_df['cis_classification'].value_counts().to_dict(),
        'risk': predictions_df['risk_classification'].value_counts().to_dict(),
        'severity': predictions_df['severity'].value_counts().to_dict()
    },
    'critical_outliers': {
        'count': len(critical_outliers),
        'top_10': critical_outliers.head(10)[['district', 'state', 'risk_score', 'cis_mean']].to_dict('records')
    },
    'top_5_states': state_baselines.head(5)[['cis_median', 'cis_rank']].to_dict('index'),
    'bottom_5_states': state_baselines.tail(5)[['cis_median', 'cis_rank']].to_dict('index')
}

with open(CONFIG.BASELINE_REPORT, 'w') as f:
    json.dump(baseline_report, f, indent=2)
print(f"✅ Saved: {CONFIG.BASELINE_REPORT}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("🎉 BASELINE COMPARISON COMPLETE!")
print("="*80)

print(f"\n📊 NATIONAL BENCHMARKS ESTABLISHED:")
print(f"   • CIS Median: {national_baselines['cis_mean']['median']:.3f}")
print(f"   • Risk Median: {national_baselines['risk_score']['median']:.1f}")
print(f"   • Enrollment Median: {national_baselines['enrol_mean']['median']:.0f}/month")

print(f"\n📊 COMPARISONS ADDED:")
print(f"   ✅ CIS vs National Median")
print(f"   ✅ CIS vs State Median")
print(f"   ✅ Risk vs National Median")
print(f"   ✅ Risk vs State Median")
print(f"   ✅ Percentile Rankings (0-100)")
print(f"   ✅ Classification (Above/Average/Below)")
print(f"   ✅ Severity Flags (Critical/High/Medium/Low)")

print(f"\n📊 KEY INSIGHTS:")
print(f"   • {len(critical_outliers)} districts are CRITICAL outliers")
print(f"   • {(predictions_df['cis_classification'] == 'Below Average').sum()} districts below CIS average")
print(f"   • {(predictions_df['severity'] == '🔴 CRITICAL').sum()} districts in CRITICAL severity")

print(f"\n📁 OUTPUT FILES:")
print(f"   • {CONFIG.ENHANCED_PREDICTIONS}")
print(f"   • {CONFIG.OUTPUT_DIR}/STATE_BASELINES.csv")
print(f"   • {CONFIG.BASELINE_REPORT}")

print(f"\n✅ GOVERNMENT REQUIREMENT SATISFIED:")
print("   ✅ Every metric now has national context")
print("   ✅ State-level comparisons available")
print("   ✅ Percentile rankings for prioritization")
print("   ✅ 'Above/Below Average' labels for policy decisions")

print("\n🚀 Ready for Step 2: Action Playbook Generator!")
print("="*80)
