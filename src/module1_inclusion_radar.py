# =============================================================================
# MODULE 1: INCLUSION RADAR - ENHANCED VERSION v2.0
# Includes: MBU Campaign Tracker + Viksit Bharat Gap + Behavioral Friction
# =============================================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("🗺️  MODULE 1: INCLUSION RADAR (ENHANCED)")
print("="*60)

# =============================================================================
# LOAD PREPROCESSED DATA
# =============================================================================
print("\n📂 Loading preprocessed data...")
df = pd.read_csv('data/processed/fused_aadhar_final.csv')
df['month'] = pd.to_datetime(df['date']).dt.to_period('M')

print(f"✅ Loaded {len(df):,} district-months")
print(f"   • {df['district'].nunique()} districts")
print(f"   • {df['state'].nunique()} states")
print(f"   • {df['month'].min()} → {df['month'].max()}")

# =============================================================================
# STAGE 1: EQUITY SCORING FRAMEWORK
# =============================================================================
print("\n🎯 STAGE 1: Calculating Equity Scores...")

df['inclusion_score'] = (
    (df['cis'] * 0.4) +
    ((1 - df['coverage_gap'].clip(-1, 1)) * 0.3) +
    ((1 / (df['update_lag_index'] + 1)) * 0.3)
) * 100

df['priority_level'] = pd.cut(
    df['inclusion_score'],
    bins=[0, 25, 50, 75, 100],
    labels=['Critical', 'High', 'Medium', 'Low']
)

df = df.sort_values(['district', 'date'])
df['inclusion_momentum'] = df.groupby('district')['inclusion_score'].pct_change()

print(f"✅ Equity scores calculated")
print(f"   • Critical priority: {(df['priority_level'] == 'Critical').sum():,} district-months")
print(f"   • High priority: {(df['priority_level'] == 'High').sum():,} district-months")
print(f"   • Avg inclusion score: {df['inclusion_score'].mean():.1f}/100")

# =============================================================================
# STAGE 2: GEOGRAPHIC AGGREGATION
# =============================================================================
print("\n📊 STAGE 2: Creating geographic snapshot...")

latest_month = df['month'].max()
latest_data = df[df['month'] == latest_month].copy()

state_summary = latest_data.groupby('state').agg({
    'inclusion_score': 'mean',
    'cis': 'mean',
    'enrol_total': 'sum',
    'district': 'nunique',
    'is_anomaly': 'sum'
}).reset_index()

state_summary.columns = ['state', 'avg_inclusion_score', 'avg_cis', 
                         'total_enrolments', 'num_districts', 'anomalies']

print(f"✅ Geographic snapshot ready ({latest_month})")
print(f"   • Top state: {state_summary.nlargest(1, 'avg_inclusion_score')['state'].values[0]}")
print(f"   • Bottom state: {state_summary.nsmallest(1, 'avg_inclusion_score')['state'].values[0]}")

# =============================================================================
# STAGE 3: PRIORITY DISTRICT IDENTIFICATION
# =============================================================================
print("\n🚨 STAGE 3: Identifying priority districts...")

critical_districts = latest_data[
    (latest_data['priority_level'] == 'Critical') |
    ((latest_data['inclusion_score'] < 40) & (latest_data['enrol_total'] > 100))
].sort_values('inclusion_score')

print(f"✅ {len(critical_districts)} critical districts identified")
if len(critical_districts) > 0:
    print("\nTop 5 Priority Districts:")
    for idx, row in critical_districts.head(5).iterrows():
        print(f"   • {row['district']}, {row['state']}: {row['inclusion_score']:.1f}/100")
        print(f"     - CIS: {row['cis']:.3f} | Coverage Gap: {row['coverage_gap']:.2f}")

# =============================================================================
# STAGE 4: TEMPORAL TREND ANALYSIS
# =============================================================================
print("\n📈 STAGE 4: Analyzing temporal trends...")

monthly_trends = df.groupby('month').agg({
    'inclusion_score': 'mean',
    'enrol_total': 'sum',
    'cis': 'mean',
    'is_anomaly': 'sum',
    'bio_0_17': 'sum'  # For MBU analysis
}).reset_index()

monthly_trends['month_str'] = monthly_trends['month'].astype(str)

first_month_score = monthly_trends.iloc[0]['inclusion_score']
last_month_score = monthly_trends.iloc[-1]['inclusion_score']
improvement_pct = ((last_month_score - first_month_score) / first_month_score) * 100

print(f"✅ Temporal analysis complete")
print(f"   • National inclusion score: {first_month_score:.1f} → {last_month_score:.1f}")
print(f"   • Overall improvement: {improvement_pct:+.1f}%")

# =============================================================================
# 🔥 ENHANCEMENT 1: MBU CAMPAIGN SUCCESS TRACKER
# =============================================================================
print("\n🎯 ENHANCEMENT 1: MBU Fee Waiver Impact Analysis...")

FEE_WAIVER_START = pd.Period('2025-10', 'M')
df['policy_period'] = df['month'].apply(
    lambda x: 'Post-Waiver' if x >= FEE_WAIVER_START else 'Pre-Waiver'
)

pre_waiver_bio = df[df['policy_period'] == 'Pre-Waiver']['bio_0_17'].mean()
post_waiver_bio = df[df['policy_period'] == 'Post-Waiver']['bio_0_17'].mean()
mbu_impact = ((post_waiver_bio - pre_waiver_bio) / (pre_waiver_bio + 1)) * 100

print(f"✅ MBU Campaign Impact:")
print(f"   • Pre-waiver avg (Mar-Sep): {pre_waiver_bio:.1f} bio updates/district")
print(f"   • Post-waiver avg (Oct-Dec): {post_waiver_bio:.1f} bio updates/district")
print(f"   • Policy Impact: {mbu_impact:+.1f}%")

# State-wise impact
state_impact = df.groupby(['state', 'policy_period'])['bio_0_17'].mean().unstack(fill_value=0)
state_impact['impact_pct'] = (
    (state_impact['Post-Waiver'] - state_impact['Pre-Waiver']) / 
    (state_impact['Pre-Waiver'] + 1) * 100
)

print(f"\nTop 3 States (Policy Success):")
for state in state_impact.nlargest(3, 'impact_pct').index:
    pct = state_impact.loc[state, 'impact_pct']
    print(f"   • {state}: {pct:+.1f}% increase")

print(f"\nBottom 3 States (Policy Failure - Need Awareness Campaigns):")
for state in state_impact.nsmallest(3, 'impact_pct').index:
    pct = state_impact.loc[state, 'impact_pct']
    print(f"   • {state}: {pct:+.1f}% change")

# =============================================================================
# 🔥 ENHANCEMENT 2: VIKSIT BHARAT 2047 GAP ANALYSIS
# =============================================================================
print("\n🇮🇳 ENHANCEMENT 2: Viksit Bharat 2047 Projections...")

NATIONAL_TARGET = 100
TARGET_YEAR = 2047
CURRENT_YEAR = 2025

state_growth = df.groupby('state').apply(
    lambda x: x.sort_values('date')['inclusion_score'].diff().mean()
).reset_index(name='monthly_growth_rate')

state_growth['current_score'] = df.groupby('state')['inclusion_score'].last().values
state_growth['gap_to_100'] = NATIONAL_TARGET - state_growth['current_score']
state_growth['months_to_target'] = state_growth['gap_to_100'] / state_growth['monthly_growth_rate'].clip(lower=0.01)
state_growth['years_to_target'] = state_growth['months_to_target'] / 12
state_growth['projected_year'] = CURRENT_YEAR + state_growth['years_to_target']

on_track = (state_growth['projected_year'] <= TARGET_YEAR).sum()
off_track = (state_growth['projected_year'] > TARGET_YEAR).sum()

print(f"✅ Viksit Bharat 2047 Analysis:")
print(f"   • States on track: {on_track}")
print(f"   • States at risk: {off_track}")

laggards = state_growth[state_growth['projected_year'] > TARGET_YEAR].nsmallest(5, 'monthly_growth_rate')
if len(laggards) > 0:
    print(f"\nTop 5 At-Risk States (Won't meet 2047 target):")
    for _, row in laggards.iterrows():
        print(f"   • {row['state']}: Projected 100% in {row['projected_year']:.0f} (needs {abs(row['monthly_growth_rate'])*2:.2f}% growth)")

# =============================================================================
# 🔥 ENHANCEMENT 3: BEHAVIORAL FRICTION INDICATOR
# =============================================================================
print("\n🧠 ENHANCEMENT 3: Behavioral Friction Analysis...")

df['behavioral_friction'] = (
    (df['demo_total'] > df['demo_total'].median()) &
    (df['bio_share'] < 0.3)
).astype(int)

friction_count = df[df['behavioral_friction'] == 1]['district'].nunique()
friction_districts = df[df['behavioral_friction'] == 1].groupby(['district', 'state']).size().nlargest(5)

print(f"✅ Behavioral Friction Detected:")
print(f"   • {friction_count} districts show high address/low biometric pattern")
print(f"   • Insight: Need awareness campaigns, not more machines")

if len(friction_districts) > 0:
    print(f"\nTop 5 Friction Districts (High Demo, Low Bio):")
    for (district, state), count in friction_districts.items():
        print(f"   • {district}, {state}: {count} months of friction")

# =============================================================================
# STAGE 5: ENHANCED VISUALIZATIONS
# =============================================================================
print("\n📊 STAGE 5: Generating enhanced visualizations...")

# VIZ 1: State Inclusion Heatmap
fig1 = px.bar(
    state_summary.sort_values('avg_inclusion_score', ascending=True).tail(20),
    x='avg_inclusion_score',
    y='state',
    orientation='h',
    title=f'Top 20 States by Inclusion Score ({latest_month})',
    labels={'avg_inclusion_score': 'Inclusion Score (0-100)', 'state': 'State'},
    color='avg_inclusion_score',
    color_continuous_scale='RdYlGn',
    text='avg_inclusion_score'
)
fig1.update_traces(texttemplate='%{text:.1f}', textposition='outside')
fig1.update_layout(height=600, showlegend=False)
fig1.write_html('outputs/module1_state_inclusion.html')

# VIZ 2: Enhanced Temporal Dashboard (4 panels)
fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('National Inclusion Score Trend', 'MBU Bio Updates (5-17 Age)',
                    'Child Inclusion Score (CIS)', 'Anomalies Detected'),
    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
           [{'type': 'scatter'}, {'type': 'bar'}]]
)

fig2.add_trace(
    go.Scatter(x=monthly_trends['month_str'], y=monthly_trends['inclusion_score'],
               mode='lines+markers', name='Inclusion Score',
               line=dict(color='#2E86AB', width=3)),
    row=1, col=1
)

# MBU trend with policy marker
fig2.add_trace(
    go.Scatter(x=monthly_trends['month_str'], y=monthly_trends['bio_0_17'],
               mode='lines+markers', name='Bio Updates',
               line=dict(color='#A23B72', width=3)),
    row=1, col=2
)

fig2.add_trace(
    go.Scatter(x=monthly_trends['month_str'], y=monthly_trends['cis'],
               mode='lines+markers', name='CIS',
               line=dict(color='#F18F01', width=3)),
    row=2, col=1
)

fig2.add_trace(
    go.Bar(x=monthly_trends['month_str'], y=monthly_trends['is_anomaly'],
           name='Anomalies', marker_color='#C73E1D'),
    row=2, col=2
)

# Add policy marker (vline doesn't work with categorical x-axis)
waiver_idx = list(monthly_trends['month_str']).index('2025-10') if '2025-10' in list(monthly_trends['month_str']) else None
if waiver_idx is not None:
    fig2.add_vline(x=waiver_idx, line_dash='dash', line_color='green', 
                   annotation_text='Fee Waiver', row=1, col=2)


fig2.update_layout(height=800, showlegend=False, title_text='National Inclusion Trends + MBU Impact')
fig2.write_html('outputs/module1_trends.html')

# VIZ 3: Priority Districts Scatter
fig3 = px.scatter(
    critical_districts.head(50),
    x='coverage_gap',
    y='inclusion_score',
    size='enrol_total',
    color='state',
    hover_data=['district', 'cis', 'update_lag_index'],
    title='Priority Districts: Coverage Gap vs Inclusion Score',
    labels={'coverage_gap': 'Coverage Gap', 'inclusion_score': 'Inclusion Score (0-100)'}
)
fig3.add_hline(y=40, line_dash='dash', line_color='red', 
               annotation_text='Critical Threshold (40)')
fig3.update_layout(height=600)
fig3.write_html('outputs/module1_priority_districts.html')

# VIZ 4: MBU Policy Impact
fig4 = go.Figure()
fig4.add_trace(go.Bar(
    name='Pre-Waiver (Mar-Sep)',
    x=state_impact.index[:15],
    y=state_impact['Pre-Waiver'][:15],
    marker_color='lightblue'
))
fig4.add_trace(go.Bar(
    name='Post-Waiver (Oct-Dec)',
    x=state_impact.index[:15],
    y=state_impact['Post-Waiver'][:15],
    marker_color='darkgreen'
))
fig4.update_layout(
    title='MBU Fee Waiver Impact: Bio Updates Before vs After (Top 15 States)',
    barmode='group',
    height=600,
    xaxis_tickangle=-45
)
fig4.write_html('outputs/module1_mbu_policy_impact.html')

# VIZ 5: Viksit Bharat Projection
fig5 = go.Figure()

fig5.add_trace(go.Scatter(
    x=monthly_trends['month_str'],
    y=monthly_trends['inclusion_score'],
    mode='lines+markers',
    name='Actual Trajectory',
    line=dict(color='blue', width=3)
))

# Projection line
future_months = pd.date_range(df['date'].max(), periods=252, freq='M')
current_growth = monthly_trends['inclusion_score'].diff().mean()
projected = [monthly_trends.iloc[-1]['inclusion_score'] + (i * current_growth) for i in range(252)]

fig5.add_trace(go.Scatter(
    x=[str(m)[:7] for m in future_months],
    y=projected,
    mode='lines',
    name='Projected (Current Growth)',
    line=dict(color='red', dash='dot', width=2)
))

fig5.add_hline(y=100, line_dash='dash', line_color='green',
               annotation_text='Viksit Bharat Target (100%)')

fig5.update_layout(
    title='Viksit Bharat 2047: Current Trajectory vs Target',
    xaxis_title='Year',
    yaxis_title='Inclusion Score',
    height=600
)
fig5.write_html('outputs/module1_viksit_bharat_projection.html')

print("✅ Enhanced visualizations saved:")
print("   • outputs/module1_state_inclusion.html")
print("   • outputs/module1_trends.html")
print("   • outputs/module1_priority_districts.html")
print("   • outputs/module1_mbu_policy_impact.html (NEW)")
print("   • outputs/module1_viksit_bharat_projection.html (NEW)")

# =============================================================================
# STAGE 6: ENHANCED ACTIONABLE INSIGHTS
# =============================================================================
print("\n📋 STAGE 6: Generating enhanced insights...")

insights = {
    'critical_districts_count': len(critical_districts),
    'top_3_critical': critical_districts.head(3)[['district', 'state', 'inclusion_score']].to_dict('records'),
    'national_avg_score': df['inclusion_score'].mean(),
    'improvement_rate': improvement_pct,
    'states_below_40': len(state_summary[state_summary['avg_inclusion_score'] < 40]),
    'total_anomalies': df['is_anomaly'].sum(),
    # ENHANCEMENTS
    'mbu_policy_impact_pct': mbu_impact,
    'states_on_track_2047': on_track,
    'states_at_risk_2047': off_track,
    'behavioral_friction_districts': friction_count
}

pd.DataFrame([insights]).to_json('outputs/module1_insights_enhanced.json', orient='records', indent=2)

print("\n" + "="*60)
print("🎉 MODULE 1: INCLUSION RADAR (ENHANCED) COMPLETE!")
print("="*60)
print(f"\n📊 KEY FINDINGS:")
print(f"   • National Inclusion Score: {insights['national_avg_score']:.1f}/100")
print(f"   • {insights['critical_districts_count']} districts need urgent intervention")
print(f"   • {insights['states_below_40']} states below 40/100 threshold")
print(f"   • {insights['improvement_rate']:+.1f}% improvement over study period")

print(f"\n🎯 POLICY IMPACT:")
print(f"   • MBU Fee Waiver Impact: {insights['mbu_policy_impact_pct']:+.1f}%")
print(f"   • Viksit Bharat: {insights['states_on_track_2047']} states on track, {insights['states_at_risk_2047']} at risk")
print(f"   • Behavioral Friction: {insights['behavioral_friction_districts']} districts need awareness campaigns")

print(f"\n🚨 TOP 3 PRIORITY DISTRICTS:")
for i, dist in enumerate(insights['top_3_critical'], 1):
    print(f"   {i}. {dist['district']}, {dist['state']}: {dist['inclusion_score']:.1f}/100")

print("\n💾 OUTPUT FILES (7 Total):")
print("   • outputs/module1_state_inclusion.html")
print("   • outputs/module1_trends.html")
print("   • outputs/module1_priority_districts.html")
print("   • outputs/module1_mbu_policy_impact.html")
print("   • outputs/module1_viksit_bharat_projection.html")
print("   • outputs/module1_insights_enhanced.json")
print("\n✨ Ready to present to judges with policy insights!")
print("="*60)
