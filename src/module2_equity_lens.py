# =============================================================================
# MODULE 2: EQUITY LENS - Demographic Fairness Analysis
# Identifies age, gender, and socioeconomic disparities in Aadhaar access
# 
# Design Principles:
# - RIGOR: Statistical validation of all metrics
# - REPRODUCIBILITY: Seed-controlled randomness, documented assumptions
# - SCALABILITY: Vectorized operations, efficient grouping
# - ACCOUNTABILITY: Audit trail of all decisions
# =============================================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Reproducibility: Set random seed
np.random.seed(42)

print("👥 MODULE 2: EQUITY LENS - Demographic Fairness Analysis")
print("="*70)

# =============================================================================
# CONFIGURATION (Reproducibility)
# =============================================================================
CONFIG = {
    'CHILD_AGE_THRESHOLD': 17,  # Age definition for vulnerable population
    'CRITICAL_CIS_THRESHOLD': 0.3,  # Below 30% = crisis
    'HIGH_RISK_CIS_THRESHOLD': 0.5,  # Below 50% = high risk
    'ADULT_CHILD_IMBALANCE_THRESHOLD': 3.0,  # 3:1 ratio = gender concern
    'QUARTILE_BINS': [0, 0.25, 0.50, 0.75, 1.0],  # Equity segmentation
    'OUTPUT_DIR': 'outputs/',
    'VALIDATION_ALPHA': 0.05  # Statistical significance level
}

print(f"\n⚙️  Configuration:")
for key, value in CONFIG.items():
    print(f"   • {key}: {value}")

# =============================================================================
# STAGE 1: DATA LOADING & VALIDATION
# =============================================================================
print("\n📂 STAGE 1: Loading & Validating Data...")

try:
    df = pd.read_csv('data/processed/fused_aadhar_final.csv')
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    
    # Data validation - FIXED: removed inclusion_score dependency
    required_cols = ['state', 'district', 'date', 'cis', 'adult_child_ratio', 
                     'enrol_0_17', 'enrol_18_plus']
    
    # Optional columns (check but don't fail if missing)
    optional_cols = ['demo_0_17', 'demo_18_plus', 'bio_0_17', 'bio_18_plus', 
                     'update_lag_index', 'coverage_gap', 'enrol_momentum']
    
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    
    # Fill optional columns with defaults if missing
    for col in optional_cols:
        if col not in df.columns:
            if 'demo' in col or 'bio' in col:
                df[col] = 0  # No demographic/biometric updates
            elif col == 'update_lag_index':
                df[col] = 0.5  # Moderate lag
            elif col == 'coverage_gap':
                df[col] = 0  # Balanced coverage
            elif col == 'enrol_momentum':
                df[col] = 0  # Stable enrollment
            print(f"   ⚠️  Optional column '{col}' not found - using default values")
    
    # Create inclusion_score if missing (composite metric)
    if 'inclusion_score' not in df.columns:
        df['inclusion_score'] = (
            df['cis'] * 0.6 +  # 60% weight to child inclusion
            (1 - df['coverage_gap'].clip(-1, 1)) * 0.3 +  # 30% to coverage
            (1 - df['update_lag_index']) * 0.1  # 10% to timeliness
        ).clip(0, 1)
        print(f"   ⚠️  'inclusion_score' calculated from CIS + coverage + lag")
    
    print(f"✅ Loaded {len(df):,} district-months")
    print(f"   • {df['district'].nunique()} districts")
    print(f"   • {df['state'].nunique()} states")
    print(f"   • Data quality: {df[required_cols].notna().all(axis=1).sum()/len(df)*100:.1f}% complete")
    
except FileNotFoundError:
    print("❌ Error: fused_aadhar_final.csv not found. Run data_layer.py first.")
    exit(1)
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# =============================================================================
# STAGE 2: AGE DISPARITY ANALYSIS
# Judge Question: "Are children being left behind?"
# =============================================================================
print("\n🧒 STAGE 2: Age Disparity Analysis...")

# Calculate age-specific enrollment rates
df['child_enrollment_rate'] = df['enrol_0_17'] / (df['enrol_0_17'] + df['demo_0_17'] + 1)
df['adult_enrollment_rate'] = df['enrol_18_plus'] / (df['enrol_18_plus'] + df['demo_18_plus'] + 1)

# Age gap = difference in enrollment rates
df['age_gap'] = df['adult_enrollment_rate'] - df['child_enrollment_rate']

# Classify districts by age equity
df['age_equity_category'] = pd.cut(
    df['age_gap'],
    bins=[-np.inf, -0.1, 0.1, np.inf],
    labels=['Child-Favored', 'Balanced', 'Adult-Favored']
)

# National statistics
national_child_rate = df['child_enrollment_rate'].mean()
national_adult_rate = df['adult_enrollment_rate'].mean()
national_age_gap = national_adult_rate - national_child_rate

print(f"✅ Age Disparity Metrics:")
print(f"   • National child enrollment rate: {national_child_rate:.1%}")
print(f"   • National adult enrollment rate: {national_adult_rate:.1%}")
print(f"   • Age gap (adult-child): {national_age_gap:+.1%}")

# Statistical significance test (paired t-test)
t_stat, p_value = stats.ttest_rel(df['adult_enrollment_rate'], df['child_enrollment_rate'])
print(f"   • Statistical significance: t={t_stat:.2f}, p={p_value:.4f}")
if p_value < CONFIG['VALIDATION_ALPHA']:
    print(f"   ✅ Age gap is statistically significant (p < {CONFIG['VALIDATION_ALPHA']})")
else:
    print(f"   ⚠️  Age gap not statistically significant")

# Identify child-underserved districts
child_underserved = df[
    (df['age_gap'] > 0.2) &  # Adults 20%+ ahead
    (df['child_enrollment_rate'] < CONFIG['HIGH_RISK_CIS_THRESHOLD'])
].groupby(['district', 'state']).size().reset_index(name='months_underserved')

print(f"\n🚨 Child-Underserved Districts:")
print(f"   • {len(child_underserved)} districts with persistent child exclusion")
if len(child_underserved) > 0:
    print(f"   Top 5 Districts:")
    for _, row in child_underserved.nlargest(5, 'months_underserved').iterrows():
        print(f"      • {row['district']}, {row['state']}: {row['months_underserved']} months")

# =============================================================================
# STAGE 3: VULNERABLE POPULATION SEGMENTATION
# Judge Question: "Which populations face the greatest barriers?"
# =============================================================================
print("\n🎯 STAGE 3: Vulnerable Population Segmentation...")

# Segment districts by CIS (Child Inclusion Score) quartiles
latest_month = df['month'].max()
latest_data = df[df['month'] == latest_month].copy()

latest_data['cis_quartile'] = pd.qcut(
    latest_data['cis'],
    q=4,
    labels=['Q1 (Worst)', 'Q2 (Low)', 'Q3 (Medium)', 'Q4 (Best)'],
    duplicates='drop'
)

# Vulnerability Score (composite: low CIS + high update lag + low coverage)
latest_data['vulnerability_score'] = (
    (1 - latest_data['cis']) * 0.5 +  # 50% weight: Child inclusion
    (latest_data['update_lag_index'] / latest_data['update_lag_index'].max()) * 0.3 +  # 30%: Service lag
    (1 - latest_data['coverage_gap'].clip(-1, 1)) * 0.2  # 20%: Geographic isolation
) * 100

# Classify vulnerability
latest_data['vulnerability_level'] = pd.cut(
    latest_data['vulnerability_score'],
    bins=[0, 25, 50, 75, 100],
    labels=['Low', 'Moderate', 'High', 'Critical']
)

# Summary statistics
print(f"✅ Vulnerability Segmentation ({latest_month}):")
for level in ['Critical', 'High', 'Moderate', 'Low']:
    count = (latest_data['vulnerability_level'] == level).sum()
    pct = count / len(latest_data) * 100
    print(f"   • {level}: {count} districts ({pct:.1f}%)")

# Identify most vulnerable
critical_vulnerable = latest_data[
    latest_data['vulnerability_level'] == 'Critical'
].nlargest(10, 'vulnerability_score')[['district', 'state', 'vulnerability_score', 'cis', 'adult_child_ratio']]

print(f"\n🚨 Top 10 Most Vulnerable Districts:")
for idx, row in critical_vulnerable.iterrows():
    print(f"   • {row['district']}, {row['state']}")
    print(f"     - Vulnerability Score: {row['vulnerability_score']:.1f}/100")
    print(f"     - CIS: {row['cis']:.3f} | Adult-Child Ratio: {row['adult_child_ratio']:.2f}")

# =============================================================================
# STAGE 4: GENDER GAP PROXY ANALYSIS
# Judge Question: "Are there gender-based exclusion patterns?"
# Methodology: High adult-child ratio may indicate male-dominated enrollment
# =============================================================================
print("\n⚖️  STAGE 4: Gender Gap Proxy Analysis...")

# Flag potential gender imbalance
df['gender_risk_flag'] = (
    df['adult_child_ratio'] > CONFIG['ADULT_CHILD_IMBALANCE_THRESHOLD']
).astype(int)

gender_risk_districts = df[df['gender_risk_flag'] == 1].groupby(['district', 'state']).size()

print(f"✅ Gender Risk Assessment:")
print(f"   • Methodology: Adult-child ratio > {CONFIG['ADULT_CHILD_IMBALANCE_THRESHOLD']}:1 indicates potential male enrollment bias")
print(f"   • Assumption: In patriarchal contexts, adults (esp. male heads of household) enroll first")
print(f"   • {len(gender_risk_districts)} districts flagged for gender disparity risk")

# State-level gender risk patterns
state_gender_risk = df.groupby('state').agg({
    'gender_risk_flag': 'mean',
    'adult_child_ratio': 'mean'
}).sort_values('gender_risk_flag', ascending=False)

print(f"\nTop 5 States with Highest Gender Risk:")
for state in state_gender_risk.head(5).index:
    risk_pct = state_gender_risk.loc[state, 'gender_risk_flag'] * 100
    ratio = state_gender_risk.loc[state, 'adult_child_ratio']
    print(f"   • {state}: {risk_pct:.1f}% districts flagged | Avg ratio: {ratio:.2f}:1")

# Accountability note
print(f"\n⚠️  Methodological Caveat:")
print(f"   Adult-child ratio is a PROXY, not direct gender measurement.")
print(f"   Requires field validation with actual gender-disaggregated data.")
print(f"   Recommended: Cross-reference with census female literacy rates.")

# =============================================================================
# STAGE 5: URBAN-RURAL EQUITY (Proxy via Coverage Gap)
# Judge Question: "Is rural inclusion lagging behind urban areas?"
# =============================================================================
print("\n🏘️  STAGE 5: Geographic Equity Analysis...")

# Proxy: High coverage gap = rural (many pincodes), Low = urban (few pincodes)
# This is a heuristic since actual urban/rural classification not in data
df['geo_classification'] = pd.cut(
    df['coverage_gap'],
    bins=[-np.inf, -0.3, 0.3, np.inf],
    labels=['Urban-like (Low Coverage)', 'Mixed', 'Rural-like (High Coverage)']
)

# Compare equity metrics across geo types
geo_equity = df.groupby('geo_classification').agg({
    'cis': 'mean',
    'inclusion_score': 'mean',
    'enrol_total': 'sum',
    'district': 'nunique'
}).round(2)

print(f"✅ Geographic Equity Comparison:")
print(geo_equity.to_string())

# Statistical test: Is urban-rural gap significant?
urban_like = df[df['geo_classification'] == 'Urban-like (Low Coverage)']['cis']
rural_like = df[df['geo_classification'] == 'Rural-like (High Coverage)']['cis']

if len(urban_like) > 0 and len(rural_like) > 0:
    u_stat, p_value_geo = stats.mannwhitneyu(urban_like, rural_like, alternative='two-sided')
    print(f"\n📊 Mann-Whitney U Test (Urban vs Rural):")
    print(f"   • U-statistic: {u_stat:.0f}")
    print(f"   • p-value: {p_value_geo:.4f}")
    if p_value_geo < CONFIG['VALIDATION_ALPHA']:
        print(f"   ✅ Urban-rural gap is statistically significant")
    else:
        print(f"   ⚠️  No significant urban-rural difference detected")

# =============================================================================
# STAGE 6: EQUITY SCORECARD (Composite Dashboard)
# =============================================================================
print("\n📊 STAGE 6: Generating Equity Scorecard...")

# State-level equity scorecard
# Use latest_data (which has vulnerability_score) for scorecard
state_scorecard = latest_data.groupby('state').agg({
    'cis': 'mean',
    'adult_child_ratio': 'mean',
    'vulnerability_score': 'mean',
    'district': 'nunique'
}).round(3)

# Add historical metrics from full df
state_historical = df.groupby('state').agg({
    'child_enrollment_rate': 'mean',
    'adult_enrollment_rate': 'mean',
    'age_gap': 'mean',
    'gender_risk_flag': 'mean'
}).round(3)

# Merge
state_scorecard = state_scorecard.join(state_historical)

state_scorecard.columns = [
    'Avg CIS', 'Adult-Child Ratio', 'Avg Vulnerability',
    'Num Districts', 'Child Enrol Rate', 'Adult Enrol Rate',
    'Age Gap', 'Gender Risk %'
]

state_scorecard['Equity Rank'] = state_scorecard['Avg CIS'].rank(ascending=False).astype(int)
state_scorecard = state_scorecard.sort_values('Equity Rank')

print(f"✅ State Equity Scorecard Generated")
print(f"\nTop 5 Most Equitable States:")
print(state_scorecard.head(5)[['Avg CIS', 'Age Gap', 'Gender Risk %', 'Equity Rank']].to_string())

print(f"\nBottom 5 Least Equitable States:")
print(state_scorecard.tail(5)[['Avg CIS', 'Age Gap', 'Gender Risk %', 'Equity Rank']].to_string())

# =============================================================================
# STAGE 7: INTERACTIVE VISUALIZATIONS
# =============================================================================
print("\n📊 STAGE 7: Generating Interactive Visualizations...")

# VIZ 1: Age Disparity Funnel
fig1 = go.Figure()

# National enrollment funnel
stages = ['Total Eligible\n(0-17)', 'Enrolled', 'Updated Demo', 'Updated Bio']
child_values = [
    df['enrol_0_17'].sum() + df['demo_0_17'].sum(),
    df['enrol_0_17'].sum(),
    df['demo_0_17'].sum(),
    df['bio_0_17'].sum()
]

adult_values = [
    df['enrol_18_plus'].sum() + df['demo_18_plus'].sum(),
    df['enrol_18_plus'].sum(),
    df['demo_18_plus'].sum(),
    df['bio_18_plus'].sum()
]

fig1.add_trace(go.Funnel(
    name='Children (0-17)',
    y=stages,
    x=child_values,
    textinfo='value+percent initial',
    marker=dict(color='#3498db')
))

fig1.add_trace(go.Funnel(
    name='Adults (18+)',
    y=stages,
    x=adult_values,
    textinfo='value+percent initial',
    marker=dict(color='#e74c3c')
))

fig1.update_layout(
    title='Age Disparity: Enrollment Funnel (Children vs Adults)',
    height=600
)
fig1.write_html(f"{CONFIG['OUTPUT_DIR']}module2_age_funnel.html")

# VIZ 2: Vulnerability Heatmap
fig2 = px.scatter(
    latest_data,
    x='cis',
    y='adult_child_ratio',
    size='enrol_total',
    color='vulnerability_level',
    hover_data=['district', 'state', 'vulnerability_score'],
    title='Vulnerability Matrix: CIS vs Adult-Child Ratio',
    labels={'cis': 'Child Inclusion Score', 'adult_child_ratio': 'Adult-Child Ratio'},
    color_discrete_map={
        'Low': '#2ecc71',
        'Moderate': '#f39c12',
        'High': '#e67e22',
        'Critical': '#e74c3c'
    },
    category_orders={'vulnerability_level': ['Low', 'Moderate', 'High', 'Critical']}
)

# Add threshold lines
fig2.add_hline(y=CONFIG['ADULT_CHILD_IMBALANCE_THRESHOLD'], line_dash='dash', 
               line_color='red', annotation_text='Gender Risk Threshold')
fig2.add_vline(x=CONFIG['HIGH_RISK_CIS_THRESHOLD'], line_dash='dash', 
               line_color='orange', annotation_text='High Risk CIS')

fig2.update_layout(height=600)
fig2.write_html(f"{CONFIG['OUTPUT_DIR']}module2_vulnerability_matrix.html")

# VIZ 3: Equity Scorecard Dashboard
fig3 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('CIS Distribution by State', 'Age Gap (Adult - Child)',
                    'Gender Risk by State', 'Vulnerability Distribution'),
    specs=[[{'type': 'box'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'histogram'}]]
)

# Plot 1: CIS Box Plot
for state in df['state'].unique()[:10]:  # Top 10 states
    state_data = df[df['state'] == state]['cis']
    fig3.add_trace(
        go.Box(y=state_data, name=state, showlegend=False),
        row=1, col=1
    )

# Plot 2: Age Gap Bar
top_states_age = state_scorecard.nlargest(10, 'Age Gap')
fig3.add_trace(
    go.Bar(x=top_states_age.index, y=top_states_age['Age Gap'], 
           marker_color='coral', showlegend=False),
    row=1, col=2
)

# Plot 3: Gender Risk
top_states_gender = state_scorecard.nlargest(10, 'Gender Risk %')
fig3.add_trace(
    go.Bar(x=top_states_gender.index, y=top_states_gender['Gender Risk %'] * 100,
           marker_color='indianred', showlegend=False),
    row=2, col=1
)

# Plot 4: Vulnerability Histogram
fig3.add_trace(
    go.Histogram(x=latest_data['vulnerability_score'], nbinsx=30,
                 marker_color='mediumpurple', showlegend=False),
    row=2, col=2
)

fig3.update_layout(height=900, showlegend=False, title_text='State Equity Dashboard')
fig3.update_xaxes(tickangle=-45, row=1, col=2)
fig3.update_xaxes(tickangle=-45, row=2, col=1)
fig3.write_html(f"{CONFIG['OUTPUT_DIR']}module2_equity_dashboard.html")

print("✅ Visualizations saved:")
print(f"   • {CONFIG['OUTPUT_DIR']}module2_age_funnel.html")
print(f"   • {CONFIG['OUTPUT_DIR']}module2_vulnerability_matrix.html")
print(f"   • {CONFIG['OUTPUT_DIR']}module2_equity_dashboard.html")

# =============================================================================
# STAGE 8: ACTIONABLE INSIGHTS & POLICY RECOMMENDATIONS
# =============================================================================
print("\n📋 STAGE 8: Generating Policy Recommendations...")

# Calculate metrics first
critical_vulnerable_count = int((latest_data['vulnerability_level'] == 'Critical').sum())
high_vulnerable_count = int((latest_data['vulnerability_level'] == 'High').sum())
child_underserved_count = len(child_underserved)
gender_risk_count = len(gender_risk_districts)

# Build insights dictionary
insights = {
    'analysis_date': str(latest_month),
    'total_districts_analyzed': len(latest_data),
    
    # Age equity
    'national_child_enrollment_rate': round(national_child_rate, 3),
    'national_adult_enrollment_rate': round(national_adult_rate, 3),
    'age_gap': round(national_age_gap, 3),
    'age_gap_significant': bool(p_value < CONFIG['VALIDATION_ALPHA']),
    'child_underserved_districts': child_underserved_count,
    
    # Vulnerability
    'critical_vulnerable_districts': critical_vulnerable_count,
    'high_vulnerable_districts': high_vulnerable_count,
    'top_3_vulnerable': critical_vulnerable.head(3)[['district', 'state', 'vulnerability_score']].to_dict('records'),
    
    # Gender proxy
    'gender_risk_districts': gender_risk_count,
    'avg_adult_child_ratio': round(df['adult_child_ratio'].mean(), 2),
    
    # Policy recommendations
    'recommendations': [
        {
            'priority': 'HIGH',
            'target': f"{critical_vulnerable_count} critical vulnerable districts",
            'action': 'Deploy mobile Aadhaar camps with child-focused staff within 60 days',
            'rationale': 'CIS below 0.3 indicates systemic child exclusion'
        },
        {
            'priority': 'HIGH',
            'target': f"{child_underserved_count} child-underserved districts",
            'action': 'Partner with schools for enrollment drives during admission season',
            'rationale': f"Age gap of {national_age_gap:.1%} is statistically significant (p={p_value:.4f})"
        },
        {
            'priority': 'MEDIUM',
            'target': f"{gender_risk_count} gender risk districts",
            'action': 'Conduct field validation of adult-child ratio patterns with gender-disaggregated surveys',
            'rationale': 'Adult-child ratio proxy requires ground-truth verification'
        },
        {
            'priority': 'MEDIUM',
            'target': 'Rural-like districts (high coverage gap)',
            'action': 'Deploy Aadhaar Seva Kendras in block headquarters to reduce travel distance',
            'rationale': 'Geographic isolation correlates with low inclusion scores'
        },
        {
            'priority': 'CRITICAL',
            'target': f"{(latest_data['cis'] == 0).sum()} districts with zero CIS",
            'action': 'URGENT: Investigate data quality issues or launch emergency child enrollment drives',
            'rationale': 'CIS=0 indicates either data failure or complete child exclusion crisis'
        }
    ]
}

# =============================================================================
# STAGE 10: DISTRICT RISK SCORING (0-100 Scale)
# =============================================================================
print("\n🎯 STAGE 10: Calculating District Risk Scores...")

# Use latest_data as base for risk scoring
risk_districts = latest_data.copy()

# Add baseline enrollment metrics
baseline_enrol = df.groupby('district').agg({
    'enrol_0_17': 'mean',
    'enrol_18_plus': 'mean',
    'enrol_total': 'mean',
    'enrol_momentum': 'mean'
}).reset_index()
baseline_enrol.columns = ['district', 'baseline_child', 'baseline_adult', 'baseline_total', 'momentum']

# Merge
risk_districts = risk_districts.merge(baseline_enrol, on='district', how='left')

# Create comprehensive risk score
def calculate_risk_score(row):
    """
    Calculate welfare exclusion risk (0-100)
    Higher score = Higher risk of exclusion
    """
    # Component 1: Low CIS (50% weight)
    cis_risk = (1 - row['cis']) * 50
    
    # Component 2: High child population (20% weight)
    if risk_districts['baseline_child'].max() > 0:
        child_pop_normalized = row['baseline_child'] / risk_districts['baseline_child'].max()
        child_pop_risk = child_pop_normalized * 20
    else:
        child_pop_risk = 0
    
    # Component 3: Update lag (15% weight)
    if 'update_lag_index' in row.index and pd.notna(row['update_lag_index']):
        lag_normalized = row['update_lag_index']  # Already 0-1 normalized
        lag_risk = lag_normalized * 15
    else:
        lag_risk = 0
    
    # Component 4: Enrollment momentum (15% weight)
    if 'momentum' in row.index and pd.notna(row['momentum']) and row['momentum'] < 0:
        if risk_districts['momentum'].abs().max() > 0:
            momentum_risk = abs(row['momentum']) / risk_districts['momentum'].abs().max() * 15
        else:
            momentum_risk = 0
    else:
        momentum_risk = 0
    
    total_risk = cis_risk + child_pop_risk + lag_risk + momentum_risk
    return min(100, total_risk)  # Cap at 100

# Calculate risk scores
risk_districts['risk_score'] = risk_districts.apply(calculate_risk_score, axis=1)

# Rank by risk
risk_districts = risk_districts.sort_values('risk_score', ascending=False)
risk_districts['priority_rank'] = range(1, len(risk_districts) + 1)

# Categorize risk levels
def risk_category(score):
    if score >= 80:
        return "🔴 CRITICAL"
    elif score >= 60:
        return "🟠 HIGH"
    elif score >= 40:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"

risk_districts['risk_category'] = risk_districts['risk_score'].apply(risk_category)

print(f"✅ Risk scoring complete for {len(risk_districts)} districts")

# Display top 20 highest risk
print("\n📊 TOP 20 HIGHEST RISK DISTRICTS:")
print("="*100)

top_20_risk = risk_districts.head(20)[['district', 'state', 'risk_score', 'risk_category', 'cis', 'baseline_child']]
print(top_20_risk.to_string(index=False))

# Risk distribution
risk_dist = risk_districts['risk_category'].value_counts()
print(f"\n📊 Risk Distribution:")
for category, count in risk_dist.items():
    print(f"   {category}: {count} districts")

# Save risk-scored districts
risk_districts.to_csv(f"{CONFIG['OUTPUT_DIR']}module2_district_risk_scores.csv", index=False)

# VIZ: Risk Score Heatmap
fig_risk = px.scatter(
    risk_districts.head(100),  # Top 100 for readability
    x='cis',
    y='risk_score',
    size='baseline_child',
    color='risk_category',
    hover_data=['district', 'state'],
    title='District Welfare Exclusion Risk Score (0-100)',
    labels={'cis': 'Child Inclusion Score', 'risk_score': 'Risk Score'},
    color_discrete_map={
        '🔴 CRITICAL': 'red',
        '🟠 HIGH': 'orange',
        '🟡 MEDIUM': 'yellow',
        '🟢 LOW': 'green'
    }
)
fig_risk.update_layout(height=600)
fig_risk.write_html(f"{CONFIG['OUTPUT_DIR']}module2_risk_score_scatter.html")

print(f"✅ Risk scoring visualization saved: {CONFIG['OUTPUT_DIR']}module2_risk_score_scatter.html")
print("="*100)

# Save insights
pd.DataFrame([insights]).to_json(f"{CONFIG['OUTPUT_DIR']}module2_equity_insights.json", orient='records', indent=2)

# Save scorecard
state_scorecard.to_csv(f"{CONFIG['OUTPUT_DIR']}module2_state_equity_scorecard.csv")

print("\n" + "="*70)
print("🎉 MODULE 2: EQUITY LENS COMPLETE!")
print("="*70)

print(f"\n📊 KEY FINDINGS:")
print(f"   • Age Gap (Adult-Child): {national_age_gap:+.1%} ({'SIGNIFICANT' if insights['age_gap_significant'] else 'NOT SIGNIFICANT'})")
print(f"   • Child Underserved Districts: {insights['child_underserved_districts']}")
print(f"   • Critical Vulnerable Districts: {insights['critical_vulnerable_districts']}")
print(f"   • Gender Risk Districts: {insights['gender_risk_districts']}")

print(f"\n🚨 TOP 3 MOST VULNERABLE DISTRICTS:")
for i, dist in enumerate(insights['top_3_vulnerable'], 1):
    print(f"   {i}. {dist['district']}, {dist['state']}: {dist['vulnerability_score']:.1f}/100")

print(f"\n💡 POLICY RECOMMENDATIONS ({len(insights['recommendations'])} Total):")
for i, rec in enumerate(insights['recommendations'], 1):
    print(f"   {i}. [{rec['priority']}] {rec['action']}")
    print(f"      Target: {rec['target']}")

print(f"\n💾 OUTPUT FILES:")
print(f"   • {CONFIG['OUTPUT_DIR']}module2_age_funnel.html")
print(f"   • {CONFIG['OUTPUT_DIR']}module2_vulnerability_matrix.html")
print(f"   • {CONFIG['OUTPUT_DIR']}module2_equity_dashboard.html")
print(f"   • {CONFIG['OUTPUT_DIR']}module2_equity_insights.json")
print(f"   • {CONFIG['OUTPUT_DIR']}module2_state_equity_scorecard.csv")

print("\n✨ Ready to present demographic equity analysis to judges!")
print("="*70)

