# =============================================================================
# MODULE 5: POLICY IMPACT SIMULATOR - What-If Analysis Engine
# Version: 2.1 (Fixed for Available Data Columns)
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pathlib import Path
import json
from datetime import datetime

print("🎯 MODULE 5: POLICY IMPACT SIMULATOR - What-If Analysis Engine")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================
class PolicyConfig:
    """Policy intervention parameters"""
    
    # Intervention Costs (per district per month)
    MOBILE_CAMP_COST = 50000
    SCHOOL_DRIVE_COST = 30000
    SENIOR_OUTREACH_COST = 40000
    DIGITAL_LITERACY_COST = 60000
    
    # Expected Impact Rates (enrollment increase %)
    MOBILE_CAMP_IMPACT = 0.15  # 15% increase
    SCHOOL_DRIVE_IMPACT = 0.25  # 25% increase (children focus)
    SENIOR_OUTREACH_IMPACT = 0.12  # 12% increase
    DIGITAL_LITERACY_IMPACT = 0.18  # 18% increase
    
    # Economic Value
    ENROLLMENT_VALUE = 100
    
    # Default Budget
    DEFAULT_BUDGET = 10000000  # ₹1 crore
    
    # Implementation Timeline
    IMPLEMENTATION_MONTHS = 6
    
    # Output
    OUTPUT_DIR = Path('outputs')
    
    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = PolicyConfig()
CONFIG.setup()

print(f"\n⚙️  Configuration:")
print(f"   • Mobile Camp Cost: ₹{CONFIG.MOBILE_CAMP_COST:,}/district/month")
print(f"   • School Drive Cost: ₹{CONFIG.SCHOOL_DRIVE_COST:,}/district/month")
print(f"   • Default Budget: ₹{CONFIG.DEFAULT_BUDGET:,}")
print(f"   • Implementation Period: {CONFIG.IMPLEMENTATION_MONTHS} months")

# =============================================================================
# STAGE 1: LOAD BASELINE DATA
# =============================================================================
print("\n📂 STAGE 1: Loading Baseline Data...")

try:
    df = pd.read_csv('data/processed/fused_aadhar_final.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Data loaded: {len(df):,} district-months")
    print(f"   • Districts: {df['district'].nunique()}")
    print(f"   • States: {df['state'].nunique()}")

except FileNotFoundError:
    print("❌ Error: fused_aadhar_final.csv not found. Run data_layer.py first.")
    exit(1)

# Check available columns
available_cols = df.columns.tolist()
print(f"\n📋 Available enrollment columns:")
enrol_cols = [col for col in available_cols if 'enrol' in col.lower()]
for col in enrol_cols:
    print(f"   • {col}")

# Calculate baseline metrics
baseline = df.groupby('district').agg({
    'enrol_total': 'mean',
    'enrol_0_17': 'mean',
    'cis': 'mean',
    'state': 'first'
}).reset_index()

baseline.columns = ['district', 'baseline_total', 'baseline_child', 'baseline_cis', 'state']

# Calculate adult enrollment (total - children)
baseline['baseline_adult'] = baseline['baseline_total'] - baseline['baseline_child']
baseline['baseline_adult'] = baseline['baseline_adult'].clip(lower=0)

print(f"\n📊 Baseline Metrics (Average per District):")
print(f"   • Total enrollment: {baseline['baseline_total'].mean():,.0f}/month")
print(f"   • Child (0-17): {baseline['baseline_child'].mean():,.0f}/month")
print(f"   • Adult (18+): {baseline['baseline_adult'].mean():,.0f}/month")
print(f"   • Average CIS: {baseline['baseline_cis'].mean():.3f}")

# =============================================================================
# STAGE 2: POLICY SCENARIO MODELING
# =============================================================================
print("\n🎯 STAGE 2: Policy Scenario Modeling...")

class PolicyScenario:
    """Model policy intervention impact"""
    
    def __init__(self, name, cost_per_district, impact_rate, description, target_group):
        self.name = name
        self.cost_per_district = cost_per_district
        self.impact_rate = impact_rate
        self.description = description
        self.target_group = target_group  # 'child', 'adult', or 'both'
    
    def calculate_impact(self, baseline_data, num_districts, months):
        """Calculate enrollment impact"""
        
        # Select priority districts (lowest CIS)
        priority_districts = baseline_data.nsmallest(num_districts, 'baseline_cis')
        
        # Calculate enrollment increase based on target group
        if self.target_group == 'child':
            target_increase = priority_districts['baseline_child'].sum() * self.impact_rate * months
            child_increase = target_increase
            adult_increase = 0
        elif self.target_group == 'adult':
            target_increase = priority_districts['baseline_adult'].sum() * self.impact_rate * months
            child_increase = 0
            adult_increase = target_increase
        else:  # both
            child_increase = priority_districts['baseline_child'].sum() * self.impact_rate * months
            adult_increase = priority_districts['baseline_adult'].sum() * self.impact_rate * months
            target_increase = child_increase + adult_increase
        
        total_increase = child_increase + adult_increase
        
        # Calculate costs
        total_cost = self.cost_per_district * num_districts * months
        
        # Calculate benefits
        total_benefit = total_increase * CONFIG.ENROLLMENT_VALUE
        
        # ROI
        roi = ((total_benefit - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        # CIS improvement
        avg_baseline_cis = priority_districts['baseline_cis'].mean()
        avg_new_enrollments = total_increase / num_districts / months
        avg_baseline_enrollments = priority_districts['baseline_total'].mean()
        estimated_new_cis = avg_new_enrollments / (avg_baseline_enrollments + avg_new_enrollments) if (avg_baseline_enrollments + avg_new_enrollments) > 0 else avg_baseline_cis
        cis_improvement = estimated_new_cis - avg_baseline_cis
        
        return {
            'scenario': self.name,
            'districts_covered': num_districts,
            'duration_months': months,
            'total_cost': total_cost,
            'child_enrollment_increase': child_increase,
            'adult_enrollment_increase': adult_increase,
            'total_enrollment_increase': total_increase,
            'economic_benefit': total_benefit,
            'net_benefit': total_benefit - total_cost,
            'roi_percentage': roi,
            'cis_improvement': cis_improvement,
            'cost_per_enrollment': total_cost / total_increase if total_increase > 0 else 0,
            'priority_districts': priority_districts['district'].tolist(),
            'priority_states': priority_districts['state'].tolist()
        }


# Define scenarios
scenarios = [
    PolicyScenario(
        name='Mobile Aadhaar Camps',
        cost_per_district=CONFIG.MOBILE_CAMP_COST,
        impact_rate=CONFIG.MOBILE_CAMP_IMPACT,
        description='Deploy mobile enrollment centers in underserved districts',
        target_group='both'
    ),
    PolicyScenario(
        name='School Enrollment Drives',
        cost_per_district=CONFIG.SCHOOL_DRIVE_COST,
        impact_rate=CONFIG.SCHOOL_DRIVE_IMPACT,
        description='Partner with schools for child Aadhaar enrollment campaigns',
        target_group='child'
    ),
    PolicyScenario(
        name='Senior Citizen Outreach',
        cost_per_district=CONFIG.SENIOR_OUTREACH_COST,
        impact_rate=CONFIG.SENIOR_OUTREACH_IMPACT,
        description='Door-to-door enrollment for elderly citizens',
        target_group='adult'
    ),
    PolicyScenario(
        name='Digital Literacy Programs',
        cost_per_district=CONFIG.DIGITAL_LITERACY_COST,
        impact_rate=CONFIG.DIGITAL_LITERACY_IMPACT,
        description='Train community workers on Aadhaar enrollment process',
        target_group='both'
    )
]

print(f"✅ {len(scenarios)} policy scenarios defined:")
for scenario in scenarios:
    print(f"   • {scenario.name}: ₹{scenario.cost_per_district:,}/district/month (Target: {scenario.target_group})")

# =============================================================================
# STAGE 3: SCENARIO COMPARISON
# =============================================================================
print("\n📊 STAGE 3: Scenario Impact Analysis...")

budget = CONFIG.DEFAULT_BUDGET
months = CONFIG.IMPLEMENTATION_MONTHS

scenario_results = []

for scenario in scenarios:
    max_districts = int(budget / (scenario.cost_per_district * months))
    max_districts = min(max_districts, len(baseline))
    
    if max_districts > 0:
        impact = scenario.calculate_impact(baseline, max_districts, months)
        scenario_results.append(impact)
        
        print(f"\n🎯 {scenario.name}:")
        print(f"   • Districts covered: {impact['districts_covered']}")
        print(f"   • Total cost: ₹{impact['total_cost']:,.0f}")
        print(f"   • Enrollment increase: {impact['total_enrollment_increase']:,.0f}")
        print(f"      └─ Children: {impact['child_enrollment_increase']:,.0f}")
        print(f"      └─ Adults: {impact['adult_enrollment_increase']:,.0f}")
        print(f"   • Economic benefit: ₹{impact['economic_benefit']:,.0f}")
        print(f"   • ROI: {impact['roi_percentage']:,.1f}%")
        print(f"   • Cost per enrollment: ₹{impact['cost_per_enrollment']:,.0f}")

comparison_df = pd.DataFrame(scenario_results)

best_roi = comparison_df.loc[comparison_df['roi_percentage'].idxmax()]
best_impact = comparison_df.loc[comparison_df['total_enrollment_increase'].idxmax()]
best_efficiency = comparison_df.loc[comparison_df['cost_per_enrollment'].idxmin()]

print(f"\n🏆 BEST SCENARIOS:")
print(f"   • Highest ROI: {best_roi['scenario']} ({best_roi['roi_percentage']:.1f}%)")
print(f"   • Maximum Impact: {best_impact['scenario']} ({best_impact['total_enrollment_increase']:,.0f} enrollments)")
print(f"   • Most Cost-Efficient: {best_efficiency['scenario']} (₹{best_efficiency['cost_per_enrollment']:,.0f}/enrollment)")

# =============================================================================
# STAGE 4: OPTIMAL BUDGET ALLOCATION
# =============================================================================
print("\n💰 STAGE 4: Optimal Budget Allocation...")

# Allocate: 60% to best ROI, 30% to second best, 10% to third
sorted_by_roi = comparison_df.sort_values('roi_percentage', ascending=False)
allocation_pcts = [0.60, 0.30, 0.10]

optimal_allocation = {}
total_impact = 0

for i, (idx, row) in enumerate(sorted_by_roi.head(3).iterrows()):
    budget_allocated = CONFIG.DEFAULT_BUDGET * allocation_pcts[i]
    scenario_obj = scenarios[i % len(scenarios)]
    
    max_districts = int(budget_allocated / (scenario_obj.cost_per_district * months))
    max_districts = min(max_districts, len(baseline))
    
    if max_districts > 0:
        impact = scenario_obj.calculate_impact(baseline, max_districts, months)
        optimal_allocation[row['scenario']] = {
            'budget': budget_allocated,
            'districts': max_districts,
            'impact': impact['total_enrollment_increase'],
            'roi': impact['roi_percentage']
        }
        total_impact += impact['total_enrollment_increase']

print(f"\n📊 Optimal Allocation (₹{CONFIG.DEFAULT_BUDGET:,} budget):")
for scenario_name, alloc in optimal_allocation.items():
    if alloc['budget'] > 0:
        print(f"\n   • {scenario_name}:")
        print(f"      Budget: ₹{alloc['budget']:,.0f} ({alloc['budget']/CONFIG.DEFAULT_BUDGET*100:.0f}%)")
        print(f"      Districts: {alloc['districts']}")
        print(f"      Enrollment increase: {alloc['impact']:,.0f}")
        print(f"      ROI: {alloc['roi']:.1f}%")

print(f"\n   📈 Total Expected Impact: {total_impact:,.0f} enrollments")

# =============================================================================
# STAGE 5: VISUALIZATIONS
# =============================================================================
print("\n📊 STAGE 5: Generating Interactive Dashboards...")

# VIZ 1: Scenario Comparison
fig1 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('ROI Comparison', 'Total Enrollment Impact')
)

fig1.add_trace(
    go.Bar(x=comparison_df['scenario'], y=comparison_df['roi_percentage'],
           marker_color='green', name='ROI %', text=comparison_df['roi_percentage'].apply(lambda x: f'{x:.1f}%'),
           textposition='outside'),
    row=1, col=1
)

fig1.add_trace(
    go.Bar(x=comparison_df['scenario'], y=comparison_df['total_enrollment_increase'],
           marker_color='blue', name='Enrollments', text=comparison_df['total_enrollment_increase'].apply(lambda x: f'{x:,.0f}'),
           textposition='outside'),
    row=1, col=2
)

fig1.update_xaxes(tickangle=-45)
fig1.update_layout(height=500, showlegend=False, title_text=f'Policy Scenario Comparison (₹{CONFIG.DEFAULT_BUDGET:,} Budget)')
fig1.write_html(CONFIG.OUTPUT_DIR / 'module5_scenario_comparison.html')

# VIZ 2: Cost Efficiency
fig2 = px.bar(
    comparison_df.sort_values('cost_per_enrollment'),
    x='scenario',
    y='cost_per_enrollment',
    title='Cost per Enrollment (Lower is Better)',
    labels={'cost_per_enrollment': 'Cost per Enrollment (₹)', 'scenario': 'Scenario'},
    color='cost_per_enrollment',
    color_continuous_scale='RdYlGn_r',
    text='cost_per_enrollment'
)
fig2.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
fig2.update_layout(height=500, showlegend=False)
fig2.write_html(CONFIG.OUTPUT_DIR / 'module5_cost_efficiency.html')

# VIZ 3: Demographic Impact
demographic_impact = []
for result in scenario_results:
    demographic_impact.extend([
        {'Scenario': result['scenario'], 'Age Group': 'Children (0-17)', 'Increase': result['child_enrollment_increase']},
        {'Scenario': result['scenario'], 'Age Group': 'Adults (18+)', 'Increase': result['adult_enrollment_increase']}
    ])

demo_df = pd.DataFrame(demographic_impact)

fig3 = px.bar(
    demo_df,
    x='Scenario',
    y='Increase',
    color='Age Group',
    title='Enrollment Impact by Demographic Group',
    labels={'Increase': 'Enrollment Increase'},
    barmode='group'
)
fig3.update_xaxes(tickangle=-45)
fig3.update_layout(height=600)
fig3.write_html(CONFIG.OUTPUT_DIR / 'module5_demographic_impact.html')

# VIZ 4: Optimal Allocation
allocation_viz = [{'Scenario': k, 'Budget': v['budget']} for k, v in optimal_allocation.items() if v['budget'] > 0]
alloc_df = pd.DataFrame(allocation_viz)

fig4 = px.pie(
    alloc_df,
    values='Budget',
    names='Scenario',
    title=f'Optimal Budget Allocation (₹{CONFIG.DEFAULT_BUDGET:,})',
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig4.update_traces(textposition='inside', textinfo='percent+label')
fig4.update_layout(height=600)
fig4.write_html(CONFIG.OUTPUT_DIR / 'module5_optimal_allocation.html')

# VIZ 5: ROI vs Impact Scatter
fig5 = px.scatter(
    comparison_df,
    x='total_enrollment_increase',
    y='roi_percentage',
    size='districts_covered',
    color='scenario',
    title='ROI vs Total Impact (Bubble Size = Districts Covered)',
    labels={'total_enrollment_increase': 'Total Enrollment Increase', 'roi_percentage': 'ROI (%)'},
    hover_data=['cost_per_enrollment']
)
fig5.update_layout(height=600)
fig5.write_html(CONFIG.OUTPUT_DIR / 'module5_roi_vs_impact.html')

print("✅ Visualizations saved:")
print(f"   • {CONFIG.OUTPUT_DIR}/module5_scenario_comparison.html")
print(f"   • {CONFIG.OUTPUT_DIR}/module5_cost_efficiency.html")
print(f"   • {CONFIG.OUTPUT_DIR}/module5_demographic_impact.html")
print(f"   • {CONFIG.OUTPUT_DIR}/module5_optimal_allocation.html")
print(f"   • {CONFIG.OUTPUT_DIR}/module5_roi_vs_impact.html")

# =============================================================================
# STAGE 6: SAVE RESULTS
# =============================================================================
print("\n💾 STAGE 6: Saving Results...")

recommendations = {
    'timestamp': datetime.now().isoformat(),
    'budget_analyzed': CONFIG.DEFAULT_BUDGET,
    'implementation_months': CONFIG.IMPLEMENTATION_MONTHS,
    'best_roi_scenario': {
        'name': best_roi['scenario'],
        'roi': float(best_roi['roi_percentage']),
        'total_impact': float(best_roi['total_enrollment_increase'])
    },
    'optimal_allocation': {k: {ki: float(vi) if isinstance(vi, (np.floating, np.integer)) else int(vi) if isinstance(vi, (int, np.int_)) else vi 
                                for ki, vi in v.items()} 
                           for k, v in optimal_allocation.items()},
    'total_expected_impact': float(total_impact)
}

with open(CONFIG.OUTPUT_DIR / 'module5_policy_recommendations.json', 'w') as f:
    json.dump(recommendations, f, indent=2)

comparison_df.to_csv(CONFIG.OUTPUT_DIR / 'module5_scenario_comparison.csv', index=False)

# Top 20 priority districts
for result in scenario_results:
    priority_df = pd.DataFrame({
        'district': result['priority_districts'][:20],
        'state': result['priority_states'][:20],
        'scenario': result['scenario']
    })
    priority_df.to_csv(
        CONFIG.OUTPUT_DIR / f"module5_priority_districts_{result['scenario'].replace(' ', '_').lower()}.csv",
        index=False
    )

print("✅ Results saved:")
print(f"   • {CONFIG.OUTPUT_DIR}/module5_policy_recommendations.json")
print(f"   • {CONFIG.OUTPUT_DIR}/module5_scenario_comparison.csv")
print(f"   • {CONFIG.OUTPUT_DIR}/module5_priority_districts_*.csv (4 files)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("🎉 MODULE 5: POLICY IMPACT SIMULATOR COMPLETE!")
print("="*80)

print(f"\n🏆 KEY RECOMMENDATIONS:")
print(f"   • Best ROI Scenario: {best_roi['scenario']} ({best_roi['roi_percentage']:.1f}%)")
print(f"   • Maximum Impact: {best_impact['scenario']} ({best_impact['total_enrollment_increase']:,.0f} enrollments)")
print(f"   • Most Cost-Efficient: {best_efficiency['scenario']} (₹{best_efficiency['cost_per_enrollment']:,.0f}/enrollment)")

print(f"\n💰 OPTIMAL ALLOCATION (₹{CONFIG.DEFAULT_BUDGET:,}):")
for scenario_name, alloc in optimal_allocation.items():
    pct = alloc['budget'] / CONFIG.DEFAULT_BUDGET * 100
    print(f"   • {scenario_name}: ₹{alloc['budget']:,.0f} ({pct:.0f}%) → {alloc['impact']:,.0f} enrollments")

print(f"\n📈 TOTAL EXPECTED IMPACT: {total_impact:,.0f} enrollments over {CONFIG.IMPLEMENTATION_MONTHS} months")

print(f"\n✨ CAPABILITIES DEMONSTRATED:")
print("   ✅ Multi-scenario policy analysis (4 scenarios)")
print("   ✅ Cost-benefit optimization")
print("   ✅ Demographic-targeted interventions")
print("   ✅ ROI-driven recommendations")
print("   ✅ Priority district identification")
print("   ✅ Interactive what-if dashboards")

print("\n🚀 Ready for policy presentation!")
print("="*80)
