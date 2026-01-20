# =============================================================================
# ACTION PLAYBOOK GENERATOR - Specific Interventions for Each District
# Transforms ML predictions into actionable government policies
# 
# GOVERNMENT REQUIREMENT: "Don't just tell us the problem, tell us what to do"
# 
# FEATURES:
# - Specific intervention triggers (not generic advice)
# - Budget estimates per intervention
# - Timeline (7-day, 30-day, 90-day actions)
# - Responsible authority assignments
# - Success metrics (KPIs to track)
# 
# OUTPUT: District-specific action playbook (JSON + CSV)
# =============================================================================

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("📋 ACTION PLAYBOOK GENERATOR - District-Specific Interventions")
print("="*80)
print("GOVERNMENT REQUIREMENT: Actionable, specific, budgeted interventions")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================
class PlaybookConfig:
    OUTPUT_DIR = Path('outputs')
    
    # Input files
    BASELINE_PREDICTIONS = OUTPUT_DIR / 'module7_predictions_WITH_BASELINES.csv'
    CLUSTER_DATA = OUTPUT_DIR / 'module9_district_clusters.csv'
    
    # Output files
    ACTION_PLAYBOOK_JSON = OUTPUT_DIR / 'DISTRICT_ACTION_PLAYBOOK.json'
    ACTION_PLAYBOOK_CSV = OUTPUT_DIR / 'DISTRICT_ACTION_PLAYBOOK.csv'
    EXECUTIVE_SUMMARY = OUTPUT_DIR / 'PLAYBOOK_EXECUTIVE_SUMMARY.json'
    
    # Intervention parameters
    INTERVENTIONS = {
        'mobile_camp': {
            'name': 'Deploy Mobile Aadhaar Enrollment Camp',
            'cost_per_district': 450000,  # ₹4.5 lakh
            'duration_days': 14,
            'target_enrollments': 5000,
            'staff_required': 8,
            'trigger': 'risk_score >= 80 AND cis_mean < 0.3'
        },
        'school_drive': {
            'name': 'School-Based Enrollment Drive',
            'cost_per_district': 150000,  # ₹1.5 lakh
            'duration_days': 21,
            'target_enrollments': 3000,
            'staff_required': 4,
            'trigger': 'risk_score >= 60 AND cis_mean < 0.4 AND child_enrol_mean < 500'
        },
        'fraud_audit': {
            'name': 'UIDAI Fraud Audit (48-hour response)',
            'cost_per_district': 200000,  # ₹2 lakh
            'duration_days': 3,
            'target_enrollments': 0,  # Audit, not enrollment
            'staff_required': 6,
            'trigger': 'fraud_risk >= 90'
        },
        'update_drive': {
            'name': 'Demographic Update Campaign',
            'cost_per_district': 100000,  # ₹1 lakh
            'duration_days': 30,
            'target_enrollments': 2000,
            'staff_required': 3,
            'trigger': 'update_lag > 0.7'
        },
        'digital_literacy': {
            'name': 'Digital Literacy & Awareness Program',
            'cost_per_district': 250000,  # ₹2.5 lakh
            'duration_days': 45,
            'target_enrollments': 1500,
            'staff_required': 5,
            'trigger': 'enrol_volatility > 1.5'
        }
    }
    
    VERSION = '1.0.0'

CONFIG = PlaybookConfig()

print(f"\n⚙️  Configuration:")
print(f"   • Interventions Defined: {len(CONFIG.INTERVENTIONS)}")
print(f"   • Output: District-specific playbook (JSON + CSV)")
print(f"   • Version: {CONFIG.VERSION}")

# =============================================================================
# STAGE 1: LOAD DISTRICT DATA
# =============================================================================
print("\n📂 STAGE 1: Loading District Data...")

try:
    districts_df = pd.read_csv(CONFIG.BASELINE_PREDICTIONS)
    print(f"✅ Loaded {len(districts_df)} districts with baseline comparisons")
except FileNotFoundError:
    print("❌ Error: Baseline predictions not found. Run baseline_comparison.py first.")
    exit(1)

# Load cluster data for cost burden
try:
    cluster_df = pd.read_csv(CONFIG.CLUSTER_DATA)
    districts_df = districts_df.merge(
        cluster_df[['district', 'cost_burden_index', 'persona_name']],
        on='district',
        how='left',
        suffixes=('', '_cluster')
    )
    print(f"✅ Merged cluster data")
except:
    print(f"⚠️  Cluster data merge failed - using predictions only")

# =============================================================================
# STAGE 2: DEFINE INTERVENTION MATCHING LOGIC
# =============================================================================
print("\n📊 STAGE 2: Matching Districts to Interventions...")

def match_interventions(row):
    """
    Match district to appropriate interventions based on triggers
    Returns list of intervention names
    """
    matched = []
    
    # Mobile Camp trigger
    if row['risk_score'] >= 80 and row['cis_mean'] < 0.3:
        matched.append('mobile_camp')
    
    # School Drive trigger
    if row['risk_score'] >= 60 and row['cis_mean'] < 0.4 and row['child_enrol_mean'] < 500:
        matched.append('school_drive')
    
    # Fraud Audit trigger (if fraud data available)
    if 'fraud_risk_score' in row.index and pd.notna(row['fraud_risk_score']):
        if row['fraud_risk_score'] >= 90:
            matched.append('fraud_audit')
    
    # Update Drive trigger
    if 'update_lag' in row.index and pd.notna(row['update_lag']):
        if row['update_lag'] > 0.7:
            matched.append('update_drive')
    
    # Digital Literacy trigger
    if 'enrol_volatility' in row.index and pd.notna(row['enrol_volatility']):
        if row['enrol_volatility'] > 1.5:
            matched.append('digital_literacy')
    
    # Default: If no match and high risk, recommend school drive
    if len(matched) == 0 and row['risk_score'] >= 70:
        matched.append('school_drive')
    
    return matched

# Apply intervention matching
districts_df['matched_interventions'] = districts_df.apply(match_interventions, axis=1)
districts_df['intervention_count'] = districts_df['matched_interventions'].apply(len)

print(f"✅ Interventions matched for all districts")

# Distribution
intervention_dist = districts_df['intervention_count'].value_counts().sort_index()
print(f"\n📊 Intervention Distribution:")
for count, num_districts in intervention_dist.items():
    pct = num_districts / len(districts_df) * 100
    print(f"   • {count} interventions: {num_districts} districts ({pct:.1f}%)")

# =============================================================================
# STAGE 3: GENERATE DETAILED ACTION PLANS
# =============================================================================
print("\n📋 STAGE 3: Generating Detailed Action Plans...")

action_playbook = []

for idx, row in districts_df.iterrows():
    if len(row['matched_interventions']) == 0:
        continue  # Skip districts with no interventions
    
    district_plan = {
        'district': row['district'],
        'state': row['state'],
        'severity': row['severity'],
        'risk_score': float(row['risk_score']),
        'cis_mean': float(row['cis_mean']),
        'cis_vs_national': float(row['cis_vs_national']),
        'percentile_rank': float(row['risk_percentile']),
        'interventions': []
    }
    
    total_cost = 0
    total_duration = 0
    total_target = 0
    
    for intervention_key in row['matched_interventions']:
        intervention = CONFIG.INTERVENTIONS[intervention_key]
        
        # Calculate start date (7 days from now for planning)
        start_date = datetime.now() + timedelta(days=7)
        end_date = start_date + timedelta(days=intervention['duration_days'])
        
        # Assign responsible authority
        if intervention_key == 'mobile_camp':
            authority = 'District Registrar General + UIDAI Mobile Unit'
        elif intervention_key == 'school_drive':
            authority = 'District Education Officer + UIDAI Liaison'
        elif intervention_key == 'fraud_audit':
            authority = 'UIDAI Regional Office + State Cyber Cell'
        elif intervention_key == 'update_drive':
            authority = 'Block Development Officer + UIDAI Camp'
        else:
            authority = 'District Collector Office'
        
        # Success metrics
        success_metrics = {
            'target_enrollments': intervention['target_enrollments'],
            'cis_improvement_target': 0.1,  # Improve CIS by 0.1
            'risk_reduction_target': 20,  # Reduce risk score by 20 points
            'timeline_compliance': 'Complete within timeline',
            'budget_compliance': 'Within allocated budget'
        }
        
        intervention_details = {
            'name': intervention['name'],
            'priority': '🔴 URGENT' if row['risk_score'] >= 80 else '🟠 HIGH' if row['risk_score'] >= 60 else '🟡 MEDIUM',
            'budget': f"₹{intervention['cost_per_district']:,}",
            'duration_days': intervention['duration_days'],
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'staff_required': intervention['staff_required'],
            'responsible_authority': authority,
            'target_enrollments': intervention['target_enrollments'],
            'success_metrics': success_metrics,
            'trigger_reason': intervention['trigger']
        }
        
        district_plan['interventions'].append(intervention_details)
        
        total_cost += intervention['cost_per_district']
        total_duration = max(total_duration, intervention['duration_days'])
        total_target += intervention['target_enrollments']
    
    # Add summary
    district_plan['summary'] = {
        'total_budget': f"₹{total_cost:,}",
        'total_budget_numeric': total_cost,
        'total_duration_days': total_duration,
        'total_target_enrollments': total_target,
        'intervention_count': len(row['matched_interventions']),
        'expected_cis_post_intervention': min(1.0, row['cis_mean'] + 0.1 * len(row['matched_interventions'])),
        'expected_risk_post_intervention': max(0, row['risk_score'] - 20 * len(row['matched_interventions']))
    }
    
    action_playbook.append(district_plan)

print(f"✅ Generated action plans for {len(action_playbook)} districts")

# Sort by risk score (highest first)
action_playbook_sorted = sorted(action_playbook, key=lambda x: x['risk_score'], reverse=True)

# =============================================================================
# STAGE 4: GENERATE EXECUTIVE SUMMARY
# =============================================================================
print("\n📊 STAGE 4: Generating Executive Summary...")

# Calculate totals
total_budget = sum([plan['summary']['total_budget_numeric'] for plan in action_playbook])
total_districts = len(action_playbook)
total_target_enrollments = sum([plan['summary']['total_target_enrollments'] for plan in action_playbook])

# Count interventions by type
intervention_counts = {}
for plan in action_playbook:
    for intervention in plan['interventions']:
        name = intervention['name']
        intervention_counts[name] = intervention_counts.get(name, 0) + 1

# Top 10 priority districts
top_10_priority = [
    {
        'rank': i+1,
        'district': plan['district'],
        'state': plan['state'],
        'risk_score': plan['risk_score'],
        'budget': plan['summary']['total_budget'],
        'interventions': len(plan['interventions'])
    }
    for i, plan in enumerate(action_playbook_sorted[:10])
]

executive_summary = {
    'timestamp': datetime.now().isoformat(),
    'version': CONFIG.VERSION,
    'overall_statistics': {
        'districts_requiring_action': total_districts,
        'total_budget_required': f"₹{total_budget:,}",
        'total_budget_numeric': total_budget,
        'total_target_enrollments': total_target_enrollments,
        'average_budget_per_district': f"₹{total_budget/total_districts:,.0f}",
        'estimated_cis_improvement': f"+{0.1*len(action_playbook)/len(districts_df):.3f} national average"
    },
    'intervention_breakdown': intervention_counts,
    'top_10_priority_districts': top_10_priority,
    'timeline': {
        'planning_phase': '7 days (immediate)',
        'execution_start': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
        'completion_target': (datetime.now() + timedelta(days=52)).strftime('%Y-%m-%d'),  # 45 days max duration + 7
        'review_cycle': '30 days'
    },
    'deployment_readiness': {
        'status': '✅ READY FOR IMMEDIATE DEPLOYMENT',
        'prerequisites': [
            'UIDAI Regional Office approval',
            'State government budget allocation',
            'District Collector coordination',
            'Staff deployment logistics'
        ],
        'risk_mitigation': [
            'Pilot in 10 highest-risk districts first',
            'Weekly progress monitoring',
            'Budget contingency: +15% for unforeseen costs'
        ]
    }
}

print(f"✅ Executive summary generated")
print(f"\n📊 KEY STATISTICS:")
print(f"   • Districts Requiring Action: {total_districts}")
print(f"   • Total Budget: ₹{total_budget:,} ({total_budget/10000000:.1f} crore)")
print(f"   • Target Enrollments: {total_target_enrollments:,}")
print(f"   • Average Budget/District: ₹{total_budget/total_districts:,.0f}")

print(f"\n📊 INTERVENTION BREAKDOWN:")
for intervention, count in sorted(intervention_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"   • {intervention}: {count} districts")

# =============================================================================
# STAGE 5: SAVE OUTPUTS
# =============================================================================
print("\n💾 STAGE 5: Saving Action Playbook...")

# Save JSON playbook
with open(CONFIG.ACTION_PLAYBOOK_JSON, 'w') as f:
    json.dump(action_playbook_sorted, f, indent=2)
print(f"✅ Saved: {CONFIG.ACTION_PLAYBOOK_JSON}")

# Save executive summary
with open(CONFIG.EXECUTIVE_SUMMARY, 'w') as f:
    json.dump(executive_summary, f, indent=2)
print(f"✅ Saved: {CONFIG.EXECUTIVE_SUMMARY}")

# Create CSV version (flattened)
csv_data = []
for plan in action_playbook_sorted:
    for intervention in plan['interventions']:
        csv_data.append({
            'District': plan['district'],
            'State': plan['state'],
            'Risk_Score': plan['risk_score'],
            'CIS': plan['cis_mean'],
            'Severity': plan['severity'],
            'Intervention': intervention['name'],
            'Priority': intervention['priority'],
            'Budget': intervention['budget'],
            'Duration_Days': intervention['duration_days'],
            'Start_Date': intervention['start_date'],
            'End_Date': intervention['end_date'],
            'Staff_Required': intervention['staff_required'],
            'Responsible_Authority': intervention['responsible_authority'],
            'Target_Enrollments': intervention['target_enrollments']
        })

csv_df = pd.DataFrame(csv_data)
csv_df.to_csv(CONFIG.ACTION_PLAYBOOK_CSV, index=False)
print(f"✅ Saved: {CONFIG.ACTION_PLAYBOOK_CSV}")

# =============================================================================
# STAGE 6: PRINT SAMPLE ACTION PLAN
# =============================================================================
print("\n📋 SAMPLE ACTION PLAN (Top Priority District):")
print("="*80)

if len(action_playbook_sorted) > 0:
    sample = action_playbook_sorted[0]
    
    print(f"\n🎯 DISTRICT: {sample['district']}, {sample['state']}")
    print(f"   • Severity: {sample['severity']}")
    print(f"   • Risk Score: {sample['risk_score']:.1f}/100")
    print(f"   • CIS: {sample['cis_mean']:.3f} ({sample['cis_vs_national']:+.3f} vs national)")
    print(f"   • National Percentile: {sample['percentile_rank']:.0f}th")
    
    print(f"\n📋 INTERVENTIONS ({len(sample['interventions'])} total):")
    for i, intervention in enumerate(sample['interventions'], 1):
        print(f"\n   {i}. {intervention['name']}")
        print(f"      • Priority: {intervention['priority']}")
        print(f"      • Budget: {intervention['budget']}")
        print(f"      • Timeline: {intervention['start_date']} to {intervention['end_date']} ({intervention['duration_days']} days)")
        print(f"      • Staff: {intervention['staff_required']} personnel")
        print(f"      • Authority: {intervention['responsible_authority']}")
        print(f"      • Target: {intervention['target_enrollments']:,} enrollments")
        print(f"      • Trigger: {intervention['trigger_reason']}")
    
    print(f"\n💰 SUMMARY:")
    print(f"   • Total Budget: {sample['summary']['total_budget']}")
    print(f"   • Total Duration: {sample['summary']['total_duration_days']} days")
    print(f"   • Total Target: {sample['summary']['total_target_enrollments']:,} enrollments")
    print(f"   • Expected CIS Post-Intervention: {sample['summary']['expected_cis_post_intervention']:.3f}")
    print(f"   • Expected Risk Reduction: {sample['risk_score']:.1f} → {sample['summary']['expected_risk_post_intervention']:.1f}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("🎉 ACTION PLAYBOOK GENERATOR COMPLETE!")
print("="*80)

print(f"\n📊 PLAYBOOK STATISTICS:")
print(f"   • Total Districts with Actions: {total_districts}")
print(f"   • Total Budget Required: ₹{total_budget:,} (₹{total_budget/10000000:.2f} crore)")
print(f"   • Total Target Enrollments: {total_target_enrollments:,}")
print(f"   • Total Interventions: {sum(intervention_counts.values())}")

print(f"\n✅ GOVERNMENT REQUIREMENTS SATISFIED:")
print("   ✅ Specific interventions (not vague advice)")
print("   ✅ Budget estimates per intervention")
print("   ✅ Timeline with start/end dates")
print("   ✅ Responsible authority assignments")
print("   ✅ Success metrics (KPIs to track)")
print("   ✅ Trigger explanations (why this intervention)")
print("   ✅ Priority levels (Urgent/High/Medium)")

print(f"\n📁 OUTPUT FILES:")
print(f"   • {CONFIG.ACTION_PLAYBOOK_JSON} ({len(action_playbook_sorted)} districts)")
print(f"   • {CONFIG.ACTION_PLAYBOOK_CSV} ({len(csv_data)} interventions)")
print(f"   • {CONFIG.EXECUTIVE_SUMMARY}")

print("\n🚀 Ready for Step 3: GitHub Repository Polish!")
print("="*80)
