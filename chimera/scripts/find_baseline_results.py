#!/usr/bin/env python3
"""
Check the PDM Run That Just Completed
=====================================
Verify the PDM baseline that just finished has proper metrics.
"""

from pathlib import Path
import pandas as pd

# The PDM run that just completed
pdm_run = Path("/home/ulixes/nuplan/exp/exp/simulation/closed_loop_reactive_agents/pdm_closed_planner/test14-hard/baseline/2025-07-10-14-37-28")

print("🔍 Checking PDM Baseline Run")
print("=" * 70)
print(f"Path: {pdm_run}")
print()

if not pdm_run.exists():
    print("❌ Directory not found!")
    exit(1)

# Check runner report
runner_report = pdm_run / "runner_report.parquet"
if runner_report.exists():
    df = pd.read_parquet(runner_report)
    print(f"✅ Runner report: {len(df)} scenarios")
    print(f"   Successful: {df['succeeded'].sum()}")
    print(f"   Failed: {(~df['succeeded']).sum()}")
else:
    print("❌ No runner report found")

# Check aggregated metrics
agg_metrics = pdm_run / "aggregated_metrics.parquet"
if agg_metrics.exists():
    print(f"\n✅ Aggregated metrics found!")
    try:
        df = pd.read_parquet(agg_metrics)
        print(f"   Scenarios: {len(df)}")
        
        # Show key metrics
        key_metrics = [
            'closed_loop_cls_weighted_average',
            'no_ego_at_fault_collisions',
            'ego_is_comfortable',
            'ego_is_making_progress',
            'time_to_collision_within_bound',
            'speed_limit_compliance',
            'drivable_area_compliance',
            'driving_direction_compliance'
        ]
        
        print("\n   Scores:")
        for metric in key_metrics:
            if metric in df.columns:
                score = df[metric].mean()
                print(f"   - {metric}: {score:.4f}")
        
        # Overall score
        if 'closed_loop_cls_weighted_average' in df.columns:
            overall = df['closed_loop_cls_weighted_average'].mean()
            print(f"\n   🏆 OVERALL SCORE: {overall:.4f}")
            
    except Exception as e:
        print(f"   ❌ Error reading metrics: {e}")
else:
    print("\n❌ No aggregated metrics found")
    print("   But the logs show 'Metric aggregator: 00:00:00 [HH:MM:SS]'")
    print("   Check if file was created after the script ended")

# Check for other indicators
print("\n📊 Other completion indicators:")

# Metrics directory
metrics_dir = pdm_run / "metrics"
if metrics_dir.exists():
    pickle_count = len(list(metrics_dir.glob("*.pickle")))
    print(f"✅ metrics/ directory: {pickle_count} pickle files")
else:
    print("❌ No metrics/ directory")

# Histogram plots
histograms = list(pdm_run.glob("metric_*.png"))
print(f"✅ Histogram plots: {len(histograms)} files")

# List all files in directory
print("\n📁 All files in directory:")
for f in sorted(pdm_run.iterdir()):
    if f.is_file():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   {f.name} ({size_mb:.1f} MB)")

# Now check other runs for comparison
print("\n\n🔍 Checking Other Complete Runs")
print("=" * 70)

base_dir = Path("/home/ulixes/nuplan/exp/exp/simulation/closed_loop_reactive_agents")

# Find all aggregated_metrics.parquet files
all_metrics = list(base_dir.rglob("**/test14-hard/**/aggregated_metrics.parquet"))

valid_runs = []
for metrics_file in all_metrics:
    try:
        df = pd.read_parquet(metrics_file)
        if len(df) >= 270:  # Close to 272
            score = df['closed_loop_cls_weighted_average'].mean()
            if score != 0.5:  # Not our fake default
                valid_runs.append({
                    'path': metrics_file.parent,
                    'planner': metrics_file.parts[-5],  # Extract planner name
                    'score': score,
                    'scenarios': len(df)
                })
    except:
        pass

if valid_runs:
    print("\nFound complete runs with valid metrics:")
    for run in sorted(valid_runs, key=lambda x: x['score'], reverse=True):
        print(f"\n{run['planner']}:")
        print(f"  Path: {run['path']}")
        print(f"  Score: {run['score']:.4f}")
        print(f"  Scenarios: {run['scenarios']}")