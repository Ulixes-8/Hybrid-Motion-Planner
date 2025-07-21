# %% [markdown]
# # NuBoard Visualization: test14-hard Comparison
# 
# Compare PDM-Closed, Hybrid, and Diffusion planners on all 272 test14-hard scenarios

# %%
import os
import sys
from pathlib import Path

# Setup environment
home = Path.home()
sys.path.extend([
    str(home / "chimera"),
    str(home / "chimera" / "nuplan-devkit"),
    str(home / "chimera" / "tuplan_garage"),
    str(home / "chimera" / "diffusion_planner"),
])

os.environ.update({
    "NUPLAN_DEVKIT_ROOT": str(home / "chimera" / "nuplan-devkit"),
    "NUPLAN_DATA_ROOT": str(home / "nuplan" / "dataset"),
    "NUPLAN_MAPS_ROOT": str(home / "nuplan" / "dataset" / "maps"),
    "NUPLAN_EXP_ROOT": str(home / "nuplan" / "exp"),
})

print("✅ Environment configured")

# %% [markdown]
# ## Define Simulation Paths

# %%
# Your full test14-hard reactive runs
simulation_results = {
    "PDM-Closed": "/home/ulixes/nuplan/exp/exp/simulation/closed_loop_reactive_agents/pdm_closed_planner/test14-hard/baseline/2025-07-10-14-37-28",
    # "Hybrid": "/home/ulixes/nuplan/exp/exp/simulation/closed_loop_reactive_agents/hybrid_planner/test14-hard/hybrid_planner_fixed/2025-07-09-15-58-13_full",
    "Diffusion": "/home/ulixes/nuplan/exp/exp/simulation/closed_loop_reactive_agents/diffusion_planner/test14-hard/diffusion_planner_release/model_2025-07-04-15-14-17",
    "Hybrid MINI": "/home/ulixes/nuplan/exp/exp/simulation/closed_loop_reactive_agents/hybrid_planner/test14-hard/hybrid_planner_fixed/2025-07-11-22-32-06",
    "Hybrid DDM MINI": "/home/ulixes/nuplan/exp/exp/simulation/closed_loop_reactive_agents/hybrid_planner/test14-hard/hybrid_planner_fixed/2025-07-21-20-20-33",


    # This is refinement 1
    # "Diffusion Ref 1 MINI": "/home/ulixes/nuplan/exp/exp/simulation/closed_loop_reactive_agents/diffusion_planner/test14-hard/diffusion_planner_release/model_2025-07-15-11-31-44", # This is Refinement 2
    # "Diffusion Ref 2 MINI": "/home/ulixes/nuplan/exp/exp/simulation/closed_loop_reactive_agents/diffusion_planner/test14-hard/diffusion_planner_release/model_2025-07-15-15-39-07",
    # # Refinement 3
    # "Diffusion Ref 3 MINI": "/home/ulixes/nuplan/exp/exp/simulation/closed_loop_reactive_agents/diffusion_planner/test14-hard/diffusion_planner_release/model_2025-07-16-11-50-38",
    # # Refinement 4
    # "Diffusion Ref 4 MINI": "/home/ulixes/nuplan/exp/exp/simulation/closed_loop_reactive_agents/diffusion_planner/test14-hard/diffusion_planner_release/model_2025-07-16-14-16-29"
    # "Diffion Refined": "/home/ulixes/nuplan/exp/exp/simulation/closed_loop_reactive_agents/diffusion_planner/test14-hard/diffusion_planner_release/model_2025-07-16-16-20-47",

}

# Verify paths exist
valid_paths = []
for name, path in simulation_results.items():
    p = Path(path)
    if p.exists():
        nuboard_count = len(list(p.rglob("*.nuboard")))
        print(f"✅ {name}: {nuboard_count} scenarios found")
        valid_paths.append(path)
    else:
        print(f"❌ {name}: Path not found!")

# %% [markdown]
# ## Check Available Metrics

# %%
import pandas as pd

print("📊 Checking for metrics:")
print("-" * 50)

for name, path in simulation_results.items():
    p = Path(path)
    metrics_file = p / "aggregated_metrics.parquet"
    
    print(f"\n{name}:")
    if metrics_file.exists():
        try:
            df = pd.read_parquet(metrics_file)
            score = df['closed_loop_cls_weighted_average'].mean()
            print(f"  Overall Score: {score:.4f}")
            
            # Show key metrics
            key_metrics = ['no_ego_at_fault_collisions', 'ego_is_comfortable', 'ego_is_making_progress']
            for metric in key_metrics:
                if metric in df.columns:
                    print(f"  {metric}: {df[metric].mean():.4f}")
        except Exception as e:
            print(f"  Error reading metrics: {e}")
    else:
        print("  No aggregated metrics (visualization only)")

# %% [markdown]
# ## Launch NuBoard

# %%
import hydra
from nuplan.planning.script.run_nuboard import main as main_nuboard

# Configuration
CONFIG_PATH = str(home / "chimera" / "nuplan-devkit" / "nuplan" / "planning" / "script" / "config" / "nuboard")

# Initialize Hydra
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=CONFIG_PATH)

# Create configuration with all three planners
cfg = hydra.compose(config_name='default_nuboard', overrides=[
    'scenario_builder=nuplan',
    f'simulation_path={valid_paths}',
    'port_number=6599'
])

print("\n🚀 Launching NuBoard...")
print("🌐 Open your browser at: http://localhost:6599")
print("\n💡 Tips:")
print("• Use dropdown to switch between PDM, Hybrid, and Diffusion")
print("• Select same scenario across planners to compare")
print("• Look for differences in lane change decisions")
print("• Compare collision avoidance strategies")

# This will block until you close NuBoard
main_nuboard(cfg)