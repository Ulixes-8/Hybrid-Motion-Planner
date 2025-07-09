#!/bin/bash
# Minimal test for simulation with just 1 scenario

cd ~/chimera
source chimera/scripts/setup_env.sh

echo "Testing hybrid planner with minimal configuration..."

export PYTHONPATH="$HOME/chimera:$HOME/chimera/nuplan-devkit:$HOME/chimera/tuplan_garage:$HOME/chimera/diffusion_planner:$PYTHONPATH"

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=hybrid_planner \
    scenario_filter=test14-hard \
    scenario_builder=nuplan_challenge \
    scenario_filter.limit_total_scenarios=1 \
    experiment_uid=hybrid_planner/debug/$(date '+%Y-%m-%d-%H-%M-%S') \
    verbose=true \
    worker=sequential \
    hydra.searchpath="[pkg://chimera.config, pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"