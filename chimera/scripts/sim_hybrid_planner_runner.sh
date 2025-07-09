#!/bin/bash
# sim_hybrid_planner_runner.sh
# Run hybrid planner on test14-hard benchmark

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

###################################
# Environment Configuration
###################################
export NUPLAN_DEVKIT_ROOT="$HOME/nuplan-devkit"
export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"

# Add chimera to Python path
export PYTHONPATH="$HOME/chimera:$HOME/chimera/nuplan-devkit:$HOME/chimera/tuplan_garage:$HOME/chimera/diffusion_planner:$PYTHONPATH"

cd $HOME/chimera

SPLIT="test14-hard"
CHALLENGE="closed_loop_nonreactive_agents"
EXPERIMENT_ID="hybrid_planner/$SPLIT/test/$(date '+%Y-%m-%d-%H-%M-%S')"

echo "Running Hybrid Planner on test14-hard"
echo "Experiment ID: $EXPERIMENT_ID"

# Run simulation with correct searchpath for chimera configs
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=hybrid_planner \
    scenario_filter=$SPLIT \
    scenario_builder=nuplan_challenge \
    scenario_filter.limit_total_scenarios=5 \
    experiment_uid=$EXPERIMENT_ID \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=32 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.25 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://chimera.config.planner, pkg://chimera.config.simulation, pkg://chimera.config.scenario_filter, pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"