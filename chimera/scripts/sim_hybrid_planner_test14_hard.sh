#!/bin/bash
# sim_hybrid_planner_test14_hard_fixed.sh
# Version with automatic cleanup

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

###################################
# Cleanup Function
###################################
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    
    # Stop Ray
    ray stop --force 2>/dev/null || true
    
    # Kill related processes
    pkill -f "run_simulation.py" 2>/dev/null || true
    pkill -f "ray::" 2>/dev/null || true
    
    # Clear shared memory
    rm -rf /dev/shm/plasma_* 2>/dev/null || true
    rm -rf /dev/shm/ray* 2>/dev/null || true
    rm -rf /tmp/ray* 2>/dev/null || true
    
    # Clear any Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    
    echo "✅ Cleanup complete"
}

# Set trap to cleanup on exit (including Ctrl+C)
trap cleanup EXIT INT TERM

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

###################################
# Memory Settings
###################################
export RAY_memory_monitor_refresh_ms=0
export RAY_object_store_memory=1000000000    # 1GB
export RAY_plasma_directory=/tmp              
export RAY_DISABLE_IMPORT_WARNING=1

# System limits
ulimit -v unlimited  
ulimit -n 65536      

# Pre-cleanup to ensure clean start
cleanup

###################################
# Benchmark Configuration
###################################
SPLIT="test14-hard"
CHALLENGE="closed_loop_reactive_agents"
BRANCH_NAME="hybrid_planner_fixed"
PLANNER="hybrid_planner"
EXPERIMENT_ID="$PLANNER/$SPLIT/$BRANCH_NAME/$(date '+%Y-%m-%d-%H-%M-%S')"

echo "🚀 Hybrid Planner Benchmark"
echo "=========================="
echo "Configuration:"
echo "  • Planner: Hybrid (PDM@10Hz + Diffusion@2Hz)"
echo "  • Challenge: $CHALLENGE"
echo "  • Scenarios: test14-hard (272 scenarios)"
echo "  • Workers: 32 threads"
echo "  • GPU allocation: 0.25 per simulation"
echo ""

echo "🚀 Starting benchmark..."
echo "Experiment ID: $EXPERIMENT_ID"
START_TIME=$(date +%s)
START_DATE=$(date)

###################################
# Run Full Benchmark
###################################
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    scenario_builder=nuplan_challenge \
    scenario_filter=$SPLIT \
    experiment_uid=$EXPERIMENT_ID \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=32 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.25 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://chimera.config, pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"

# Save exit code
EXIT_CODE=$?

###################################
# Final Report
###################################
END_TIME=$(date +%s)
END_DATE=$(date)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "🏆 BENCHMARK COMPLETE!"
else
    echo "⚠️  BENCHMARK INTERRUPTED!"
fi
echo "====================="
echo "⏱️  Started: $START_DATE"
echo "⏱️  Ended: $END_DATE" 
echo "⏱️  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "📂 Results location:"
echo "   $NUPLAN_EXP_ROOT/exp/simulation/$CHALLENGE/$EXPERIMENT_ID"

# Cleanup will happen automatically due to trap
exit $EXIT_CODE