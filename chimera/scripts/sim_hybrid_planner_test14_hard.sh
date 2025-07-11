#!/bin/bash
# sim_hybrid_planner_test14_hard_fixed.sh
# Fixed version with better memory management

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

###################################
# FIXED Memory Settings
###################################
# More conservative memory settings to avoid mmap failures
export RAY_memory_monitor_refresh_ms=0
export RAY_object_store_memory=1000000000    # Reduced to 1GB
export RAY_plasma_directory=/tmp              # Use /tmp instead of /dev/shm
export RAY_DISABLE_IMPORT_WARNING=1

# System limits - more conservative
ulimit -v unlimited  # Remove virtual memory limit
ulimit -n 65536      # Increase file descriptors

# Check shared memory before starting
echo "🔍 Checking system resources..."
echo "Shared memory (/dev/shm):"
df -h /dev/shm
echo ""
echo "Available RAM:"
free -h
echo ""

###################################
# Benchmark Configuration
###################################
SPLIT="test14-hard"
CHALLENGE="closed_loop_reactive_agents"
BRANCH_NAME="hybrid_planner_fixed"
PLANNER="hybrid_planner"
EXPERIMENT_ID="$PLANNER/$SPLIT/$BRANCH_NAME/$(date '+%Y-%m-%d-%H-%M-%S')"

echo "🚀 Hybrid Planner Benchmark (Memory-Safe Version)"
echo "================================================"
echo "Configuration:"
echo "  • Planner: Hybrid (PDM@10Hz + Diffusion@2Hz)"
echo "  • Challenge: $CHALLENGE"
echo "  • Scenarios: test14-hard"
echo "  • Workers: 32 threads (reduced for stability)"
echo "  • GPU allocation: 0.25 per simulation"
echo "  • Memory: 1GB object store (reduced)"
echo ""

# Check if we have enough shared memory
SHARED_MEM_SIZE=$(df /dev/shm | tail -1 | awk '{print $4}')
SHARED_MEM_SIZE_MB=$((SHARED_MEM_SIZE / 1024))

if [ $SHARED_MEM_SIZE_MB -lt 2048 ]; then
    echo "⚠️  WARNING: Shared memory is low ($SHARED_MEM_SIZE_MB MB available)"
    echo "    Recommended: sudo mount -o remount,size=8G /dev/shm"
    echo ""
fi

# Confirm before running
read -p "🏁 Start benchmark with fixed settings? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Benchmark cancelled."
    exit 1
fi

echo "🚀 Starting benchmark..."
echo "Experiment ID: $EXPERIMENT_ID"
START_TIME=$(date +%s)
START_DATE=$(date)

###################################
# Run with Conservative Settings First
###################################
# Start with fewer scenarios to test
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    scenario_builder=nuplan_challenge \
    scenario_filter=$SPLIT \
    scenario_filter.limit_total_scenarios=10 \
    experiment_uid=$EXPERIMENT_ID \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=16 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.25 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://chimera.config, pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"

# Check if it succeeded
if [ $? -eq 0 ]; then
    echo "✅ Initial test successful! Running full benchmark..."
    
    # Run full benchmark with same safe settings
    python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
        +simulation=$CHALLENGE \
        planner=$PLANNER \
        scenario_builder=nuplan_challenge \
        scenario_filter=$SPLIT \
        experiment_uid="${EXPERIMENT_ID}_full" \
        verbose=true \
        worker=ray_distributed \
        worker.threads_per_node=32 \
        distributed_mode='SINGLE_NODE' \
        number_of_gpus_allocated_per_simulation=0.25 \
        enable_simulation_progress_bar=true \
        hydra.searchpath="[pkg://chimera.config, pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
else
    echo "❌ Initial test failed. Please check the error messages above."
fi

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
echo "🏆 BENCHMARK COMPLETE!"
echo "====================="
echo "⏱️  Started: $START_DATE"
echo "⏱️  Ended: $END_DATE" 
echo "⏱️  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "📂 Results location:"
echo "   $NUPLAN_EXP_ROOT/exp/simulation/$CHALLENGE/$EXPERIMENT_ID"