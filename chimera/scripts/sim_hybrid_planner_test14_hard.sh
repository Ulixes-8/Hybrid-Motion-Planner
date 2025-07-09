#!/bin/bash
# sim_hybrid_planner_optimal.sh  
# Optimized runner based on performance test results

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
# Optimal Performance Settings (Based on Test Results)
###################################
# Memory management
export RAY_memory_monitor_refresh_ms=0
export RAY_object_store_memory=2500000000    # 2.5GB (slightly higher)
export RAY_worker_heap_memory_bytes=600000000 # 600MB per worker
export RAY_DISABLE_IMPORT_WARNING=1

# System limits  
ulimit -v 15000000  # 25GB limit

###################################
# Benchmark Configuration
###################################
SPLIT="test14-hard"
# CHALLENGE="closed_loop_nonreactive_agents"
CHALLENGE="closed_loop_reactive_agents"
BRANCH_NAME="hybrid_planner_optimal"
PLANNER="hybrid_planner"
EXPERIMENT_ID="$PLANNER/$SPLIT/$BRANCH_NAME/$(date '+%Y-%m-%d-%H-%M-%S')"

###################################
# Performance Estimates
###################################
ESTIMATED_SCENARIOS=300
SCENARIOS_PER_SEC=0.037  # Based on balanced test (slightly optimistic)
ESTIMATED_HOURS=$(echo "scale=1; $ESTIMATED_SCENARIOS / $SCENARIOS_PER_SEC / 3600" | bc -l)

echo "🚀 OPTIMAL Hybrid Planner Benchmark"
echo "==================================="
echo "Configuration (based on performance tests):"
echo "  • Planner: Hybrid (PDM@10Hz + Diffusion@2Hz)"
echo "  • Challenge: $CHALLENGE"
echo "  • Scenarios: ~$ESTIMATED_SCENARIOS (full test14-hard)"
echo "  • Workers: 64 threads (optimal from testing)"
echo "  • GPU allocation: 0.2 per simulation"
echo "  • Memory: 2.5GB object store + 600MB/worker"
echo ""
echo "📊 Performance expectations:"
echo "  • Rate: ~$SCENARIOS_PER_SEC scenarios/sec"
echo "  • Estimated runtime: ~$ESTIMATED_HOURS hours"
echo "  • Expected to complete around: $(date -d "+${ESTIMATED_HOURS} hours" '+%Y-%m-%d %H:%M')"
echo ""

# Confirm before running
read -p "🏁 Launch optimal benchmark? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Benchmark cancelled."
    exit 1
fi

echo "🚀 Starting optimal benchmark..."
echo "Experiment ID: $EXPERIMENT_ID"
START_TIME=$(date +%s)
START_DATE=$(date)

###################################
# Optimal Simulation Command
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
    number_of_gpus_allocated_per_simulation=0.2 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://chimera.config, pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"

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
echo "🏆 OPTIMAL BENCHMARK COMPLETE!"
echo "============================="
echo "⏱️  Started: $START_DATE"
echo "⏱️  Ended: $END_DATE" 
echo "⏱️  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "📂 Results location:"
echo "   $NUPLAN_EXP_ROOT/exp/simulation/$CHALLENGE/$EXPERIMENT_ID"
echo ""
echo "🔍 Key analysis commands:"
echo "   cd $NUPLAN_EXP_ROOT/exp/simulation/$CHALLENGE/$EXPERIMENT_ID"
echo "   ls -la                                    # List all result files"
echo "   head aggregated_metrics.parquet          # Overall performance summary"
echo ""
echo "🧠 Hybrid Intelligence Insights to Look For:"
echo "   • PDM vs Diffusion usage frequency"
echo "   • Scenario types where each planner excels"  
echo "   • Score distributions (collision, comfort, progress)"
echo "   • Switching decision accuracy"