#!/bin/bash
# sim_hybrid_planner_ultra_fast.sh
# Ultra-optimized hybrid planner for maximum speed with memory safety

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
# Ultra Performance Settings
###################################
# Aggressive RAY settings for speed
export RAY_memory_monitor_refresh_ms=0
export RAY_object_store_memory=3000000000    # 3GB object store (more than diffusion)
export RAY_worker_heap_memory_bytes=750000000 # 750MB per worker (more than diffusion)
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DISABLE_USAGE_STATS=1

# Python optimizations
export PYTHONOPTIMIZE=1                      # Remove asserts and debug info
export OMP_NUM_THREADS=4                     # Optimize NumPy/PyTorch threading

# System memory limits
ulimit -v 30000000  # Slightly higher limit (30GB) for better performance

###################################
# Benchmark Configuration  
###################################
SPLIT="test14-hard"
CHALLENGE="closed_loop_reactive_agents"
BRANCH_NAME="hybrid_planner_ultra"
PLANNER="hybrid_planner"
EXPERIMENT_ID="$PLANNER/$SPLIT/$BRANCH_NAME/$(date '+%Y-%m-%d-%H-%M-%S')"

###################################
# Performance Analysis
###################################
echo "🚀 ULTRA-FAST Hybrid Planner Benchmark"
echo "======================================"
echo "Configuration:"
echo "  • Planner: Hybrid (PDM@10Hz + Diffusion@2Hz)"
echo "  • Challenge: $CHALLENGE" 
echo "  • Scenarios: ~300 (full test14-hard)"
echo "  • Workers: 128 threads (ultra-parallel)"
echo "  • GPU allocation: 0.125 per sim (8x parallel)"
echo "  • Memory: 3GB object store + 750MB/worker"
echo ""
echo "Expected performance:"
echo "  • ~2-3x faster than standard settings"
echo "  • ~1.5-2.5 hours total runtime"
echo "  • High CPU/GPU utilization"
echo ""

# Confirm before running
read -p "🏃‍♂️ Launch ultra-fast benchmark? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Benchmark cancelled."
    exit 1
fi

echo "🏁 Starting ultra-fast benchmark..."
START_TIME=$(date +%s)

###################################
# Ultra-Fast Simulation Command
###################################
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    scenario_builder=nuplan_challenge \
    scenario_filter=$SPLIT \
    experiment_uid=$EXPERIMENT_ID \
    verbose=false \
    worker=ray_distributed \
    worker.threads_per_node=128 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.125 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://chimera.config, pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"

###################################
# Performance Report
###################################
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "🏆 ULTRA-FAST BENCHMARK COMPLETE!"
echo "================================"
echo "⏱️  Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "📊 Results: $NUPLAN_EXP_ROOT/exp/simulation/$CHALLENGE/$EXPERIMENT_ID"
echo ""
echo "🔍 Quick analysis:"
echo "   cd $NUPLAN_EXP_ROOT/exp/simulation/$CHALLENGE/$EXPERIMENT_ID"
echo "   ls -la"
echo ""
echo "📈 Performance files:"
echo "   • aggregated_metrics.parquet - Overall scores"
echo "   • runner_report.parquet - Per-scenario results"  
echo "   • Histogram plots in metric_*.png"
echo ""
echo "🧠 Hybrid Intelligence Analysis:"
echo "   • Check switching frequency (PDM vs Diffusion usage)"
echo "   • Compare scores: collision, comfort, progress, TTC"
echo "   • Analyze scenario-specific performance patterns"