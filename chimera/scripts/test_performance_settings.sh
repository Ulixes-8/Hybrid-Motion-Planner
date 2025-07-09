#!/bin/bash
# test_performance_settings.sh
# Quick performance test with different settings to find optimal configuration

cd ~/chimera
source chimera/scripts/setup_env.sh

echo "🧪 Testing Hybrid Planner Performance Settings"
echo "============================================="

# Function to run a quick test
run_performance_test() {
    local test_name="$1"
    local threads="$2"
    local gpu_alloc="$3"
    local scenarios="$4"
    
    echo ""
    echo "📊 Test: $test_name"
    echo "   Threads: $threads | GPU: $gpu_alloc | Scenarios: $scenarios"
    
    start_time=$(date +%s)
    
    python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
        +simulation=closed_loop_reactive_agents \
        planner=hybrid_planner \
        scenario_builder=nuplan_challenge \
        scenario_filter=test14-hard \
        scenario_filter.limit_total_scenarios=$scenarios \
        experiment_uid=hybrid_planner/perf_test/${test_name}_$(date '+%H-%M-%S') \
        verbose=false \
        worker=ray_distributed \
        worker.threads_per_node=$threads \
        distributed_mode='SINGLE_NODE' \
        number_of_gpus_allocated_per_simulation=$gpu_alloc \
        enable_simulation_progress_bar=false \
        hydra.searchpath="[pkg://chimera.config, pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
        > /dev/null 2>&1
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    scenarios_per_second=$(echo "scale=2; $scenarios / $duration" | bc -l)
    
    echo "   ⏱️  Duration: ${duration}s | Rate: ${scenarios_per_second} scenarios/sec"
}

# Set up memory limits
export RAY_memory_monitor_refresh_ms=0
export RAY_object_store_memory=2000000000
export RAY_worker_heap_memory_bytes=500000000
ulimit -v 25000000

echo "Testing different configurations with 5 scenarios each..."

# Test 1: Conservative (like your current working setup)
run_performance_test "conservative" 32 0.25 5

# Test 2: Balanced (moderate speedup)
run_performance_test "balanced" 64 0.2 5

# Test 3: Aggressive (maximum speed)
run_performance_test "aggressive" 128 0.125 5

echo ""
echo "🏆 Performance Test Complete!"
echo "================================"
echo ""
echo "💡 Recommendations:"
echo "   • Conservative: Most stable, good for overnight runs"
echo "   • Balanced: Good speed/stability tradeoff" 
echo "   • Aggressive: Maximum speed, monitor for crashes"
echo ""
echo "🚀 To run full benchmark with chosen setting:"
echo "   ./sim_hybrid_planner_test14_hard.sh     (conservative)"
echo "   ./sim_hybrid_planner_ultra_fast.sh      (aggressive)"