#!/bin/bash
# check_memory.sh - Quick memory status check

echo "🔍 Memory Status Check"
echo "===================="

echo -e "\n📊 RAM Usage:"
free -h

echo -e "\n💾 Shared Memory (/dev/shm):"
df -h /dev/shm
ls -la /dev/shm/ 2>/dev/null | grep -E "(ray|plasma)" | wc -l | xargs echo "Ray/Plasma files:"

echo -e "\n🐍 Python/Ray Processes:"
ps aux | grep -E "(ray|python|diffusion)" | grep -v grep | wc -l | xargs echo "Active processes:"

echo -e "\n🎮 GPU Memory:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

echo -e "\n🧹 To clean up, run:"
echo "   ray stop --force"
echo "   pkill -f 'run_simulation.py'"
echo "   rm -rf /dev/shm/ray* /dev/shm/plasma_* /tmp/ray*"