#!/bin/bash
# Quick test script for baseline evaluation

echo "Starting quick test on 100 COCO images..."
echo "Expected time: ~10 minutes"
echo ""

cd /Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/finalcode/baselines

# Run quick test
python run_quick_baselines.py \
    --quick-test \
    --output quick_test_results.json \
    --device cuda

echo ""
echo "Quick test completed!"
echo "Results saved to: quick_test_results.json"
echo ""
echo "To view results:"
echo "  cat quick_test_results.json"
echo ""
echo "To run full COCO 5K test (2-4 hours):"
echo "  python run_quick_baselines.py --output baseline_results.json"
