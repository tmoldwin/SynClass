#!/bin/bash

# Script to analyze sweep results
# Usage: ./run_analysis.sh [sweep_directory]

if [ $# -eq 0 ]; then
    echo "Usage: ./run_analysis.sh [sweep_directory]"
    echo ""
    echo "Available sweep directories:"
    ls -d sweep_* 2>/dev/null | head -10
    echo ""
    echo "Example: ./run_analysis.sh sweep_20241210_143022"
    exit 1
fi

SWEEP_DIR="$1"

if [ ! -d "$SWEEP_DIR" ]; then
    echo "Error: Directory '$SWEEP_DIR' does not exist"
    echo ""
    echo "Available sweep directories:"
    ls -d sweep_* 2>/dev/null
    exit 1
fi

echo "Analyzing sweep directory: $SWEEP_DIR"
echo "This will generate comprehensive analysis and visualizations..."
echo ""

python analyze_sweep_results.py "$SWEEP_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Analysis complete!"
    echo "📊 Results saved to: $SWEEP_DIR/analysis/"
    echo "📈 Visualizations: $SWEEP_DIR/analysis/sweep_analysis.png"
    echo "📋 Summary: $SWEEP_DIR/analysis/summary.txt"
    echo ""
    echo "To view the best configurations:"
    echo "cat $SWEEP_DIR/analysis/summary.txt"
else
    echo ""
    echo "❌ Analysis failed. Check the error messages above."
    exit 1
fi
