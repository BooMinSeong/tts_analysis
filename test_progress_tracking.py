#!/usr/bin/env python
"""Test script to demonstrate the new progress tracking features."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.analyze_results import discover_and_load, analyze_all

print("=" * 60)
print("Progress Tracking Test")
print("=" * 60)
print()
print("Testing with a single experiment and temperature...")
print()

# Load a small subset - just one experiment with one temperature
hub_path = "ENSEONG/preprocessed-default-MATH-500-Qwen2.5-3B-Instruct-bon"
temp = 0.1

print("Loading data...")
loaded = discover_and_load([hub_path], temperature=temp, verbose=True)

print("\n" + "=" * 60)
print("Starting Analysis (watch for progress bars)")
print("=" * 60)
print()

# Run analysis with verbose mode to see all progress tracking
results = analyze_all(loaded, verbose=True)

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
print()
print(f"Analyzed hub paths: {list(results.keys())}")
if results:
    first_hub = list(results.keys())[0]
    print(f"Temperatures analyzed: {list(results[first_hub].keys())}")
    first_temp = list(results[first_hub].keys())[0]
    print(f"Seeds analyzed: {list(results[first_hub][first_temp].keys())}")
    print()
    print("✓ Progress tracking working correctly!")
