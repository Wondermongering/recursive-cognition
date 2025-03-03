"""
Basic Exploration Example

This example demonstrates the core functionality of the RecursiveCognitionSystem
by creating a simple system and exploring a single trajectory.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path to import the package
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from recursive_cognition import RecursiveCognitionSystem


def run_basic_example():
    """Run a basic exploration of a recursive cognition system."""
    print("Running basic exploration example...\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a system with default parameters
    system = RecursiveCognitionSystem(
        embedding_dim=20,
        nonlinearity_strength=0.3,
        memory_strength=0.0,
        perturbation_level=0.0
    )
    
    # Print system information
    system_info = system.get_system_info()
    print("System Information:")
    print(f"Embedding dimension: {system.embedding_dim}")
    print(f"Nonlinearity strength: {system.nonlinearity_strength}")
    print(f"Memory strength: {system.memory_strength}")
    print(f"Perturbation level: {system.perturbation_level}")
    print(f"Max eigenvalue magnitude: {system_info['max_eigenvalue_magnitude']:.4f}")
    print(f"Stability prediction: {system_info['stability_prediction']}")
    print()
    
    # Create an initial thought vector
    print("Creating initial thought vector...")
    initial_thought = np.random.randn(system.embedding_dim)
    initial_thought = initial_thought / np.linalg.norm(initial_thought)
    
    # Trace a trajectory
    print("Tracing trajectory...")
    steps = 20
    trajectory = system.trace_trajectory(initial_thought, steps)
    
    # Analyze the dynamics
    print("Analyzing dynamics...")
    dynamics = system.analyze_dynamics(trajectory)
    
    # Display results
    print("\nDynamics Analysis Results:")
    print(f"Trajectory length: {len(trajectory)} states")
    
    if dynamics['fixed_point']:
        print("Dynamics type: Fixed Point (stable convergence)")
    elif dynamics['has_cycle']:
        print(f"Dynamics type: Limit Cycle (length {dynamics['cycle_length']})")
    else:
        print("Dynamics type: Complex/Chaotic")
    
    print("\nSimilarity between consecutive states:")
    for i, sim in enumerate(dynamics['cosine_similarities']):
        print(f"  Step {i+1} → {i+2}: {sim:.6f}")
    
    # Visualize the trajectory
    print("\nVisualizing trajectory...")
    results = system.explore_single_trajectory(
        steps=steps, 
        random_init=False,
        init_state=initial_thought
    )
    
    print("\nExploration complete!")
    

if __name__ == "__main__":
    run_basic_example()
