"""
Memory Influence Example

This example explores how memory effects (influence of past states beyond the immediate last one)
affect the coherence and trajectory of recursive thinking.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Add the parent directory to the path to import the package
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from recursive_cognition import RecursiveCognitionSystem


def explore_memory_influence(memory_levels=None, steps=30, nonlinearity=0.3, use_3d=False):
    """
    Explore how different memory strengths affect recursive thinking.
    
    Parameters
    ----------
    memory_levels : list of float, optional
        Memory strength levels to explore (default: [0.0, 0.2, 0.4, 0.6, 0.8])
    steps : int, optional
        Number of recursive steps for each trajectory
    nonlinearity : float, optional
        Nonlinearity strength to use for all systems
    use_3d : bool, optional
        Whether to use 3D visualization (uses 3 PCA components instead of 2)
    """
    if memory_levels is None:
        memory_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    print(f"Exploring memory influence with memory levels: {memory_levels}")
    print(f"Fixed nonlinearity: {nonlinearity}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate a single random starting point
    embedding_dim = 20
    v_0 = np.random.randn(embedding_dim)
    v_0 = v_0 / np.linalg.norm(v_0)
    
    # Create systems with different memory strengths
    systems = [RecursiveCognitionSystem(
        embedding_dim=embedding_dim, 
        nonlinearity_strength=nonlinearity,
        memory_strength=mem
    ) for mem in memory_levels]
    
    # Trace trajectories
    trajectories = [system.trace_trajectory(v_0, steps) for system in systems]
    
    # Analyze dynamics and collect results
    results = []
    for i, (system, traj, mem) in enumerate(zip(systems, trajectories, memory_levels)):
        analysis = system.analyze_dynamics(traj)
        
        dynamics_type = "Fixed Point" if analysis['fixed_point'] else \
                       f"Limit Cycle (len={analysis['cycle_length']})" if analysis['has_cycle'] else \
                       "Complex/Chaotic"
        
        # Calculate measures of coherence/smoothness
        cosine_similarities = analysis['cosine_similarities']
        avg_similarity = np.mean(cosine_similarities)
        min_similarity = np.min(cosine_similarities)
        similarity_std = np.std(cosine_similarities)
        
        results.append({
            'Memory Strength': mem,
            'Dynamics Type': dynamics_type,
            'Final Similarity': analysis['final_similarity'],
            'Average Similarity': avg_similarity,
            'Min Similarity': min_similarity,  # Lower values indicate sharper turns
            'Similarity Std': similarity_std   # Higher values indicate less smooth trajectories
        })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Visualize trajectories with PCA
    pca_components = 3 if use_3d else 2
    all_points = np.vstack(trajectories)
    pca = PCA(n_components=pca_components)
    all_projected = pca.fit_transform(all_points)
    
    # Split back into separate trajectories
    projected_trajectories = []
    idx = 0
    for traj in trajectories:
        traj_len = len(traj)
        projected_trajectories.append(all_projected[idx:idx+traj_len])
        idx += traj_len
    
    # Plot trajectories
    if use_3d:
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each trajectory
        colors = plt.cm.cool(np.linspace(0, 1, len(memory_levels)))
        for i, (traj, mem, color) in enumerate(zip(projected_trajectories, memory_levels, colors)):
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-', color=color, linewidth=2, 
                   label=f"Memory = {mem}", alpha=0.8)
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], s=100, color=color, 
                      edgecolor='black', zorder=10)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], s=150, color=color, 
                      marker='*', edgecolor='black', zorder=10)
        
        explained_var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({explained_var[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({explained_var[1]:.1%} variance)")
        ax.set_zlabel(f"PC3 ({explained_var[2]:.1%} variance)")
        ax.set_title("Effect of Memory on Recursive Thinking Trajectories", fontsize=14)
        ax.legend()
        
    else:
        plt.figure(figsize=(12, 10))
        
        # Plot each trajectory
        colors = plt.cm.cool(np.linspace(0, 1, len(memory_levels)))
        for i, (traj, mem, color) in enumerate(zip(projected_trajectories, memory_levels, colors)):
            plt.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2, 
                     label=f"Memory = {mem}", alpha=0.8)
            plt.scatter(traj[0, 0], traj[0, 1], s=100, color=color, edgecolor='black', zorder=10)
            plt.scatter(traj[-1, 0], traj[-1, 1], s=150, color=color, marker='*', 
                       edgecolor='black', zorder=10)
        
        explained_var = pca.explained_variance_ratio_
        plt.xlabel(f"PC1 ({explained_var[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({explained_var[1]:.1%} variance)")
        plt.title("Effect of Memory on Recursive Thinking Trajectories", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize memory effects on trajectory smoothness
    plt.figure(figsize=(12, 8))
    
    # Plot cosine similarities over time for each memory level
    plt.subplot(2, 1, 1)
    for i, (analysis, mem, color) in enumerate(zip([system.analyze_dynamics(traj) for system, traj in 
                                                 zip(systems, trajectories)], 
                                                memory_levels, colors)):
        similarities = analysis['cosine_similarities']
        steps_range = np.arange(1, len(similarities) + 1)
        plt.plot(steps_range, similarities, '-', color=color, linewidth=2, 
                 label=f"Memory = {mem}")
    
    plt.xlabel("Recursive Step")
    plt.ylabel("Cosine Similarity")
    plt.title("Trajectory Smoothness: Step-by-Step Similarity")
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot relationship between memory strength and smoothness metrics
    plt.subplot(2, 1, 2)
    plt.plot(results_df['Memory Strength'], results_df['Average Similarity'], 'o-', 
             label="Average Similarity", linewidth=2)
    plt.plot(results_df['Memory Strength'], results_df['Min Similarity'], 's-', 
             label="Minimum Similarity", linewidth=2)
    plt.plot(results_df['Memory Strength'], results_df['Similarity Std'], '^-', 
             label="Similarity Std Dev", linewidth=2)
    
    plt.xlabel("Memory Strength")
    plt.ylabel("Smoothness Metric")
    plt.title("Memory Strength vs. Trajectory Smoothness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Display results
    print("\nMemory Influence Analysis Results:")
    print(results_df)
    
    # Analyze and report trends
    print("\nKey Findings:")
    
    # Check effect on trajectory smoothness
    corr_avg = results_df['Memory Strength'].corr(results_df['Average Similarity'])
    if corr_avg > 0.7:
        print("- Increasing memory strength significantly improves trajectory smoothness")
    elif corr_avg > 0.3:
        print("- Increasing memory strength moderately improves trajectory smoothness")
    elif corr_avg > -0.3:
        print("- Memory strength has limited effect on trajectory smoothness")
    else:
        print("- Increasing memory strength decreases trajectory smoothness")
    
    # Check effect on stability
    corr_final = results_df['Memory Strength'].corr(results_df['Final Similarity'])
    if corr_final > 0.7:
        print("- Increasing memory strength significantly improves stability")
    elif corr_final > 0.3:
        print("- Increasing memory strength moderately improves stability")
    elif corr_final > -0.3:
        print("- Memory strength has limited effect on stability")
    else:
        print("- Increasing memory strength decreases stability")
    
    # Check for dynamics type transitions
    dynamics_types = results_df['Dynamics Type'].tolist()
    if len(set(dynamics_types)) > 1:
        print("- Memory strength causes transitions between different dynamic regimes:")
        for mem, dtype in zip(memory_levels, dynamics_types):
            print(f"  Memory = {mem}: {dtype}")
    else:
        print(f"- All memory levels produce the same dynamics type: {dynamics_types[0]}")
    
    return results_df


def visualize_memory_comparison(memory_pairs=None, steps=50, nonlinearity=0.3, perturbation=0.02):
    """
    Compare trajectories with and without memory under perturbation.
    
    This demonstrates how memory provides resilience against external perturbations.
    
    Parameters
    ----------
    memory_pairs : list of tuple, optional
        Pairs of memory strengths to compare (default: [(0.0, 0.5), (0.0, 0.8)])
    steps : int, optional
        Number of recursive steps for each trajectory
    nonlinearity : float, optional
        Nonlinearity strength to use for all systems
    perturbation : float, optional
        Level of perturbation/noise to apply
    """
    if memory_pairs is None:
        memory_pairs = [(0.0, 0.5), (0.0, 0.8)]
    
    print(f"Comparing memory effects under perturbation (level: {perturbation})...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # For each pair of memory strengths
    for mem1, mem2 in memory_pairs:
        # Create two systems with different memory strengths
        system1 = RecursiveCognitionSystem(
            embedding_dim=20,
            nonlinearity_strength=nonlinearity,
            memory_strength=mem1,
            perturbation_level=perturbation
        )
        
        system2 = RecursiveCognitionSystem(
            embedding_dim=20,
            nonlinearity_strength=nonlinearity,
            memory_strength=mem2,
            perturbation_level=perturbation
        )
        
        # Use same initial state and same transformation matrix for fair comparison
        system2.W = system1.W.copy()
        
        # Generate initial state
        v_0 = np.random.randn(20)
        v_0 = v_0 / np.linalg.norm(v_0)
        
        # Generate trajectories
        traj1 = system1.trace_trajectory(v_0, steps)
        traj2 = system2.trace_trajectory(v_0, steps)
        
        # Visualize with PCA
        all_points = np.vstack([traj1, traj2])
        pca = PCA(n_components=2)
        all_projected = pca.fit_transform(all_points)
        
        # Split back into separate trajectories
        traj1_2d = all_projected[:len(traj1)]
        traj2_2d = all_projected[len(traj1):]
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        # Plot trajectories
        plt.plot(traj1_2d[:, 0], traj1_2d[:, 1], 'b-o', markersize=5, linewidth=1.5, alpha=0.7,
                label=f"No/Low Memory (M={mem1})")
        plt.plot(traj2_2d[:, 0], traj2_2d[:, 1], 'r-o', markersize=5, linewidth=1.5, alpha=0.7,
                label=f"With Memory (M={mem2})")
        
        # Highlight start and end points
        plt.scatter(traj1_2d[0, 0], traj1_2d[0, 1], s=100, c='blue', marker='o', 
                   edgecolor='black', zorder=10, label='Start')
        plt.scatter(traj2_2d[0, 0], traj2_2d[0, 1], s=100, c='red', marker='o', 
                   edgecolor='black', zorder=10)
        
        plt.scatter(traj1_2d[-1, 0], traj1_2d[-1, 1], s=150, c='blue', marker='*', 
                   edgecolor='black', zorder=10, label='End')
        plt.scatter(traj2_2d[-1, 0], traj2_2d[-1, 1], s=150, c='red', marker='*', 
                   edgecolor='black', zorder=10)
        
        # Add a light connecting line between corresponding points to show divergence
        for i in range(min(len(traj1_2d), len(traj2_2d))):
            plt.plot([traj1_2d[i, 0], traj2_2d[i, 0]], [traj1_2d[i, 1], traj2_2d[i, 1]], 
                     'k-', alpha=0.1)
        
        # Calculate average point-to-point distance to quantify divergence
        distances = np.sqrt(np.sum((traj1_2d - traj2_2d) ** 2, axis=1))
        avg_distance = np.mean(distances)
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'Memory Effect on Trajectory Under Perturbation\n'
                 f'Memory Comparison: {mem1} vs {mem2} (Avg. Distance: {avg_distance:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Calculate and display stability metrics
        sim1 = np.mean([np.dot(traj1[i], traj1[i+1]) / 
                       (np.linalg.norm(traj1[i]) * np.linalg.norm(traj1[i+1])) 
                       for i in range(len(traj1)-1)])
        
        sim2 = np.mean([np.dot(traj2[i], traj2[i+1]) / 
                       (np.linalg.norm(traj2[i]) * np.linalg.norm(traj2[i+1])) 
                       for i in range(len(traj2)-1)])
        
        # Calculate path smoothness (less variation = smoother)
        smoothness1 = np.std([np.dot(traj1[i], traj1[i+1]) / 
                             (np.linalg.norm(traj1[i]) * np.linalg.norm(traj1[i+1])) 
                             for i in range(len(traj1)-1)])
        
        smoothness2 = np.std([np.dot(traj2[i], traj2[i+1]) / 
                             (np.linalg.norm(traj2[i]) * np.linalg.norm(traj2[i+1])) 
                             for i in range(len(traj2)-1)])
        
        print(f"\nComparison Results (Memory {mem1} vs {mem2}):")
        print(f"Average similarity - No/low memory: {sim1:.4f}, With memory: {sim2:.4f}")
        print(f"Path smoothness (lower is better) - No/low memory: {smoothness1:.4f}, "
              f"With memory: {smoothness2:.4f}")
        print(f"Average geometric distance between trajectories: {avg_distance:.4f}")
        
        if sim2 > sim1:
            improvement = (sim2 - sim1) / sim1 * 100
            print(f"Memory improves stability by {improvement:.1f}%")
        else:
            reduction = (sim1 - sim2) / sim1 * 100
            print(f"Memory reduces stability by {reduction:.1f}%")
        
        if smoothness2 < smoothness1:
            improvement = (smoothness1 - smoothness2) / smoothness1 * 100
            print(f"Memory improves path smoothness by {improvement:.1f}%")
        else:
            reduction = (smoothness2 - smoothness1) / smoothness1 * 100
            print(f"Memory reduces path smoothness by {reduction:.1f}%")


if __name__ == "__main__":
    # Basic memory influence exploration
    results = explore_memory_influence(memory_levels=[0.0, 0.2, 0.4, 0.6, 0.8])
    
    # Optional: Compare trajectories with and without memory under perturbation
    visualize_memory_comparison(memory_pairs=[(0.0, 0.5)], perturbation=0.05)
