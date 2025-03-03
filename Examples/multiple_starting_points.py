"""
Multiple Starting Points Example

This example explores how different initial thoughts evolve within the same system,
showing whether they converge to similar endpoints or diverge into different patterns.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# Add the parent directory to the path to import the package
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from recursive_cognition import RecursiveCognitionSystem


def explore_multiple_starting_points(num_points=5, steps=30, nonlinearity=0.3, save_fig=False):
    """
    Explore how different starting thoughts evolve in the same cognitive system.
    
    Parameters
    ----------
    num_points : int
        Number of different starting points to explore
    steps : int
        Number of recursive steps to trace for each starting point
    nonlinearity : float
        Nonlinearity strength for the system
    save_fig : bool
        Whether to save the figure to a file
    """
    print(f"Exploring {num_points} different starting points with {steps} steps each...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create the cognitive system
    system = RecursiveCognitionSystem(
        embedding_dim=20,
        nonlinearity_strength=nonlinearity,
        memory_strength=0.0,
        perturbation_level=0.0
    )
    
    # Generate multiple random starting points
    starting_points = [np.random.randn(system.embedding_dim) for _ in range(num_points)]
    starting_points = [v / np.linalg.norm(v) for v in starting_points]
    
    # Trace trajectories for each starting point
    trajectories = [system.trace_trajectory(sp, steps) for sp in starting_points]
    
    # Combine all points for PCA
    all_points = np.vstack(trajectories)
    
    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    all_projected = pca.fit_transform(all_points)
    
    # Split back into separate trajectories
    projected_trajectories = []
    idx = 0
    for traj in trajectories:
        traj_len = len(traj)
        projected_trajectories.append(all_projected[idx:idx+traj_len])
        idx += traj_len
    
    # Analyze dynamics for each trajectory
    results = []
    for i, traj in enumerate(trajectories):
        analysis = system.analyze_dynamics(traj)
        
        dynamics_type = "Fixed Point" if analysis['fixed_point'] else \
                       f"Limit Cycle (len={analysis['cycle_length']})" if analysis['has_cycle'] else \
                       "Complex/Chaotic"
        
        # Calculate distance to other final points
        final_point = traj[-1]
        distances = []
        for j, other_traj in enumerate(trajectories):
            if i != j:
                other_final = other_traj[-1]
                sim = np.dot(final_point, other_final) / (np.linalg.norm(final_point) * np.linalg.norm(other_final))
                distances.append(1 - sim)  # Convert similarity to distance
        
        avg_distance = np.mean(distances) if distances else 0
        
        results.append({
            'Trajectory': i+1,
            'Dynamics Type': dynamics_type,
            'Final Similarity': analysis['final_similarity'],
            'Avg Distance to Others': avg_distance
        })
    
    # Create a DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Plot the trajectories
    plt.figure(figsize=(12, 10))
    
    # Plot each trajectory with different color
    colors = plt.cm.tab10(np.linspace(0, 1, num_points))
    for i, (traj, color) in enumerate(zip(projected_trajectories, colors)):
        plt.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2, 
                 label=f"Trajectory {i+1}", alpha=0.8)
        plt.scatter(traj[0, 0], traj[0, 1], s=100, color=color, edgecolor='black', zorder=10)
        plt.text(traj[0, 0], traj[0, 1], f"{i+1}", fontsize=12, ha='right')
        plt.scatter(traj[-1, 0], traj[-1, 1], s=150, color=color, marker='*', edgecolor='black', zorder=10)
    
    explained_var = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({explained_var[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({explained_var[1]:.1%} variance)")
    plt.title(f"Multiple Starting Points in Thought Space\nNonlinearity = {nonlinearity}", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_fig:
        plt.savefig(f"multiple_starting_points_nl{nonlinearity}.png", dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    # Display results
    print("\nTrajectory Analysis Results:")
    print(results_df)
    
    # Check for convergence among final points
    final_points = [traj[-1] for traj in trajectories]
    similarities = []
    for i in range(len(final_points)):
        for j in range(i+1, len(final_points)):
            sim = np.dot(final_points[i], final_points[j]) / (
                np.linalg.norm(final_points[i]) * np.linalg.norm(final_points[j]))
            similarities.append(sim)
    
    avg_similarity = np.mean(similarities)
    print(f"\nAverage similarity between final points: {avg_similarity:.4f}")
    
    if avg_similarity > 0.9:
        print("FINDING: Trajectories strongly converge to similar final states")
    elif avg_similarity > 0.7:
        print("FINDING: Trajectories moderately converge")
    elif avg_similarity > 0.5:
        print("FINDING: Trajectories show weak convergence")
    else:
        print("FINDING: Trajectories diverge to different final states")
    
    return results_df


if __name__ == "__main__":
    # Explore with default parameters
    results = explore_multiple_starting_points(num_points=5, steps=30, nonlinearity=0.3)
    
    # Optional: Explore with different nonlinearity levels
    # for nl in [0.0, 0.3, 0.7, 1.0]:
    #     explore_multiple_starting_points(num_points=5, steps=30, nonlinearity=nl, save_fig=True)
