"""
Nonlinearity Effects Example

This example explores how different levels of nonlinearity affect the dynamics
of recursive thinking, from linear systems to highly nonlinear systems.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap

# Add the parent directory to the path to import the package
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from recursive_cognition import RecursiveCognitionSystem


def explore_nonlinearity_effects(nonlinearity_levels=None, steps=30, save_results=False):
    """
    Explore how different levels of nonlinearity affect system dynamics.
    
    Parameters
    ----------
    nonlinearity_levels : list of float, optional
        Nonlinearity levels to explore (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    steps : int, optional
        Number of recursive steps for each trajectory
    save_results : bool, optional
        Whether to save results and figures
    """
    if nonlinearity_levels is None:
        nonlinearity_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print(f"Exploring nonlinearity effects with levels: {nonlinearity_levels}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate a single random starting point
    embedding_dim = 20
    v_0 = np.random.randn(embedding_dim)
    v_0 = v_0 / np.linalg.norm(v_0)
    
    # Create systems with different nonlinearity levels
    systems = [RecursiveCognitionSystem(embedding_dim=embedding_dim, nonlinearity_strength=nl) 
               for nl in nonlinearity_levels]
    
    # Trace trajectories
    trajectories = [system.trace_trajectory(v_0, steps) for system in systems]
    
    # Analyze trajectories and collect results
    analysis_results = []
    for i, (system, traj, nl) in enumerate(zip(systems, trajectories, nonlinearity_levels)):
        analysis = system.analyze_dynamics(traj)
        system_info = system.get_system_info()
        
        dynamics_type = "Fixed Point" if analysis['fixed_point'] else \
                       f"Limit Cycle (len={analysis['cycle_length']})" if analysis['has_cycle'] else \
                       "Complex/Chaotic"
        
        # Average rate of change (average distance between consecutive states)
        avg_change_rate = 1 - np.mean(analysis['cosine_similarities'])
        
        analysis_results.append({
            'Nonlinearity': nl,
            'Dynamics Type': dynamics_type,
            'Final Similarity': analysis['final_similarity'],
            'Avg Change Rate': avg_change_rate,
            'Max Eigenvalue': abs(system_info['max_eigenvalue_magnitude']),
            'Stability': system_info['stability_prediction']
        })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(analysis_results)
    
    # Visualize trajectories with PCA
    all_points = np.vstack(trajectories)
    pca = PCA(n_components=2)
    all_projected = pca.fit_transform(all_points)
    
    # Split back into separate trajectories
    projected_trajectories = []
    idx = 0
    for traj in trajectories:
        traj_len = len(traj)
        projected_trajectories.append(all_projected[idx:idx+traj_len])
        idx += traj_len
    
    # Plot trajectories
    plt.figure(figsize=(12, 10))
    
    # Create a custom colormap for nonlinearity levels
    cmap = LinearSegmentedColormap.from_list('nonlinearity', 
                                           [(0, 'blue'), (0.5, 'purple'), (1, 'red')])
    colors = cmap(np.array(nonlinearity_levels))
    
    for i, (traj, nl, color) in enumerate(zip(projected_trajectories, nonlinearity_levels, colors)):
        plt.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2, 
                 label=f"NL = {nl}", alpha=0.8)
        plt.scatter(traj[0, 0], traj[0, 1], s=100, color=color, edgecolor='black', zorder=10)
        plt.scatter(traj[-1, 0], traj[-1, 1], s=150, color=color, marker='*', 
                   edgecolor='black', zorder=10)
    
    explained_var = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({explained_var[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({explained_var[1]:.1%} variance)")
    plt.title("Effect of Nonlinearity on Recursive Thinking Trajectories", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_results:
        plt.savefig("nonlinearity_trajectories.png", dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    # Plot stability metrics across nonlinearity levels
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_df['Nonlinearity'], results_df['Final Similarity'], 
             'o-', linewidth=2, markersize=8)
    plt.xlabel("Nonlinearity Strength")
    plt.ylabel("Final Similarity")
    plt.title("Stability vs. Nonlinearity")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['Nonlinearity'], results_df['Avg Change Rate'], 
             'o-', linewidth=2, markersize=8, color='orange')
    plt.xlabel("Nonlinearity Strength")
    plt.ylabel("Average Rate of Change")
    plt.title("Thought Evolution Rate vs. Nonlinearity")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_results:
        plt.savefig("nonlinearity_metrics.png", dpi=300, bbox_inches='tight')
        results_df.to_csv("nonlinearity_results.csv", index=False)
    
    plt.show()
    
    # Display results
    print("\nNonlinearity Analysis Results:")
    print(results_df)
    
    # Summarize findings
    print("\nKey Findings:")
    if results_df['Final Similarity'].corr(results_df['Nonlinearity']) < -0.5:
        print("- Increasing nonlinearity decreases stability (strong negative correlation)")
    elif results_df['Final Similarity'].corr(results_df['Nonlinearity']) < -0.2:
        print("- Increasing nonlinearity slightly decreases stability (weak negative correlation)")
    else:
        print("- Nonlinearity has limited effect on final stability")
        
    if results_df['Avg Change Rate'].corr(results_df['Nonlinearity']) > 0.5:
        print("- Increasing nonlinearity significantly increases the rate of thought evolution")
    elif results_df['Avg Change Rate'].corr(results_df['Nonlinearity']) > 0.2:
        print("- Increasing nonlinearity moderately increases the rate of thought evolution")
    else:
        print("- Nonlinearity has limited effect on the rate of thought evolution")
        
    # Check for transitions in dynamics type
    dynamics_types = results_df['Dynamics Type'].tolist()
    if len(set(dynamics_types)) > 1:
        print("- Nonlinearity causes transitions between different dynamic regimes:")
        for nl, dtype in zip(nonlinearity_levels, dynamics_types):
            print(f"  NL = {nl}: {dtype}")
    else:
        print(f"- All nonlinearity levels produce the same dynamics type: {dynamics_types[0]}")
    
    return results_df


if __name__ == "__main__":
    # Run with default nonlinearity levels
    results = explore_nonlinearity_effects(save_results=False)
    
    # Optional: Run with finer-grained nonlinearity levels
    # detailed_levels = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
    # detailed_results = explore_nonlinearity_effects(nonlinearity_levels=detailed_levels, 
    #                                              save_results=True)
