"""
Perturbation Analysis Example

This example explores how external perturbations (noise/stimuli) affect recursive thinking,
testing resilience of different thinking patterns to disturbance.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation

# Add the parent directory to the path to import the package
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from recursive_cognition import RecursiveCognitionSystem


def explore_perturbation_effects(perturbation_levels=None, steps=30, trials=5,
                                nonlinearity=0.3, memory=0.0, create_animation=False):
    """
    Explore how different levels of perturbation affect system dynamics.
    
    Parameters
    ----------
    perturbation_levels : list of float, optional
        Perturbation levels to explore (default: [0.0, 0.01, 0.05, 0.1, 0.2])
    steps : int, optional
        Number of recursive steps for each trajectory
    trials : int, optional
        Number of trials to run for each perturbation level
    nonlinearity : float, optional
        Nonlinearity strength to use for all systems
    memory : float, optional
        Memory strength to use for all systems
    create_animation : bool, optional
        Whether to create an animation of the perturbation effects
    """
    if perturbation_levels is None:
        perturbation_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    
    print(f"Exploring perturbation effects with levels: {perturbation_levels}")
    print(f"Fixed parameters - Nonlinearity: {nonlinearity}, Memory: {memory}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate a single random starting point
    embedding_dim = 20
    v_0 = np.random.randn(embedding_dim)
    v_0 = v_0 / np.linalg.norm(v_0)
    
    # Prepare storage for results
    stability_data = {level: [] for level in perturbation_levels}
    final_states = {level: [] for level in perturbation_levels}
    
    # Create a fixed transformation matrix for consistency
    W = np.random.randn(embedding_dim, embedding_dim) * 0.1
    
    # Run trials for each perturbation level
    for level in perturbation_levels:
        system = RecursiveCognitionSystem(
            embedding_dim=embedding_dim,
            nonlinearity_strength=nonlinearity,
            memory_strength=memory,
            perturbation_level=level
        )
        
        # Use the same transformation matrix for consistency
        system.W = W.copy()
        
        for trial in range(trials):
            # Reset memory for each trial
            system.reset_memory()
            
            # Generate trajectory
            trajectory = system.trace_trajectory(v_0, steps)
            
            # Analyze dynamics
            analysis = system.analyze_dynamics(trajectory)
            
            # Store similarity data for stability analysis
            stability_data[level].append(analysis['cosine_similarities'])
            
            # Store final states for convergence analysis
            final_states[level].append(trajectory[-1])
    
    # Analyze and plot stability across perturbation levels
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(perturbation_levels)))
    
    # Plot average stability for each perturbation level
    for (level, sims), color in zip(stability_data.items(), colors):
        # Average over trials
        avg_sims = np.mean(sims, axis=0)
        steps_range = np.arange(1, len(avg_sims) + 1)
        
        plt.plot(steps_range, avg_sims, '-', color=color, linewidth=2.5,
                 label=f"Noise {level}", alpha=0.9)
        
        # Add shaded region for variability across trials
        if trials > 1:
            std_sims = np.std(sims, axis=0)
            plt.fill_between(steps_range,
                             np.clip(avg_sims - std_sims, 0, 1),
                             np.clip(avg_sims + std_sims, 0, 1),
                             color=color, alpha=0.2)
    
    plt.xlabel("Recursive Depth", fontsize=12)
    plt.ylabel("Stability (Cosine Similarity)", fontsize=12)
    plt.title("Effect of Perturbation on Recursive Thinking Stability", fontsize=14)
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate and display stability metrics
    results = []
    
    for level in perturbation_levels:
        # Average stability across trials
        avg_stability = np.mean([np.mean(sim_data) for sim_data in stability_data[level]])
        stability_variance = np.mean([np.var(sim_data) for sim_data in stability_data[level]])
        
        # Final similarity (average of last step similarities across trials)
        final_sims = [sim_data[-1] for sim_data in stability_data[level]]
        avg_final_sim = np.mean(final_sims)
        
        # Convergence analysis - how similar are the final states across trials?
        if trials > 1:
            final_state_similarities = []
            for i in range(trials):
                for j in range(i+1, trials):
                    sim = np.dot(final_states[level][i], final_states[level][j]) / (
                        np.linalg.norm(final_states[level][i]) * np.linalg.norm(final_states[level][j]))
                    final_state_similarities.append(sim)
            
            convergence = np.mean(final_state_similarities)
        else:
            convergence = np.nan
        
        results.append({
            'Perturbation': level,
            'Avg Stability': avg_stability,
            'Stability Variance': stability_variance,
            'Final Similarity': avg_final_sim,
            'Convergence': convergence
        })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Plot metrics vs perturbation level
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_df['Perturbation'], results_df['Avg Stability'], 'o-', 
             linewidth=2, markersize=8)
    plt.xlabel("Perturbation Level")
    plt.ylabel("Average Stability")
    plt.title("Stability vs. Perturbation")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['Perturbation'], results_df['Stability Variance'], 'o-', 
             linewidth=2, markersize=8, color='orange')
    plt.xlabel("Perturbation Level")
    plt.ylabel("Stability Variance")
    plt.title("Variability vs. Perturbation")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Display results
    print("\nPerturbation Analysis Results:")
    print(results_df)
    
    # If trials > 1, plot convergence across trials
    if trials > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['Perturbation'], results_df['Convergence'], 'o-',
                 linewidth=2, markersize=8, color='green')
        plt.xlabel("Perturbation Level")
        plt.ylabel("Convergence (Similarity Across Trials)")
        plt.title("Effect of Perturbation on Outcome Predictability")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Create animation if requested
    if create_animation and trials > 0:
        create_perturbation_animation(perturbation_levels, steps, embedding_dim, 
                                     nonlinearity, memory, W)
    
    # Analyze and report trends
    print("\nKey Findings:")
    
    # Stability trend
    corr_stability = results_df['Perturbation'].corr(results_df['Avg Stability'])
    if corr_stability < -0.7:
        print("- Increasing perturbation significantly decreases stability")
    elif corr_stability < -0.3:
        print("- Increasing perturbation moderately decreases stability")
    else:
        print("- Perturbation has limited effect on stability")
    
    # Variability trend
    corr_variance = results_df['Perturbation'].corr(results_df['Stability Variance'])
    if corr_variance > 0.7:
        print("- Increasing perturbation significantly increases variability/unpredictability")
    elif corr_variance > 0.3:
        print("- Increasing perturbation moderately increases variability/unpredictability")
    else:
        print("- Perturbation has limited effect on variability")
    
    # Convergence trend (if applicable)
    if trials > 1:
        corr_convergence = results_df['Perturbation'].corr(results_df['Convergence'])
        if np.isnan(corr_convergence):
            print("- Could not calculate convergence trend")
        elif corr_convergence < -0.7:
            print("- Increasing perturbation significantly reduces outcome predictability")
        elif corr_convergence < -0.3:
            print("- Increasing perturbation moderately reduces outcome predictability")
        else:
            print("- Perturbation has limited effect on outcome predictability")
    
    return results_df


def create_perturbation_animation(perturbation_levels, steps, embedding_dim, 
                                 nonlinearity, memory, W=None):
    """
    Create an animation showing how perturbation affects trajectory evolution.
    
    Parameters
    ----------
    perturbation_levels : list of float
        Perturbation levels to animate
    steps : int
        Number of recursive steps to simulate
    embedding_dim : int
        Dimensionality of the embedding
    nonlinearity : float
        Nonlinearity strength
    memory : float
        Memory strength
    W : ndarray, optional
        Transformation matrix (if None, a new one will be created)
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create transformation matrix if not provided
    if W is None:
        W = np.random.randn(embedding_dim, embedding_dim) * 0.1
    
    # Generate initial state
    v_0 = np.random.randn(embedding_dim)
    v_0 = v_0 / np.linalg.norm(v_0)
    
    # Generate complete trajectories for each perturbation level
    all_trajectories = []
    for level in perturbation_levels:
        system = RecursiveCognitionSystem(
            embedding_dim=embedding_dim,
            nonlinearity_strength=nonlinearity,
            memory_strength=memory,
            perturbation_level=level
        )
        system.W = W.copy()
        trajectory = system.trace_trajectory(v_0, steps)
        all_trajectories.append(trajectory)
    
    # PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    all_points = np.vstack(all_trajectories)
    pca = PCA(n_components=2)
    all_projected = pca.fit_transform(all_points)
    
    # Split back into separate trajectories
    projected_trajectories = []
    idx = 0
    for traj in all_trajectories:
        traj_len = len(traj)
        projected_trajectories.append(all_projected[idx:idx+traj_len])
        idx += traj_len
    
    # Setup colors
    colors = plt.cm.plasma(np.linspace(0, 1, len(perturbation_levels)))
    
    # Setup lines and points for animation
    lines = []
    points = []
    
    for color in colors:
        line, = ax.plot([], [], '-', color=color, alpha=0.7, linewidth=2)
        point, = ax.plot([], [], 'o', color=color, markersize=10)
        lines.append(line)
        points.append(point)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], color=color, lw=2, 
                                 label=f'Perturbation = {level}')
                       for color, level in zip(colors, perturbation_levels)]
    ax.legend(handles=legend_elements)
    
    # Set axis limits
    x_min, x_max = all_projected[:, 0].min(), all_projected[:, 0].max()
    y_min, y_max = all_projected[:, 1].min(), all_projected[:, 1].max()
    
    # Add some padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Labels and title
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    title = ax.set_title('Step 0')
    
    # Animation initialization function
    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        title.set_text('Step 0')
        return lines + points + [title]
    
    # Animation update function
    def update(frame):
        for i, (line, point, traj) in enumerate(zip(lines, points, projected_trajectories)):
            if frame < len(traj):
                # Update line to show trajectory up to current frame
                line.set_data(traj[:frame+1, 0], traj[:frame+1, 1])
                # Update point to show current position
                point.set_data([traj[frame, 0]], [traj[frame, 1]])
        
        title.set_text(f'Step {frame}')
        return lines + points + [title]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=steps+1, init_func=init, blit=True, interval=200)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save animation or show it
    try:
        ani.save('perturbation_animation.gif', writer='pillow', fps=5)
        print("Animation saved as 'perturbation_animation.gif'")
    except Exception as e:
        print(f"Could not save animation: {e}")
        plt.show()


if __name__ == "__main__":
    # Run with default parameters
    results = explore_perturbation_effects(
        perturbation_levels=[0.0, 0.01, 0.05, 0.1, 0.2],
        steps=30,
        trials=5,
        nonlinearity=0.3,
        memory=0.0,
        create_animation=False  # Set to True to create animation
    )
