"""
Visualization utilities for the Recursive Cognition Explorer.

This module provides functions for visualizing trajectories, stability curves,
phase spaces, and other aspects of recursive cognition systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import matplotlib.patches as patches


def visualize_trajectory(trajectory, title=None, highlight_points=None, cmap='viridis', 
                        show_start_end=True, pca_components=2, figsize=(10, 8),
                        explained_variance=True, grid=True, alpha=0.8):
    """
    Visualize a single trajectory using PCA for dimensionality reduction.
    
    Parameters
    ----------
    trajectory : ndarray
        Array of states forming the trajectory
    title : str, optional
        Plot title
    highlight_points : list of int, optional
        Indices of specific points to highlight
    cmap : str or matplotlib colormap, optional
        Colormap to use for trajectory progression
    show_start_end : bool, optional
        Whether to highlight start and end points
    pca_components : int, optional
        Number of PCA components to use (2 or 3)
    figsize : tuple, optional
        Figure size
    explained_variance : bool, optional
        Whether to include explained variance in axis labels
    grid : bool, optional
        Whether to show grid
    alpha : float, optional
        Transparency level for plot elements
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    ax : matplotlib.axes.Axes
        The axes containing the plot
    pca : sklearn.decomposition.PCA
        The fitted PCA object
    """
    if pca_components not in (2, 3):
        raise ValueError("pca_components must be either 2 or 3")
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=pca_components)
    trajectory_reduced = pca.fit_transform(trajectory)
    
    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    
    if pca_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        plot_func = ax.scatter
    else:
        ax = fig.add_subplot(111)
        plot_func = ax.scatter
    
    # Plot the trajectory points
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(trajectory)))
    
    if pca_components == 3:
        ax.plot(trajectory_reduced[:, 0], trajectory_reduced[:, 1], trajectory_reduced[:, 2],
               '-', color='gray', alpha=alpha*0.5, linewidth=1)
        scatter = ax.scatter(trajectory_reduced[:, 0], trajectory_reduced[:, 1], trajectory_reduced[:, 2],
                           c=np.arange(len(trajectory)), cmap=cmap, s=50, alpha=alpha)
    else:
        ax.plot(trajectory_reduced[:, 0], trajectory_reduced[:, 1],
               '-', color='gray', alpha=alpha*0.5, linewidth=1)
        scatter = ax.scatter(trajectory_reduced[:, 0], trajectory_reduced[:, 1],
                           c=np.arange(len(trajectory)), cmap=cmap, s=50, alpha=alpha)
    
    # Add a colorbar to show progression
    cbar = plt.colorbar(scatter, ax=ax, label='Recursion Depth')
    
    # Highlight start and end points if requested
    if show_start_end:
        if pca_components == 3:
            ax.scatter(trajectory_reduced[0, 0], trajectory_reduced[0, 1], trajectory_reduced[0, 2],
                      color='green', s=100, label='Start', edgecolor='black', zorder=10)
            ax.scatter(trajectory_reduced[-1, 0], trajectory_reduced[-1, 1], trajectory_reduced[-1, 2],
                      color='red', s=100, marker='*', label='End', edgecolor='black', zorder=10)
        else:
            ax.scatter(trajectory_reduced[0, 0], trajectory_reduced[0, 1],
                      color='green', s=100, label='Start', edgecolor='black', zorder=10)
            ax.scatter(trajectory_reduced[-1, 0], trajectory_reduced[-1, 1],
                      color='red', s=100, marker='*', label='End', edgecolor='black', zorder=10)
    
    # Highlight specific points if requested
    if highlight_points is not None:
        for idx in highlight_points:
            if idx < len(trajectory):
                if pca_components == 3:
                    ax.scatter(trajectory_reduced[idx, 0], trajectory_reduced[idx, 1], trajectory_reduced[idx, 2],
                              color='magenta', s=100, edgecolor='black', zorder=10)
                    ax.text(trajectory_reduced[idx, 0], trajectory_reduced[idx, 1], trajectory_reduced[idx, 2],
                           f'Step {idx}', fontsize=10)
                else:
                    ax.scatter(trajectory_reduced[idx, 0], trajectory_reduced[idx, 1],
                              color='magenta', s=100, edgecolor='black', zorder=10)
                    ax.text(trajectory_reduced[idx, 0], trajectory_reduced[idx, 1],
                           f'Step {idx}', fontsize=10)
    
    # Set labels with explained variance if requested
    if explained_variance:
        explained_var = pca.explained_variance_ratio_
        ax.set_xlabel(f'Principal Component 1 ({explained_var[0]:.1%} variance)')
        ax.set_ylabel(f'Principal Component 2 ({explained_var[1]:.1%} variance)')
        if pca_components == 3:
            ax.set_zlabel(f'Principal Component 3 ({explained_var[2]:.1%} variance)')
    else:
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        if pca_components == 3:
            ax.set_zlabel('Principal Component 3')
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add grid if requested
    if grid:
        ax.grid(alpha=0.3)
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    
    return fig, ax, pca


def visualize_multiple_trajectories(trajectories, labels=None, colors=None, title=None,
                                  pca_components=2, figsize=(12, 10), alpha=0.8,
                                  show_start_end=True, common_pca=True):
    """
    Visualize multiple trajectories together using PCA.
    
    Parameters
    ----------
    trajectories : list of ndarray
        List of trajectory arrays
    labels : list of str, optional
        Labels for each trajectory
    colors : list or colormap, optional
        Colors for each trajectory
    title : str, optional
        Plot title
    pca_components : int, optional
        Number of PCA components to use (2 or 3)
    figsize : tuple, optional
        Figure size
    alpha : float, optional
        Transparency level for plot elements
    show_start_end : bool, optional
        Whether to highlight start and end points
    common_pca : bool, optional
        Whether to use a common PCA for all trajectories
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    ax : matplotlib.axes.Axes
        The axes containing the plot
    pca : sklearn.decomposition.PCA
        The fitted PCA object
    """
    if pca_components not in (2, 3):
        raise ValueError("pca_components must be either 2 or 3")
    
    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    
    if pca_components == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    # Generate colors if not provided
    if colors is None:
        if isinstance(colors, str):
            cmap = plt.cm.get_cmap(colors)
        else:
            cmap = plt.cm.tab10
        colors = cmap(np.linspace(0, 1, len(trajectories)))
    
    # Generate labels if not provided
    if labels is None:
        labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]
    
    # Apply PCA for dimensionality reduction
    if common_pca:
        # Stack all trajectories for a common PCA
        all_points = np.vstack(trajectories)
        pca = PCA(n_components=pca_components)
        all_reduced = pca.fit_transform(all_points)
        
        # Split back into separate trajectories
        reduced_trajectories = []
        idx = 0
        for traj in trajectories:
            traj_len = len(traj)
            reduced_trajectories.append(all_reduced[idx:idx+traj_len])
            idx += traj_len
    else:
        # Apply separate PCA for each trajectory
        pca = PCA(n_components=pca_components)
        reduced_trajectories = [pca.fit_transform(traj) for traj in trajectories]
    
    # Plot each trajectory
    for i, (traj, label, color) in enumerate(zip(reduced_trajectories, labels, colors)):
        if pca_components == 3:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-', color=color, linewidth=2,
                   label=label, alpha=alpha)
        else:
            ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2,
                   label=label, alpha=alpha)
        
        # Highlight start and end points if requested
        if show_start_end:
            if pca_components == 3:
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], s=100, color=color,
                          edgecolor='black', zorder=10)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], s=150, color=color,
                          marker='*', edgecolor='black', zorder=10)
            else:
                ax.scatter(traj[0, 0], traj[0, 1], s=100, color=color,
                          edgecolor='black', zorder=10)
                ax.scatter(traj[-1, 0], traj[-1, 1], s=150, color=color,
                          marker='*', edgecolor='black', zorder=10)
    
    # Set labels and title
    if common_pca:
        explained_var = pca.explained_variance_ratio_
        ax.set_xlabel(f'Principal Component 1 ({explained_var[0]:.1%} variance)')
        ax.set_ylabel(f'Principal Component 2 ({explained_var[1]:.1%} variance)')
        if pca_components == 3:
            ax.set_zlabel(f'Principal Component 3 ({explained_var[2]:.1%} variance)')
    else:
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        if pca_components == 3:
            ax.set_zlabel('Principal Component 3')
    
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add grid and legend
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    return fig, ax, pca


def plot_stability_curve(similarity_data, labels=None, colors=None, title="Stability Analysis",
                       figsize=(10, 6), grid=True, fill_std=True, ylim=(0, 1)):
    """
    Plot stability curves showing similarity between consecutive states.
    
    Parameters
    ----------
    similarity_data : list or dict
        List of arrays containing similarity values, or dictionary mapping labels to arrays
    labels : list of str, optional
        Labels for each curve
    colors : list or colormap, optional
        Colors for each curve
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    grid : bool, optional
        Whether to show grid
    fill_std : bool, optional
        Whether to fill standard deviation region (if similarity_data contains multiple trials)
    ylim : tuple, optional
        Y-axis limits
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert dictionary to list if needed
    if isinstance(similarity_data, dict):
        labels = list(similarity_data.keys())
        similarity_data = list(similarity_data.values())
    
    # Generate colors if not provided
    if colors is None:
        if isinstance(colors, str):
            cmap = plt.cm.get_cmap(colors)
        else:
            cmap = plt.cm.tab10
        colors = cmap(np.linspace(0, 1, len(similarity_data)))
    
    # Generate labels if not provided
    if labels is None:
        labels = [f'Series {i+1}' for i in range(len(similarity_data))]
    
    # Plot each stability curve
    for i, (data, label, color) in enumerate(zip(similarity_data, labels, colors)):
        # Check if data contains multiple trials
        if isinstance(data[0], (list, np.ndarray)):
            # Multiple trials - compute mean and std
            data_array = np.array(data)
            mean_data = np.mean(data_array, axis=0)
            std_data = np.std(data_array, axis=0)
            
            # Plot mean curve
            steps = np.arange(1, len(mean_data) + 1)
            ax.plot(steps, mean_data, '-', color=color, linewidth=2.5, label=label)
            
            # Fill standard deviation region if requested
            if fill_std:
                ax.fill_between(steps,
                               np.clip(mean_data - std_data, 0, 1),
                               np.clip(mean_data + std_data, 0, 1),
                               color=color, alpha=0.2)
        else:
            # Single trial
            steps = np.arange(1, len(data) + 1)
            ax.plot(steps, data, '-', color=color, linewidth=2, label=label)
    
    # Set labels and title
    ax.set_xlabel('Recursive Step', fontsize=12)
    ax.set_ylabel('Stability (Cosine Similarity)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Set y-axis limits
    ax.set_ylim(ylim)
    
    # Add grid if requested
    if grid:
        ax.grid(alpha=0.3)
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    
    return fig, ax


def create_trajectory_animation(trajectories, labels=None, colors=None, title=None,
                              interval=200, pca_components=2, figsize=(12, 10),
                              common_pca=True, save_path=None):
    """
    Create an animation showing the evolution of multiple trajectories.
    
    Parameters
    ----------
    trajectories : list of ndarray
        List of trajectory arrays
    labels : list of str, optional
        Labels for each trajectory
    colors : list or colormap, optional
        Colors for each trajectory
    title : str, optional
        Animation title
    interval : int, optional
        Interval between frames in milliseconds
    pca_components : int, optional
        Number of PCA components to use (2 or 3)
    figsize : tuple, optional
        Figure size
    common_pca : bool, optional
        Whether to use a common PCA for all trajectories
    save_path : str, optional
        Path to save the animation
        
    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object
    """
    if pca_components not in (2, 3):
        raise ValueError("pca_components must be either 2 or 3")
    
    # Generate colors if not provided
    if colors is None:
        if isinstance(colors, str):
            cmap = plt.cm.get_cmap(colors)
        else:
            cmap = plt.cm.tab10
        colors = [cmap(i) for i in np.linspace(0, 1, len(trajectories))]
    
    # Generate labels if not provided
    if labels is None:
        labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]
    
    # Apply PCA for dimensionality reduction
    if common_pca:
        # Stack all trajectories for a common PCA
        all_points = np.vstack(trajectories)
        pca = PCA(n_components=pca_components)
        all_reduced = pca.fit_transform(all_points)
        
        # Split back into separate trajectories
        reduced_trajectories = []
        idx = 0
        for traj in trajectories:
            traj_len = len(traj)
            reduced_trajectories.append(all_reduced[idx:idx+traj_len])
            idx += traj_len
    else:
        # Apply separate PCA for each trajectory
        pca = PCA(n_components=pca_components)
        reduced_trajectories = [pca.fit_transform(traj) for traj in trajectories]
    
    # Set up the figure
    fig = plt.figure(figsize=figsize)
    
    if pca_components == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    # Set up lines and points for animation
    lines = []
    points = []
    
    for color in colors:
        if pca_components == 3:
            line, = ax.plot([], [], [], '-', color=color, alpha=0.7, linewidth=2)
            point, = ax.plot([], [], [], 'o', color=color, markersize=10)
        else:
            line, = ax.plot([], [], '-', color=color, alpha=0.7, linewidth=2)
            point, = ax.plot([], [], 'o', color=color, markersize=10)
        
        lines.append(line)
        points.append(point)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=label)
                       for color, label in zip(colors, labels)]
    ax.legend(handles=legend_elements)
    
    # Set axis limits
    if common_pca:
        x_min, x_max = all_reduced[:, 0].min(), all_reduced[:, 0].max()
        y_min, y_max = all_reduced[:, 1].min(), all_reduced[:, 1].max()
        
        # Add some padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        if pca_components == 3:
            z_min, z_max = all_reduced[:, 2].min(), all_reduced[:, 2].max()
            z_padding = (z_max - z_min) * 0.1
            ax.set_zlim(z_min - z_padding, z_max + z_padding)
    
    # Set labels
    if common_pca:
        explained_var = pca.explained_variance_ratio_
        ax.set_xlabel(f'Principal Component 1 ({explained_var[0]:.1%} variance)')
        ax.set_ylabel(f'Principal Component 2 ({explained_var[1]:.1%} variance)')
        if pca_components == 3:
            ax.set_zlabel(f'Principal Component 3 ({explained_var[2]:.1%} variance)')
    else:
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        if pca_components == 3:
            ax.set_zlabel('Principal Component 3')
    
    # Add title
    anim_title = ax.set_title('Step 0')
    
    # Animation initialization function
    def init():
        for line, point in zip(lines, points):
            if pca_components == 3:
                line.set_data([], [])
                line.set_3d_properties([])
                point.set_data([], [])
                point.set_3d_properties([])
            else:
                line.set_data([], [])
                point.set_data([], [])
        
        anim_title.set_text('Step 0')
        return lines + points + [anim_title]
    
    # Animation update function
    def update(frame):
        for i, (line, point, traj) in enumerate(zip(lines, points, reduced_trajectories)):
            if frame < len(traj):
                # Update line to show trajectory up to current frame
                if pca_components == 3:
                    line.set_data(traj[:frame+1, 0], traj[:frame+1, 1])
                    line.set_3d_properties(traj[:frame+1, 2])
                    # Update point to show current position
                    point.set_data([traj[frame, 0]], [traj[frame, 1]])
                    point.set_3d_properties([traj[frame, 2]])
                else:
                    line.set_data(traj[:frame+1, 0], traj[:frame+1, 1])
                    # Update point to show current position
                    point.set_data([traj[frame, 0]], [traj[frame, 1]])
        
        step_title = title + f' - Step {frame}' if title else f'Step {frame}'
        anim_title.set_text(step_title)
        return lines + points + [anim_title]
    
    # Get maximum trajectory length
    max_steps = max(len(traj) for traj in reduced_trajectories)
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=max_steps, init_func=init, blit=True, interval=interval)
    
    # Add grid
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save animation if path provided
    if save_path:
        ani.save(save_path, writer='pillow', fps=1000//interval)
    
    return ani


def plot_parameter_sensitivity(parameter_values, metrics, parameter_name="Parameter",
                             metric_name="Metric", title=None, figsize=(10, 6),
                             threshold=None, color='blue', grid=True):
    """
    Plot the sensitivity of a metric to changes in a parameter.
    
    Parameters
    ----------
    parameter_values : array-like
        Values of the parameter
    metrics : array-like
        Corresponding metric values
    parameter_name : str, optional
        Name of the parameter
    metric_name : str, optional
        Name of the metric
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    threshold : float, optional
        Threshold value to highlight
    color : str, optional
        Line color
    grid : bool, optional
        Whether to show grid
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot parameter vs. metric
    ax.plot(parameter_values, metrics, 'o-', color=color, linewidth=2, markersize=8)
    
    # Add threshold line if provided
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Threshold: {threshold}')
    
    # Set labels and title
    ax.set_xlabel(parameter_name, fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    
    if title is None:
        title = f'{metric_name} vs. {parameter_name}'
    ax.set_title(title, fontsize=14)
    
    # Add grid if requested
    if grid:
        ax.grid(alpha=0.3)
    
    # Add legend if threshold provided
    if threshold is not None:
        ax.legend()
    
    plt.tight_layout()
    
    return fig, ax


def plot_eigenvalue_spectrum(eigenvalues, title="Eigenvalue Spectrum", figsize=(10, 6),
                           color='blue', unit_circle=True, grid=True):
    """
    Plot the eigenvalue spectrum of a transformation matrix.
    
    Parameters
    ----------
    eigenvalues : array-like
        Complex eigenvalues
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    color : str, optional
        Color for eigenvalue points
    unit_circle : bool, optional
        Whether to show the unit circle
    grid : bool, optional
        Whether to show grid
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract real and imaginary parts
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    # Plot eigenvalues in complex plane
    ax.scatter(real_parts, imag_parts, color=color, s=100, alpha=0.7, edgecolor='black')
    
    # Add unit circle if requested
    if unit_circle:
        circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.7,
                           label='Unit Circle')
        ax.add_patch(circle)
    
    # Set labels and title
    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Make axes equal to preserve circle shape
    ax.set_aspect('equal')
    
    # Make sure origin is included
    margin = max(1.1, max(abs(np.max(real_parts)), abs(np.min(real_parts)), 
                         abs(np.max(imag_parts)), abs(np.min(imag_parts)))) * 1.1
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    
    # Add grid if requested
    if grid:
        ax.grid(alpha=0.3)
    
    # Add zero lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add legend if unit circle shown
    if unit_circle:
        ax.legend()
    
    plt.tight_layout()
    
    return fig, ax


def plot_radar(values, categories, title=None, figsize=(8, 8), color='blue',
             fill=True, alpha=0.5, show_values=True, max_value=1.0):
    """
    Create a radar chart (spider plot) for visualizing multivariate data.
    
    Parameters
    ----------
    values : array-like
        Values for each category
    categories : list of str
        Category names
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    color : str, optional
        Color for the radar plot
    fill : bool, optional
        Whether to fill the radar plot
    alpha : float, optional
        Transparency level
    show_values : bool, optional
        Whether to show values on the plot
    max_value : float, optional
        Maximum value for scaling
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    # Number of categories
    N = len(categories)
    
    # Create angle values for each category
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Make the plot circular by repeating the first value at the end
    values = np.array(values)
    values = np.append(values, values[0])
    angles.append(angles[0])
    categories = categories + [categories[0]]
    
    # Create figure and polar axes
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2, color=color, label='_nolegend_')
    
    # Fill area if requested
    if fill:
        ax.fill(angles, values, color=color, alpha=alpha)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])
    
    # Set y limits
    ax.set_ylim(0, max_value)
    
    # Add values to the plot if requested
    if show_values:
        for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            ax.text(angle, value + max_value*0.05, f"{value:.2f}", 
                   ha='center', va='center', fontsize=9)
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14, y=1.1)
    
    plt.tight_layout()
    
    return fig, ax


def plot_similarity_matrix(trajectory, title="Similarity Matrix", figsize=(10, 8),
                         cmap='viridis', colorbar=True, grid=False, highlight_diagonal=True):
    """
    Plot the cosine similarity matrix between all states in a trajectory.
    
    Parameters
    ----------
    trajectory : ndarray
        Array of states forming the trajectory
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    cmap : str or matplotlib colormap, optional
        Colormap for the similarity matrix
    colorbar : bool, optional
        Whether to show colorbar
    grid : bool, optional
        Whether to show grid
    highlight_diagonal : bool, optional
        Whether to highlight the diagonal
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    # Calculate similarity matrix
    from recursive_cognition.analysis import cosine_similarity_matrix
    
    similarity_matrix = cosine_similarity_matrix(trajectory)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot similarity matrix
    im = ax.imshow(similarity_matrix, cmap=cmap, vmin=0, vmax=1)
    
    # Add colorbar if requested
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity')
    
    # Set labels and title
    ax.set_xlabel('State Index', fontsize=12)
    ax.set_ylabel('State Index', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add grid if requested
    if grid:
        ax.grid(alpha=0.3)
    
    # Highlight diagonal if requested
    if highlight_diagonal:
        for i in range(len(similarity_matrix)):
            ax.plot(i, i, 'o', color='red', markersize=3, alpha=0.7)
    
    plt.tight_layout()
    
    return fig, ax


def plot_eigenmode(eigenvector, pca=None, title="Eigenmode Visualization", figsize=(10, 8),
                  colormap='coolwarm', colorbar=True, grid=True, normalize=True):
    """
    Visualize an eigenvector (eigenmode) of a transformation matrix.
    
    Parameters
    ----------
    eigenvector : ndarray
        Eigenvector to visualize
    pca : sklearn.decomposition.PCA, optional
        PCA object for transforming the eigenvector into a lower-dimensional space
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    colormap : str or matplotlib colormap, optional
        Colormap for the eigenvector components
    colorbar : bool, optional
        Whether to show colorbar
    grid : bool, optional
        Whether to show grid
    normalize : bool, optional
        Whether to normalize the eigenvector
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    # Normalize eigenvector if requested
    if normalize:
        eigenvector = eigenvector / np.linalg.norm(eigenvector)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # If PCA is provided, visualize in 2D space
    if pca is not None:
        # Make eigenvector a row vector
        eigenvector_reshaped = eigenvector.reshape(1, -1)
        
        # Transform using PCA
        transformed = pca.transform(eigenvector_reshaped)[0]
        
        # Create grid and meshgrid
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(x, y)
        
        # Plot vector field
        ax.quiver(0, 0, transformed[0], transformed[1], 
                 angles='xy', scale_units='xy', scale=1, color='red',
                 label='Eigenmode Direction')
        
        # Set limits and labels
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
    else:
        # Visualize eigenvector components
        x = np.arange(len(eigenvector))
        ax.bar(x, eigenvector.real, color=plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(eigenvector))))
        
        # Set labels
        ax.set_xlabel('Component Index', fontsize=12)
        ax.set_ylabel('Component Value', fontsize=12)
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Add grid if requested
    if grid:
        ax.grid(alpha=0.3)
    
    # Add legend if PCA is provided
    if pca is not None:
        ax.legend()
    
    plt.tight_layout()
        plt.tight_layout()
    
    return fig, ax


# If executed as a script, run example visualizations
if __name__ == "__main__":
    import numpy as np
    from sklearn.decomposition import PCA

    # Example 1: Single Trajectory Visualization
    steps = 50
    dim = 10
    trajectory = np.cumsum(np.random.randn(steps, dim), axis=0)  # Simulated random walk

    fig, ax, pca = visualize_trajectory(
        trajectory,
        title="Example: Random Walk in High Dimensions",
        show_start_end=True,
        pca_components=3
    )
    plt.show()

    # Example 2: Multiple Trajectories
    num_trajectories = 3
    trajectories = [np.cumsum(np.random.randn(steps, dim), axis=0) for _ in range(num_trajectories)]
    labels = [f"Trajectory {i+1}" for i in range(num_trajectories)]

    fig, ax, pca = visualize_multiple_trajectories(
        trajectories, labels=labels, title="Comparison of Multiple Random Walks", pca_components=2
    )
    plt.show()

    # Example 3: Eigenvalue Spectrum
    matrix_size = 10
    random_matrix = np.random.randn(matrix_size, matrix_size)
    eigenvalues = np.linalg.eigvals(random_matrix)

    fig, ax = plot_eigenvalue_spectrum(
        eigenvalues, title="Eigenvalue Spectrum of a Random Matrix"
    )
    plt.show()

    # Example 4: Stability Curve Simulation
    similarity_data = [np.exp(-0.1 * np.arange(steps)) + 0.05 * np.random.randn(steps) for _ in range(3)]
    fig, ax = plot_stability_curve(
        similarity_data,
        labels=["Simulation A", "Simulation B", "Simulation C"],
        title="Stability Curves Over Recursive Steps"
    )
    plt.show()

    # Example 5: Radar Plot
    categories = ["Memory", "Adaptability", "Complexity", "Stability", "Recurrence"]
    values = np.random.rand(len(categories))
    
    fig, ax = plot_radar(
        values, categories, title="Radar Plot of Cognitive System Properties"
    )
    plt.show()
    
    print("Visualization examples completed successfully.")
