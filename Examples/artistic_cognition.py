"""
Artistic Cognition Example

This example applies the recursive cognition framework to model artistic processes
and aesthetic evolution through recursive refinement.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA

# Add the parent directory to the path to import the package
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from recursive_cognition import RecursiveCognitionSystem


class ArtisticCognitionSystem(RecursiveCognitionSystem):
    """
    Extension of RecursiveCognitionSystem with aesthetic features.
    
    This class adds aesthetically-meaningful interpretations of states
    and transformations, modeling how artistic thoughts evolve.
    """
    
    def __init__(self, aesthetic_dimensions=None, nonlinearity_strength=0.5,
                memory_strength=0.3, inspiration_level=0.02, random_seed=None):
        """
        Initialize an artistic cognition system.
        
        Parameters
        ----------
        aesthetic_dimensions : dict, optional
            Dictionary mapping dimension indices to aesthetic qualities
        nonlinearity_strength : float, optional
            How strongly nonlinear transformations are applied (0-1)
        memory_strength : float, optional
            How much past states influence current state
        inspiration_level : float, optional
            Level of random perturbation (modeling external inspiration)
        random_seed : int, optional
            Seed for random number generation
        """
        # Default aesthetic dimensions if not provided
        if aesthetic_dimensions is None:
            aesthetic_dimensions = {
                0: "Complexity",
                1: "Harmony",
                2: "Contrast",
                3: "Rhythm",
                4: "Balance",
                5: "Tension",
                6: "Novelty",
                7: "Emotional intensity",
                8: "Cultural reference",
                9: "Technical skill"
            }
        
        # Calculate embedding dimension from aesthetic dimensions
        embedding_dim = max(20, max(aesthetic_dimensions.keys()) + 1)
        
        # Initialize base system
        super().__init__(
            embedding_dim=embedding_dim,
            nonlinearity_strength=nonlinearity_strength,
            memory_strength=memory_strength,
            perturbation_level=inspiration_level,
            random_seed=random_seed
        )
        
        self.aesthetic_dimensions = aesthetic_dimensions
        self.inspiration_level = inspiration_level
        
        # Add aesthetic bias to transformation matrix
        # This creates relationships between certain aesthetic dimensions
        self._add_aesthetic_bias()
    
    def _add_aesthetic_bias(self):
        """Add aesthetic biases to the transformation matrix."""
        # Example aesthetic relationships
        # - Complexity and technical skill are positively correlated
        # - Harmony and tension are negatively correlated
        # - Balance influences harmony positively
        
        dim_map = self.aesthetic_dimensions
        
        # Find the relevant dimension indices
        complexity_idx = next((k for k, v in dim_map.items() if "omplex" in v), None)
        harmony_idx = next((k for k, v in dim_map.items() if "armon" in v), None)
        tension_idx = next((k for k, v in dim_map.items() if "ension" in v), None)
        balance_idx = next((k for k, v in dim_map.items() if "alanc" in v), None)
        skill_idx = next((k for k, v in dim_map.items() if "kill" in v), None)
        
        # Apply biases if dimensions are found
        if complexity_idx is not None and skill_idx is not None:
            self.W[complexity_idx, skill_idx] += 0.2
            self.W[skill_idx, complexity_idx] += 0.2
        
        if harmony_idx is not None and tension_idx is not None:
            self.W[harmony_idx, tension_idx] -= 0.3
            self.W[tension_idx, harmony_idx] -= 0.3
        
        if balance_idx is not None and harmony_idx is not None:
            self.W[balance_idx, harmony_idx] += 0.25
    
    def interpret_state(self, state):
        """
        Interpret a state vector in terms of aesthetic qualities.
        
        Parameters
        ----------
        state : ndarray
            State vector to interpret
            
        Returns
        -------
        dict
            Dictionary mapping aesthetic qualities to values
        """
        # Normalize values to 0-1 range
        normalized = (state + 1) / 2
        
        # Create interpretation dictionary
        interpretation = {}
        for idx, quality in self.aesthetic_dimensions.items():
            if idx < len(state):
                interpretation[quality] = normalized[idx]
        
        return interpretation
    
    def visualize_aesthetic_state(self, state, ax=None, title=None):
        """
        Visualize an aesthetic state as a radar chart.
        
        Parameters
        ----------
        state : ndarray
            State vector to visualize
        ax : matplotlib.axes.Axes, optional
            Axes on which to draw the radar chart
        title : str, optional
            Title for the radar chart
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the radar chart
        """
        interpretation = self.interpret_state(state)
        
        # Get qualities and values (up to 10 for readability)
        qualities = list(interpretation.keys())[:10]
        values = [interpretation[q] for q in qualities]
        
        # Create new axes if not provided
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        N = len(qualities)
        
        # Angle of each axis
        angles = [n / N * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Values for each angle
        values += values[:1]  # Close the loop
        
        # Draw chart
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.4)
        
        # Set angle labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(qualities)
        
        # Set y limits
        ax.set_ylim(0, 1)
        
        # Set title if provided
        if title:
            ax.set_title(title, fontsize=15)
        
        return ax
    
    def create_artistic_trajectory(self, initial_concept=None, steps=10, visualize=True):
        """
        Model the evolution of an artistic concept through recursive thinking.
        
        Parameters
        ----------
        initial_concept : ndarray, optional
            Initial concept state (random if not provided)
        steps : int, optional
            Number of recursive steps to trace
        visualize : bool, optional
            Whether to visualize the trajectory
            
        Returns
        -------
        dict
            Dictionary containing trajectory, interpretations, and visualizations
        """
        # Generate initial concept if not provided
        if initial_concept is None:
            initial_concept = np.random.randn(self.embedding_dim)
            initial_concept = initial_concept / np.linalg.norm(initial_concept)
        
        # Trace trajectory
        trajectory = self.trace_trajectory(initial_concept, steps)
        
        # Generate interpretations for each state
        interpretations = [self.interpret_state(state) for state in trajectory]
        
        if visualize:
            # Create visualization grid
            num_vis = min(steps + 1, 5)  # Show at most 5 steps for clarity
            step_indices = np.linspace(0, steps, num_vis, dtype=int)
            
            fig = plt.figure(figsize=(15, 15))
            
            # Create radar charts for selected states
            for i, idx in enumerate(step_indices):
                ax = fig.add_subplot(num_vis, 1, i + 1, polar=True)
                self.visualize_aesthetic_state(
                    trajectory[idx], 
                    ax=ax, 
                    title=f"Iteration {idx}: Artistic Concept"
                )
            
            plt.tight_layout()
            plt.show()
            
            # Create PCA visualization of full trajectory
            pca = PCA(n_components=2)
            trajectory_2d = pca.fit_transform(trajectory)
            
            plt.figure(figsize=(10, 8))
            plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'o-', linewidth=2, markersize=8)
            plt.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], s=100, c='green', 
                       edgecolor='black', zorder=10, label='Initial Concept')
            plt.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], s=150, c='red', marker='*', 
                       edgecolor='black', zorder=10, label='Final Concept')
            
            # Add step numbers
            for i, (x, y) in enumerate(trajectory_2d):
                plt.text(x, y, str(i), fontsize=9, ha='right')
            
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('Artistic Concept Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return {
            'trajectory': trajectory,
            'interpretations': interpretations
        }
    
    def compare_artistic_styles(self, num_styles=3, steps=10):
        """
        Compare different artistic styles evolving through recursive thinking.
        
        Parameters
        ----------
        num_styles : int, optional
            Number of different styles to compare
        steps : int, optional
            Number of recursive steps for each style
            
        Returns
        -------
        dict
            Dictionary containing trajectories and visualizations
        """
        # Generate different initial concepts
        initial_concepts = [np.random.randn(self.embedding_dim) for _ in range(num_styles)]
        initial_concepts = [v / np.linalg.norm(v) for v in initial_concepts]
        
        # Trace trajectories
        trajectories = [self.trace_trajectory(concept, steps) for concept in initial_concepts]
        
        # Generate interpretations
        all_interpretations = []
        for trajectory in trajectories:
            interpretations = [self.interpret_state(state) for state in trajectory]
            all_interpretations.append(interpretations)
        
        # Visualize final states
        fig = plt.figure(figsize=(15, 5))
        
        for i in range(num_styles):
            ax = fig.add_subplot(1, num_styles, i + 1, polar=True)
            self.visualize_aesthetic_state(
                trajectories[i][-1], 
                ax=ax, 
                title=f"Style {i+1}: Final Concept"
            )
        
        plt.tight_layout()
        plt.show()
        
        # Track evolution of specific aesthetic qualities
        qualities_to_track = ["Complexity", "Harmony", "Novelty", "Emotional intensity"]
        qualities_to_track = [q for q in qualities_to_track 
                             if any(q in interp[0] for interp in all_interpretations)]
        
        if qualities_to_track:
            plt.figure(figsize=(15, 10))
            
            for q_idx, quality in enumerate(qualities_to_track):
                plt.subplot(2, 2, q_idx + 1)
                
                for style_idx in range(num_styles):
                    # Extract values for this quality across all steps
                    values = [interp.get(quality, 0) for interp in all_interpretations[style_idx]]
                    steps_range = range(len(values))
                    
                    plt.plot(steps_range, values, 'o-', linewidth=2, 
                             label=f"Style {style_idx+1}")
                
                plt.xlabel("Iteration")
                plt.ylabel("Value")
                plt.title(f"Evolution of {quality}")
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        return {
            'trajectories': trajectories,
            'interpretations': all_interpretations
        }
    
    def visualize_as_abstract_art(self, state, ax=None, title=None, cmap='viridis'):
        """
        Visualize a state as abstract art.
        
        Parameters
        ----------
        state : ndarray
            State vector to visualize
        ax : matplotlib.axes.Axes, optional
            Axes on which to draw
        title : str, optional
            Title for the artwork
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap to use
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the abstract art
        """
        # Create new axes if not provided
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
        
        # Normalize the state
        interpretation = self.interpret_state(state)
        
        # Extract aesthetic values that influence the visualization
        complexity = interpretation.get("Complexity", 0.5)
        harmony = interpretation.get("Harmony", 0.5)
        contrast = interpretation.get("Contrast", 0.5)
        rhythm = interpretation.get("Rhythm", 0.5)
        balance = interpretation.get("Balance", 0.5)
        tension = interpretation.get("Tension", 0.5)
        
        # Create a grid for the abstract art
        grid_size = int(5 + complexity * 15)  # More complexity = finer grid
        
        # Generate base grid values
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Create various patterns based on aesthetic values
        pattern1 = np.sin(X * 10 * rhythm) * np.cos(Y * 10 * rhythm)
        pattern2 = np.sin((X - 0.5)**2 * 10 * tension) * np.cos((Y - 0.5)**2 * 10 * tension)
        
        # Combine patterns based on harmony
        combined = harmony * pattern1 + (1 - harmony) * pattern2
        
        # Apply contrast
        combined = (combined - combined.min()) / (combined.max() - combined.min())
        combined = (combined - 0.5) * contrast + 0.5
        
        # Apply balance by making the pattern more/less symmetric
        if balance > 0.5:
            # Make more symmetric
            weight = (balance - 0.5) * 2  # 0 to 1
            symmetric = (combined + combined[::-1, :]) / 2
            combined = weight * symmetric + (1 - weight) * combined
        
        # Draw the abstract art
        im = ax.imshow(combined, cmap=cmap, extent=[0, 1, 0, 1])
        
        # Add geometric elements based on other aesthetic dimensions
        num_elements = int(interpretation.get("Complexity", 0.5) * 10)
        
        for i in range(num_elements):
            # Random position with bias toward balance
            if balance > 0.5:
                # More balanced = more centered
                center_bias = balance * 2 - 1
                x = (np.random.random() - 0.5) * (1 - center_bias) + 0.5
                y = (np.random.random() - 0.5) * (1 - center_bias) + 0.5
            else:
                x = np.random.random()
                y = np.random.random()
            
            # Size based on emotional intensity
            size = 0.05 + interpretation.get("Emotional intensity", 0.5) * 0.15
            
            # Shape based on various factors
            shape_factor = (i / num_elements) * 2 * np.pi
            
            # Color based on emotionality
            emotion = interpretation.get("Emotional intensity", 0.5)
            color_val = (np.sin(shape_factor) + 1) / 2
            color = plt.cm.hsv(color_val)
            
            # Draw element
            if i % 3 == 0:
                # Circle
                circle = patches.Circle((x, y), size, fill=True, alpha=0.7, color=color)
                ax.add_patch(circle)
            elif i % 3 == 1:
                # Rectangle
                rect = patches.Rectangle((x, y), size, size, fill=True, alpha=0.7, color=color)
                ax.add_patch(rect)
            else:
                # Triangle
                triangle = patches.RegularPolygon((x, y), 3, size, fill=True, alpha=0.7, color=color)
                ax.add_patch(triangle)
        
        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set title if provided
        if title:
            ax.set_title(title, fontsize=15)
        
        return ax
    
    def animate_artistic_evolution(self, steps=20, save_path=None):
        """
        Create an animation of artistic concept evolution.
        
        Parameters
        ----------
        steps : int, optional
            Number of recursive steps
        save_path : str, optional
            Path to save the animation (if None, animation is displayed)
            
        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object
        """
        # Generate initial concept
        initial_concept = np.random.randn(self.embedding_dim)
        initial_concept = initial_concept / np.linalg.norm(initial_concept)
        
        # Trace trajectory
        trajectory = self.trace_trajectory(initial_concept, steps)
        
        # Setup figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Initialize with empty plot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        title = ax.set_title("Iteration 0", fontsize=15)
        
        # Animation function
        def update(frame):
            ax.clear()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            
            self.visualize_as_abstract_art(trajectory[frame], ax=ax)
            title = ax.set_title(f"Iteration {frame}", fontsize=15)
            
            return [ax]
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=len(trajectory), interval=500, blit=True)
        
        # Save or display
        if save_path:
            ani.save(save_path, writer='pillow', fps=2)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        return ani


def run_artistic_example():
    """Run an example of artistic cognition."""
    print("Running artistic cognition example...\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create artistic cognition system
    system = ArtisticCognitionSystem(
        nonlinearity_strength=0.4,
        memory_strength=0.3,
        inspiration_level=0.02
    )
    
    print("Aesthetic dimensions:")
    for idx, quality in system.aesthetic_dimensions.items():
        print(f"  {idx}: {quality}")
    
    print("\nGenerating artistic trajectory...")
    result = system.create_artistic_trajectory(steps=15)
    
    print("\nComparing different artistic styles...")
    comparison = system.compare_artistic_styles(num_styles=3, steps=10)
    
    print("\nVisualizing as abstract art...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Initial concept
    system.visualize_as_abstract_art(result['trajectory'][0], ax=axs[0], 
                                 title="Initial Concept", cmap='viridis')
    
    # Middle concept
    mid_idx = len(result['trajectory']) // 2
    system.visualize_as_abstract_art(result['trajectory'][mid_idx], ax=axs[1], 
                                 title=f"Iteration {mid_idx}", cmap='plasma')
    
    # Final concept
    system.visualize_as_abstract_art(result['trajectory'][-1], ax=axs[2], 
                                 title="Final Concept", cmap='magma')
    
    plt.tight_layout()
    plt.show()
    
    # Optional: Uncomment to create animation
    # print("\nCreating animation of artistic evolution...")
    # system.animate_artistic_evolution(steps=10, save_path="artistic_evolution.gif")
    
    print("\nArtistic cognition example complete!")


if __name__ == "__main__":
    run_artistic_example()
