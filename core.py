"""
Core implementation of the Recursive Cognition System.

This module contains the main RecursiveCognitionSystem class that implements
the recursive self-reference model, along with related utilities and helper functions.
"""

import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class RecursiveCognitionSystem:
    """
    A system that models recursive self-reference processes.
    
    This class implements a mathematical model of a system that processes its own state
    and feeds it back into itself, creating a form of self-reference. The system can be
    parameterized to explore different dynamics including fixed points, limit cycles,
    and chaotic attractors.
    
    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the thought vector/embedding space
    nonlinearity_strength : float, optional (default=0.3)
        How strongly nonlinear transformations are applied (0-1)
    memory_strength : float, optional (default=0.0)
        How much past states influence current state beyond immediate last state
    perturbation_level : float, optional (default=0.0) 
        Level of random noise added at each step (simulating external stimuli)
    random_seed : int, optional (default=None)
        Seed for random number generation (for reproducibility)
    """
    
    def __init__(self, embedding_dim, nonlinearity_strength=0.3, memory_strength=0.0, 
                 perturbation_level=0.0, random_seed=None):
        """Initialize a recursive cognition system."""
        self.embedding_dim = embedding_dim
        self.nonlinearity_strength = nonlinearity_strength
        self.memory_strength = memory_strength
        self.perturbation_level = perturbation_level
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Create fixed transformation matrices
        self.W = np.random.randn(embedding_dim, embedding_dim) * 0.1
        
        # Analyze the eigenvalues to predict system behavior
        self.eigenvalues, self.eigenvectors = eig(self.W)
        
        # Storage for memory effect
        self.memory = None
        
    def reset_memory(self):
        """Reset the memory buffer."""
        self.memory = None
        
    def process(self, v):
        """
        Transform a thought vector with memory, nonlinearity and perturbation.
        
        Parameters
        ----------
        v : ndarray
            Input thought vector/state
            
        Returns
        -------
        ndarray
            Transformed thought vector/state
        """
        # Apply the base linear transformation
        transformed = np.dot(self.W, v)
        
        # Apply controlled nonlinearity
        if self.nonlinearity_strength > 0:
            nonlinear = np.tanh(transformed)
            transformed = (1 - self.nonlinearity_strength) * transformed + self.nonlinearity_strength * nonlinear
        
        # Apply memory effect if enabled
        if self.memory_strength > 0 and self.memory is not None:
            transformed = (1 - self.memory_strength) * transformed + self.memory_strength * self.memory
        
        # Update memory with current state
        self.memory = transformed.copy()
        
        # Add perturbation (external stimuli/noise)
        if self.perturbation_level > 0:
            perturbation = np.random.randn(self.embedding_dim) * self.perturbation_level
            transformed = transformed + perturbation
        
        # Normalize to unit length
        return transformed / np.linalg.norm(transformed)

    def trace_trajectory(self, initial_state, steps):
        """
        Generate a trajectory of states from an initial state.
        
        Parameters
        ----------
        initial_state : ndarray
            Initial thought vector/state
        steps : int
            Number of recursive steps to trace
            
        Returns
        -------
        ndarray
            Array of states forming the trajectory
        """
        self.reset_memory()
        
        # Normalize initial state
        initial_state = initial_state / np.linalg.norm(initial_state)
        trajectory = [initial_state]
        
        for _ in range(steps):
            next_state = self.process(trajectory[-1])
            trajectory.append(next_state)
            
        return np.array(trajectory)
    
    def analyze_dynamics(self, trajectory):
        """
        Analyze the dynamics of a trajectory.
        
        Parameters
        ----------
        trajectory : ndarray
            Array of states forming a trajectory
            
        Returns
        -------
        dict
            Dictionary of analysis results including:
            - cosine_similarities: list of similarities between consecutive states
            - fixed_point: boolean indicating if trajectory converges to a fixed point
            - has_cycle: boolean indicating if trajectory exhibits a limit cycle
            - cycle_length: length of the cycle if one is detected
            - final_similarity: similarity between the last two states
        """
        # Calculate cosine similarities between consecutive states
        cosine_similarities = []
        for i in range(len(trajectory)-1):
            cos_sim = np.dot(trajectory[i], trajectory[i+1]) / (np.linalg.norm(trajectory[i]) * np.linalg.norm(trajectory[i+1]))
            cosine_similarities.append(cos_sim)
            
        # Check for fixed point (convergence)
        last_similarities = cosine_similarities[-3:] if len(cosine_similarities) >= 3 else cosine_similarities
        fixed_point = all(sim > 0.999 for sim in last_similarities) if last_similarities else False
        
        # Check for limit cycles (repeating patterns)
        has_cycle = False
        cycle_length = 0
        if len(trajectory) > 10 and not fixed_point:
            # Compare last state to previous states to find cycles
            last_state = trajectory[-1]
            for i in range(len(trajectory)-10, len(trajectory)-1):
                similarity = np.dot(last_state, trajectory[i]) / (np.linalg.norm(last_state) * np.linalg.norm(trajectory[i]))
                if similarity > 0.999:
                    has_cycle = True
                    cycle_length = len(trajectory) - 1 - i
                    break
                    
        return {
            'cosine_similarities': cosine_similarities,
            'fixed_point': fixed_point,
            'has_cycle': has_cycle,
            'cycle_length': cycle_length,
            'final_similarity': cosine_similarities[-1] if cosine_similarities else None
        }

    def get_system_info(self):
        """
        Get information about the system's properties.
        
        Returns
        -------
        dict
            Dictionary containing system information including:
            - max_eigenvalue_magnitude: magnitude of the largest eigenvalue
            - stability_prediction: prediction of system stability
            - eigenvalues: array of system eigenvalues
        """
        # Analyze eigenvalues to predict system behavior
        max_eigenvalue_magnitude = max(abs(ev) for ev in self.eigenvalues)
        
        return {
            'max_eigenvalue_magnitude': max_eigenvalue_magnitude,
            'stability_prediction': 'Stable' if max_eigenvalue_magnitude < 1 else 'Unstable',
            'eigenvalues': self.eigenvalues
        }
        
    def explore_single_trajectory(self, steps=30, random_init=True, init_state=None):
        """
        Generate and visualize a single trajectory.
        
        Parameters
        ----------
        steps : int, optional (default=30)
            Number of recursive steps to trace
        random_init : bool, optional (default=True)
            Whether to use a random initial state
        init_state : ndarray, optional (default=None)
            Initial state to use if random_init is False
            
        Returns
        -------
        dict
            Dictionary containing the trajectory, analysis results, and PCA results
        """
        if random_init or init_state is None:
            init_state = np.random.randn(self.embedding_dim)
            init_state = init_state / np.linalg.norm(init_state)
        
        # Generate trajectory
        trajectory = self.trace_trajectory(init_state, steps)
        
        # Analyze dynamics
        dynamics = self.analyze_dynamics(trajectory)
        
        # Visualize with PCA
        pca = PCA(n_components=2)
        trajectory_2d = pca.fit_transform(trajectory)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Plot the trajectory
        plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], '-o', linewidth=2, 
                label="Thought Trajectory")
        
        # Highlight start and end
        plt.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], s=100, c='green', 
                   edgecolor='black', zorder=10, label="Initial Thought")
        plt.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], s=150, c='red', 
                   marker='*', edgecolor='black', zorder=10, label="Final Thought")
        
        # Add colorbar to show progression
        scatter = plt.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], 
                            c=np.arange(len(trajectory)), cmap='viridis', 
                            edgecolor='k', s=80, zorder=5)
        plt.colorbar(scatter, label='Recursion Depth')
        
        explained_var = pca.explained_variance_ratio_
        plt.xlabel(f"Principal Component 1 ({explained_var[0]:.1%} variance)")
        plt.ylabel(f"Principal Component 2 ({explained_var[1]:.1%} variance)")
        
        # Determine the dynamics type for the title
        dynamics_type = "Fixed Point" if dynamics['fixed_point'] else \
                       f"Limit Cycle (len={dynamics['cycle_length']})" if dynamics['has_cycle'] else \
                       "Complex Dynamics"
        
        title = (f"Recursive Self-Reference: {dynamics_type}\n"
                f"Nonlinearity={self.nonlinearity_strength}, "
                f"Memory={self.memory_strength}, "
                f"Perturbation={self.perturbation_level}")
        plt.title(title, fontsize=14)
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Return results
        return {
            'trajectory': trajectory,
            'trajectory_2d': trajectory_2d,
            'dynamics': dynamics,
            'pca': pca
        }


def create_random_thought_vector(dim=20, seed=None):
    """
    Create a random normalized thought vector.
    
    Parameters
    ----------
    dim : int, optional (default=20)
        Dimensionality of the vector
    seed : int, optional (default=None)
        Random seed for reproducibility
        
    Returns
    -------
    ndarray
        Normalized random vector
    """
    if seed is not None:
        np.random.seed(seed)
    v = np.random.randn(dim)
    return v / np.linalg.norm(v)


def cosine_similarity(v1, v2):
    """
    Calculate cosine similarity between two vectors.
    
    Parameters
    ----------
    v1, v2 : ndarray
        Input vectors
        
    Returns
    -------
    float
        Cosine similarity between the vectors
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# Simple usage example
if __name__ == "__main__":
    # Create a system
    system = RecursiveCognitionSystem(
        embedding_dim=20,
        nonlinearity_strength=0.3,
        memory_strength=0.1
    )
    
    # Explore a single trajectory
    results = system.explore_single_trajectory(steps=30)
    
    # Display dynamics analysis
    print("Dynamics Analysis:")
    print(f"Fixed point: {results['dynamics']['fixed_point']}")
    print(f"Limit cycle: {results['dynamics']['has_cycle']}")
    if results['dynamics']['has_cycle']:
        print(f"Cycle length: {results['dynamics']['cycle_length']}")
    print(f"Final similarity: {results['dynamics']['final_similarity']:.4f}")
    
    # Display system info
    system_info = system.get_system_info()
    print("\nSystem Information:")
    print(f"Max eigenvalue magnitude: {system_info['max_eigenvalue_magnitude']:.4f}")
    print(f"Stability prediction: {system_info['stability_prediction']}")
