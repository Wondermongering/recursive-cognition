"""
Recursive Cognition Explorer

A computational framework for exploring recursive self-reference in cognitive systems.
"""

from .core import RecursiveCognitionSystem, create_random_thought_vector, cosine_similarity
from .visualization import visualize_trajectory, visualize_multiple_trajectories, plot_stability_curve
from .analysis import analyze_convergence, analyze_trajectory_smoothness, detect_cycles, eigenvalue_analysis

__version__ = '0.1.0'
__all__ = [
    'RecursiveCognitionSystem',
    'create_random_thought_vector',
    'cosine_similarity',
    'visualize_trajectory',
    'visualize_multiple_trajectories',
    'plot_stability_curve',
    'analyze_convergence',
    'analyze_trajectory_smoothness',
    'detect_cycles',
    'eigenvalue_analysis'
]
