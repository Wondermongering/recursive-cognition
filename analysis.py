"""
Analysis utilities for the Recursive Cognition Explorer.

This module provides functions for analyzing trajectories, detecting patterns,
and measuring various properties of recursive cognition systems.
"""

import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.signal import find_peaks


def cosine_similarity_matrix(trajectory):
    """
    Calculate cosine similarity matrix between all states in a trajectory.
    
    Parameters
    ----------
    trajectory : ndarray
        Array of states forming the trajectory
        
    Returns
    -------
    ndarray
        Square matrix of cosine similarities between all state pairs
    """
    # Normalize vectors
    norms = np.sqrt(np.sum(trajectory**2, axis=1))
    normalized = trajectory / norms[:, np.newaxis]
    
    # Calculate similarity matrix
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix


def analyze_convergence(trajectories, threshold=0.99):
    """
    Analyze convergence properties of one or more trajectories.
    
    Parameters
    ----------
    trajectories : list of ndarray or ndarray
        Single trajectory or list of trajectories to analyze
    threshold : float, optional
        Similarity threshold for convergence detection
        
    Returns
    -------
    dict
        Dictionary of convergence metrics
    """
    # Convert single trajectory to list
    if not isinstance(trajectories, list):
        trajectories = [trajectories]
    
    # Initialize result metrics
    results = {
        'converged': [],
        'convergence_step': [],
        'final_similarity': [],
        'min_similarity': [],
        'average_similarity': []
    }
    
    # Analyze each trajectory
    for trajectory in trajectories:
        # Calculate cosine similarities between consecutive states
        cosine_similarities = []
        for i in range(len(trajectory)-1):
            v1 = trajectory[i]
            v2 = trajectory[i+1]
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cosine_similarities.append(sim)
        
        # Check for convergence
        converged = False
        convergence_step = None
        
        # Look for consecutive similarities above threshold
        for i in range(len(cosine_similarities)-2):
            if all(sim >= threshold for sim in cosine_similarities[i:i+3]):
                converged = True
                convergence_step = i
                break
        
        # Store results
        results['converged'].append(converged)
        results['convergence_step'].append(convergence_step)
        results['final_similarity'].append(cosine_similarities[-1])
        results['min_similarity'].append(min(cosine_similarities))
        results['average_similarity'].append(np.mean(cosine_similarities))
    
    # Calculate overall metrics
    results['any_converged'] = any(results['converged'])
    results['all_converged'] = all(results['converged'])
    results['convergence_rate'] = sum(results['converged']) / len(trajectories)
    
    return results


def analyze_trajectory_smoothness(trajectory):
    """
    Analyze the smoothness of a trajectory.
    
    Parameters
    ----------
    trajectory : ndarray
        Array of states forming the trajectory
        
    Returns
    -------
    dict
        Dictionary of smoothness metrics
    """
    # Calculate cosine similarities between consecutive states
    cosine_similarities = []
    for i in range(len(trajectory)-1):
        v1 = trajectory[i]
        v2 = trajectory[i+1]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosine_similarities.append(sim)
    
    # Calculate smoothness metrics
    avg_similarity = np.mean(cosine_similarities)
    std_similarity = np.std(cosine_similarities)
    min_similarity = np.min(cosine_similarities)
    
    # Calculate second derivatives (approximation of curvature)
    curvature = []
    for i in range(1, len(cosine_similarities)-1):
        second_deriv = (cosine_similarities[i+1] - 2*cosine_similarities[i] + cosine_similarities[i-1])
        curvature.append(abs(second_deriv))
    
    avg_curvature = np.mean(curvature) if curvature else 0
    
    # Detect sharp turns (low similarity points)
    sharp_turns = [i for i, sim in enumerate(cosine_similarities) if sim < 0.9]
    
    return {
        'average_similarity': avg_similarity,
        'std_similarity': std_similarity,
        'min_similarity': min_similarity,
        'average_curvature': avg_curvature,
        'sharp_turns': sharp_turns,
        'smoothness_score': avg_similarity / (1 + avg_curvature)  # Higher is smoother
    }


def detect_cycles(trajectory, similarity_threshold=0.99, min_cycle_length=2, max_cycle_length=None):
    """
    Detect cycles (repeating patterns) in a trajectory.
    
    Parameters
    ----------
    trajectory : ndarray
        Array of states forming the trajectory
    similarity_threshold : float, optional
        Similarity threshold for cycle detection
    min_cycle_length : int, optional
        Minimum length of cycles to detect
    max_cycle_length : int, optional
        Maximum length of cycles to detect (default: half trajectory length)
        
    Returns
    -------
    dict
        Dictionary of cycle detection results
    """
    if max_cycle_length is None:
        max_cycle_length = len(trajectory) // 2
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity_matrix(trajectory)
    
    # Initialize results
    cycles = []
    
    # Check for cycles of different lengths
    for cycle_length in range(min_cycle_length, min(max_cycle_length+1, len(trajectory)//2 + 1)):
        for start_idx in range(len(trajectory) - 2*cycle_length):
            # Check if the pattern repeats
            pattern_match = True
            for offset in range(cycle_length):
                idx1 = start_idx + offset
                idx2 = start_idx + cycle_length + offset
                
                if similarity_matrix[idx1, idx2] < similarity_threshold:
                    pattern_match = False
                    break
            
            if pattern_match:
                cycles.append({
                    'start_index': start_idx,
                    'length': cycle_length,
                    'confidence': np.mean([similarity_matrix[start_idx + i, start_idx + cycle_length + i] 
                                         for i in range(cycle_length)])
                })
                break  # Move to next cycle length
    
    # Find the most likely cycle (highest confidence)
    most_likely_cycle = max(cycles, key=lambda x: x['confidence']) if cycles else None
    
    return {
        'has_cycle': len(cycles) > 0,
        'all_detected_cycles': cycles,
        'most_likely_cycle': most_likely_cycle,
        'cycle_count': len(cycles)
    }


def eigenvalue_analysis(W):
    """
    Analyze eigenvalues of a transformation matrix to predict system behavior.
    
    Parameters
    ----------
    W : ndarray
        Transformation matrix
        
    Returns
    -------
    dict
        Dictionary of eigenvalue analysis results
    """
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(W)
    
    # Calculate magnitudes and sort by magnitude
    magnitudes = np.abs(eigenvalues)
    sorted_indices = np.argsort(magnitudes)[::-1]
    
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_magnitudes = magnitudes[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Predict stability
    max_magnitude = sorted_magnitudes[0]
    is_stable = max_magnitude < 1
    
    # Check for complex eigenvalues (oscillatory behavior)
    complex_eigenvalues = [ev for ev in eigenvalues if np.imag(ev) != 0]
    has_oscillations = len(complex_eigenvalues) > 0
    
    # Dominant mode
    dominant_eigenvector = sorted_eigenvectors[:, 0]
    
    return {
        'eigenvalues': sorted_eigenvalues,
        'magnitudes': sorted_magnitudes,
        'eigenvectors': sorted_eigenvectors,
        'max_magnitude': max_magnitude,
        'is_stable': is_stable,
        'has_oscillations': has_oscillations,
        'oscillation_count': len(complex_eigenvalues) // 2,  # Complex eigenvalues come in conjugate pairs
        'dominant_eigenvector': dominant_eigenvector,
        'stability_margin': 1 - max_magnitude if is_stable else max_magnitude - 1,
        'stability_prediction': 'Stable' if is_stable else 'Unstable'
    }


def analyze_multiple_trajectories(trajectories, labels=None):
    """
    Analyze and compare multiple trajectories.
    
    Parameters
    ----------
    trajectories : list of ndarray
        List of trajectories to analyze
    labels : list of str, optional
        Labels for each trajectory
        
    Returns
    -------
    pd.DataFrame
        DataFrame with analysis results for each trajectory
    """
    if labels is None:
        labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]
    
    results = []
    
    for i, (trajectory, label) in enumerate(zip(trajectories, labels)):
        # Convergence analysis
        convergence = analyze_convergence(trajectory)
        
        # Smoothness analysis
        smoothness = analyze_trajectory_smoothness(trajectory)
        
        # Cycle detection
        cycles = detect_cycles(trajectory)
        
        # Combine results
        result = {
            'Label': label,
            'Length': len(trajectory),
            'Converged': convergence['converged'][0],
            'Convergence Step': convergence['convergence_step'][0],
            'Final Similarity': convergence['final_similarity'][0],
            'Average Similarity': convergence['average_similarity'][0],
            'Smoothness Score': smoothness['smoothness_score'],
            'Has Cycle': cycles['has_cycle'],
            'Cycle Length': cycles['most_likely_cycle']['length'] if cycles['most_likely_cycle'] else None,
            'Trajectory Type': _determine_trajectory_type(convergence, smoothness, cycles)
        }
        
        results.append(result)
    
    # Create DataFrame
    return pd.DataFrame(results)


def _determine_trajectory_type(convergence, smoothness, cycles):
    """
    Determine the type of trajectory based on analysis results.
    
    Parameters
    ----------
    convergence : dict
        Convergence analysis results
    smoothness : dict
        Smoothness analysis results
    cycles : dict
        Cycle detection results
        
    Returns
    -------
    str
        Trajectory type description
    """
    if convergence['converged'][0]:
        return 'Fixed Point (Stable Convergence)'
    elif cycles['has_cycle']:
        return f'Limit Cycle (Length {cycles["most_likely_cycle"]["length"]})'
    elif smoothness['average_similarity'][0] < 0.8:
        return 'Chaotic (Low Similarity)'
    else:
        return 'Complex (Non-convergent, Non-cyclic)'


def analyze_parameter_sensitivity(trajectories, parameters, parameter_name):
    """
    Analyze sensitivity of trajectory properties to a parameter.
    
    Parameters
    ----------
    trajectories : list of ndarray
        List of trajectories for different parameter values
    parameters : list of float
        List of parameter values corresponding to each trajectory
    parameter_name : str
        Name of the parameter being varied
        
    Returns
    -------
    dict
        Dictionary of sensitivity analysis results
    """
    # Calculate metrics for each trajectory
    metrics = []
    
    for trajectory in trajectories:
        convergence = analyze_convergence(trajectory)
        smoothness = analyze_trajectory_smoothness(trajectory)
        cycles = detect_cycles(trajectory)
        
        metrics.append({
            'converged': convergence['converged'][0],
            'convergence_step': convergence['convergence_step'][0],
            'final_similarity': convergence['final_similarity'][0],
            'average_similarity': convergence['average_similarity'][0],
            'smoothness_score': smoothness['smoothness_score'],
            'has_cycle': cycles['has_cycle'],
            'cycle_length': cycles['most_likely_cycle']['length'] if cycles['most_likely_cycle'] else None
        })
    
    # Calculate correlations between parameter and metrics
    correlations = {}
    
    for metric in ['final_similarity', 'average_similarity', 'smoothness_score']:
        metric_values = [m[metric] for m in metrics]
        
        # Filter out None values
        valid_indices = [i for i, v in enumerate(metric_values) if v is not None]
        valid_params = [parameters[i] for i in valid_indices]
        valid_values = [metric_values[i] for i in valid_indices]
        
        if len(valid_params) > 1:
            correlation = np.corrcoef(valid_params, valid_values)[0, 1]
            correlations[metric] = correlation
    
    # Detect parameter threshold for convergence
    convergence_threshold = None
    previous_converged = metrics[0]['converged']
    
    for i in range(1, len(parameters)):
        if metrics[i]['converged'] != previous_converged:
            # Threshold between parameters[i-1] and parameters[i]
            convergence_threshold = (parameters[i-1] + parameters[i]) / 2
            break
        previous_converged = metrics[i]['converged']
    
    # Detect parameter threshold for cycle formation
    cycle_threshold = None
    previous_has_cycle = metrics[0]['has_cycle']
    
    for i in range(1, len(parameters)):
        if metrics[i]['has_cycle'] != previous_has_cycle:
            # Threshold between parameters[i-1] and parameters[i]
            cycle_threshold = (parameters[i-1] + parameters[i]) / 2
            break
        previous_has_cycle = metrics[i]['has_cycle']
    
    return {
        'parameter_name': parameter_name,
        'parameter_values': parameters,
        'correlations': correlations,
        'convergence_threshold': convergence_threshold,
        'cycle_threshold': cycle_threshold,
        'sensitivity_score': max(abs(corr) for corr in correlations.values()) if correlations else 0
    }
