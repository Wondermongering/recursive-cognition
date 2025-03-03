# Recursive Cognition Explorer

A computational framework for exploring recursive self-reference in cognitive systems - how thoughts evolve when they reflect upon themselves.

## Overview

The Recursive Cognition Explorer provides tools for simulating, visualizing, and analyzing recursive self-reference processes - systems that process their own states and feed results back into themselves. This mathematical approach allows exploration of questions about consciousness, thought stability, creative processes, and the emergence of complexity from simple recursive rules.

## Key Features

- **Interactive Exploration**: Visualize how thoughts evolve through recursive processing
- **Parameter Control**: Examine how nonlinearity, memory, and perturbation affect thinking paths
- **Multiple Starting Points**: Test whether different initial thoughts converge to similar conclusions
- **Dynamics Analysis**: Detect fixed points (stable conclusions), limit cycles (repeating patterns), and chaotic attractors
- **Artistic Applications**: Explore aesthetic evolution through recursive refinement

## Examples

The repository contains several ready-to-run examples:

- `basic_exploration.py`: Simple introduction to the system's core functionality
- `multiple_starting_points.py`: Explore convergence from different initial thoughts
- `nonlinearity_effects.py`: How nonlinearity shapes thought evolution
- `memory_influence.py`: Effects of memory on thought coherence
- `perturbation_analysis.py`: Resilience of thinking patterns to external noise
- `artistic_cognition.py`: Application to creative processes

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/recursive-cognition.git
cd recursive-cognition

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run a basic example
python examples/basic_exploration.py
```

For a simple programmatic example:

```python
from recursive_cognition import RecursiveCognitionSystem

# Create a system with default parameters
system = RecursiveCognitionSystem(
    embedding_dim=20,
    nonlinearity_strength=0.3,
    memory_strength=0.1
)

# Explore a single trajectory
results = system.explore_single_trajectory(steps=30)

# Display dynamics analysis
print(f"Fixed point: {results['dynamics']['fixed_point']}")
print(f"Final similarity: {results['dynamics']['final_similarity']:.4f}")
```

## Understanding the Model

The system represents "thoughts" as vectors in a high-dimensional space. Each recursive step applies:

1. **Transformation**: A matrix multiplication that models how thinking transforms concepts
2. **Nonlinearity**: Controlled non-linear activation (like neural networks) 
3. **Memory Influence**: Optional influence from earlier states (beyond the immediate last one)
4. **Perturbation**: Random noise that simulates external influences/stimuli

Through dimensionality reduction techniques, we can visualize these high-dimensional trajectories to see how thinking evolves over time.

## Philosophical Background

This project draws inspiration from:

- Douglas Hofstadter's **strange loops** concept and the role of self-reference in consciousness
- Dynamical systems theory approach to **emergence** and complexity
- Computational models of **metacognition** (thinking about thinking)

## Applications

The framework has applications across multiple domains:

- **Cognitive Science**: Modeling how humans recursively refine their beliefs
- **AI Research**: Understanding emergent properties in self-referential systems
- **Consciousness Studies**: Exploring mathematical models of self-awareness
- **Creative Process Analysis**: Modeling how ideas evolve through successive refinement
- **Decision Theory**: Analyzing how deliberation affects belief stability

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure

```
recursive-cognition/
├── recursive_cognition/          # Core package
│   ├── __init__.py
│   ├── core.py                   # Core system implementation
│   ├── visualization.py          # Visualization tools
│   ├── analysis.py               # Analysis utilities
│   └── experiments.py            # Pre-defined experiments
├── examples/                     # Example scripts
├── notebooks/                    # Jupyter notebooks with tutorials
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Citation

If you use this work in your research, please cite:

```
@software{recursive_cognition,
  author = {Tomás Pellisari},
  title = {Recursive Cognition Explorer},
  year = {2025},
  url = {https://github.com/Wondermongering/recursive-cognition}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
