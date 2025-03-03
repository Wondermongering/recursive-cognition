# Recursive Cognition Explorer

A computational framework for exploring recursive self-reference in thought systems, inspired by concepts from cognitive science, dynamical systems theory, and consciousness studies.

## Overview

This repository provides tools for simulating, visualizing, and analyzing recursive self-reference processes — systems that process their own states and feed them back into themselves. The mathematical models here can be used to explore questions about consciousness, thought stability, idea evolution, and the emergence of complexity from simple recursive rules.

## Philosophical Background

Inspired by Douglas Hofstadter's concept of "strange loops" and the role of self-reference in consciousness, this project provides a mathematical playground for exploring questions such as:

- Do recursive thought processes naturally converge toward stable conclusions?
- How does memory affect the coherence of developing thoughts?
- What is the relationship between nonlinearity and thought complexity?
- Are there universal "attractors" in thought-space?

## Key Features

- **Multiple Cognitive Models**: Explore various forms of recursive processing with controllable parameters
- **Rich Visualizations**: Trace thought trajectories in reduced dimensional space with PCA
- **Dynamics Analysis**: Detect fixed points, limit cycles, and chaotic attractors in recursive thinking
- **Parameter Exploration**: Examine effects of nonlinearity, memory, and external perturbations
- **Eigenvalue Analysis**: Mathematically predict system behavior and stability

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/recursive-cognition.git
cd recursive-cognition

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from recursive_cognition import RecursiveCognitionSystem

# Create a basic system
system = RecursiveCognitionSystem(
    embedding_dim=20,  
    nonlinearity_strength=0.3
)

# Generate and visualize a thought trajectory
system.explore_single_trajectory()

# Run a comprehensive exploration
from recursive_cognition.experiments import run_full_exploration
run_full_exploration()
```

## Examples

See the [examples](examples/) directory for various demonstrations:

- `basic_exploration.py`: Simple starting point for understanding the system
- `multiple_starting_points.py`: Explore convergence from different initial thoughts
- `nonlinearity_effects.py`: How nonlinearity shapes thought evolution
- `memory_influence.py`: Effects of memory on thought coherence
- `perturbation_analysis.py`: Resilience of thinking patterns to noise
- `artistic_cognition.py`: Application to creative processes

## Applications

This framework can be applied to various domains:

- **Cognitive Science**: Modeling recursive aspects of human thought
- **AI Development**: Understanding emergent properties in self-referential systems
- **Consciousness Studies**: Exploring mathematical models of self-awareness
- **Creative Process Analysis**: Modeling ideation and concept evolution
- **Social Dynamics**: Simulating how ideas evolve in social networks


## Citation

If you use this work in your research, please cite:

```
@software{recursive_cognition,
  author = {Your Name},
  title = {Recursive Cognition Explorer},
  year = {2025},
  url = {https://github.com/yourusername/recursive-cognition}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
