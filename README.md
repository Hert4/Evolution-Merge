# Evolutionary Model Merge

This repository implements an evolutionary algorithm for merging Large Language Models (LLMs) based on perplexity minimization. The approach uses evolutionary computation to find optimal weight ratios for combining multiple pre-trained models to achieve better performance.

## Overview

The Evolutionary Model Merge algorithm works as follows:

1. **Initialization**: Create an initial population of weight vectors that define how to merge different models
2. **Evaluation**: For each individual in the population, merge the models according to its weights and evaluate using perplexity on a validation set
3. **Selection**: Select the best performing individuals based on their perplexity scores
4. **Crossover**: Combine the weights of selected individuals to create new offspring
5. **Mutation**: Randomly perturb some weights to introduce diversity
6. **Repeat**: Continue for the specified number of generations

The algorithm optimizes for weights that minimize perplexity, which typically correlates with better model performance.

## Key Parameters

- `population_size`: Number of weight combinations to evaluate in each generation
- `generations`: Number of evolutionary iterations
- `mutation_rate`: Probability of randomly perturbing weights
- `eval_samples`: Number of examples to use for perplexity evaluation

These parameters can be tuned based on available computational resources and desired accuracy.

## Installation

To install the package and its dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from evolution_merge.core import evolutionary_merge

# Define models to merge (must have the same architecture)
models = [
    {'name': 'beyoru/MinCoder-4B-Exp'},
    {'name': 'TMLR-Group-HF/Self-Certainty-Qwen3-8B-Base-MATH'}
]

# Run evolutionary merge
best_weights, best_score, merged_model, tokenizer = evolutionary_merge(
    models=models,
    population_size=10,
    generations=5,
    eval_samples=50
)

print(f"Best weights: {best_weights}")
print(f"Best perplexity: {best_score:.4f}")
```

### Custom Parameters

You can customize the algorithm parameters:

```python
best_weights, best_score, merged_model, tokenizer = evolutionary_merge(
    models=models,
    population_size=20,      # Larger population
    generations=10,          # More generations
    mutation_rate=0.15,      # Different mutation rate
    eval_samples=100         # More evaluation samples
)
```

## Project Structure

```
evolution_merge/
├── evolution_merge/          # Main package directory
│   ├── __init__.py          # Package initialization
│   ├── core.py              # Core evolutionary merge implementation
│   ├── main.py              # Main execution module
│   └── config.py            # Configuration parameters
├── examples/                # Example scripts
├── tests/                   # Test files
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup file
├── README.md               # This file
└── LICENSE                 # License information
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- NumPy
- TQDM

For a complete list of dependencies, see `requirements.txt`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation is based on the concept of evolutionary computation applied to model merging, with the goal of optimizing language model performance through weight blending.

## Contribute:
> Thank you, Qwen3, for helping me to reconstruct this repo.
