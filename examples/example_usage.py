"""
Example usage script for Evolutionary Model Merge

This script demonstrates how to use the evolutionary merge algorithm with different configurations.
"""

from evolution_merge.core import evolutionary_merge
from evolution_merge.config import DEFAULT_PARAMS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_basic_usage():
    """
    Basic example of using the evolutionary merge algorithm
    """
    print("=== Basic Example ===")
    
    # Define models to merge (example models - make sure they have the same architecture)
    models = [
        {'name': 'beyoru/MinCoder-4B-Exp'},
        {'name': 'TMLR-Group-HF/Self-Certainty-Qwen3-8B-Base-MATH'}
    ]

    
    print(f"Models to merge: {[model['name'] for model in models]}")
    
    try:
        # Run evolutionary merge with default parameters
        best_weights, best_score, merged_model, tokenizer = evolutionary_merge(
            models=models,
            **DEFAULT_PARAMS  # Using default parameters from config
        )
        
        print(f"Best weights: {best_weights}")
        print(f"Best perplexity: {best_score:.4f}")
        print("Merged model created successfully!\n")
        
    except Exception as e:
        print(f"Error during evolutionary merge: {str(e)}")
        print("This may happen if the models are not compatible for merging.\n")


def example_custom_parameters():
    """
    Example with custom parameters for the evolutionary merge algorithm
    """
    print("=== Custom Parameters Example ===")
    
    # Define models to merge
    models = [
        {'name': 'microsoft/DialoGPT-medium'},
        {'name': 'microsoft/DialoGPT-small'}
    ]
    
    print(f"Models to merge: {[model['name'] for model in models]}")
    
    try:
        # Run evolutionary merge with custom parameters
        best_weights, best_score, merged_model, tokenizer = evolutionary_merge(
            models=models,
            population_size=8,      # Smaller population
            generations=4,          # Fewer generations for faster execution
            mutation_rate=0.15,     # Custom mutation rate
            eval_samples=20         # Fewer evaluation samples for faster execution
        )
        
        print(f"Best weights: {best_weights}")
        print(f"Best perplexity: {best_score:.4f}")
        print("Merged model created with custom parameters!\n")
        
    except Exception as e:
        print(f"Error during evolutionary merge: {str(e)}")
        print("This may happen if the models are not compatible for merging.\n")


def example_different_models():
    """
    Example with different models (commented out to avoid errors if models don't exist)
    """
    print("=== Different Models Example (Template) ===")
    
    # Example of how to use with different compatible models
    # IMPORTANT: Make sure models have the same architecture for merging
    
    models = [
        # {'name': 'your-compatible-model-1'},
        # {'name': 'your-compatible-model-2'}
    ]
    
    if models:  # Only run if we have models defined
        print(f"Models to merge: {[model['name'] for model in models]}")
        
        try:
            best_weights, best_score, merged_model, tokenizer = evolutionary_merge(
                models=models,
                population_size=6,
                generations=3,
                eval_samples=15
            )
            
            print(f"Best weights: {best_weights}")
            print(f"Best perplexity: {best_score:.4f}")
            print("Merged model created with different models!\n")
            
        except Exception as e:
            print(f"Error during evolutionary merge: {str(e)}")
            print("Make sure to use models that have the same base architecture for merging.\n")
    else:
        print("No models specified for this example.")
        print("To use this example, specify compatible models with the same architecture.\n")


def main():
    """
    Main function to run all examples
    """
    print("Evolutionary Model Merge - Example Usage\n")
    
    example_basic_usage()
    example_custom_parameters()
    example_different_models()
    
    print("Examples completed!")


if __name__ == "__main__":
    main()