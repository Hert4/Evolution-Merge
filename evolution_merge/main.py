"""
Main module for Evolutionary Model Merge

This module provides the main function demonstrating how to use the evolutionary merge algorithm.
"""

from .core import evolutionary_merge
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main function demonstrating how to use the evolutionary merge algorithm
    """
    # Example model specification for compatible models
    # NOTE: In practice, you would need models with the same architecture for merging
    models_spec = [
        {'name': 'microsoft/DialoGPT-medium'},  # Example model
        {'name': 'microsoft/DialoGPT-small'}    # Example model - must be same architecture
    ]
    
    print("Starting Evolutionary Model Merge Process...")
    print(f"Models to merge: {[model['name'] for model in models_spec]}")
    print(f"Population size: 5, Generations: 3, Eval samples: 10")
    
    try:
        # Run the evolutionary merge
        best_weights, best_score, merged_model, tokenizer = evolutionary_merge(
            models=models_spec,
            population_size=5,
            generations=3,
            eval_samples=10
        )
        
        print(f"\nOptimization Complete!")
        print(f"Best weights: {best_weights}")
        print(f"Best perplexity: {best_score:.4f}")
        
        # Show an example of how you might use the merged model
        print(f"\nMerged model is ready for use.")
        print("Example usage after merging:")
        print("# Generate text with merged model")
        print("# inputs = tokenizer(\"Your prompt here\", return_tensors=\"pt\").to(\"cuda:0\")")
        print("# outputs = merged_model.generate(**inputs, max_length=100)")
        print("# text = tokenizer.decode(outputs[0], skip_special_tokens=True)")
        
        return best_weights, best_score, merged_model, tokenizer
        
    except Exception as e:
        print(f"Error during evolutionary merge: {str(e)}")
        print("\nNote: This example requires compatible models with the same architecture.")
        print("Make sure to use models that have the same base architecture for merging.")
        return None, None, None, None


if __name__ == "__main__":
    best_weights, best_score, merged_model, tokenizer = main()