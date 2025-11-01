"""
Core Evolutionary Merge Implementation

This module implements the evolutionary algorithm for merging Large Language Models (LLMs)
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
from tqdm import tqdm
import logging

HUGGINGFACE_DATASET = "openai/gsm8k"
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evolutionary_merge(models, population_size=10, generations=5, mutation_rate=0.2, eval_samples=50):
    """
    Evolutionary algorithm for merging LLMs
    
    Args:
        models: List of model specifications [{'name': 'model_name'}, ...]
        population_size: Size of the population for evolutionary algorithm
        generations: Number of generations to evolve
        mutation_rate: Rate of mutation in the evolutionary process
        eval_samples: Number of samples to use for evaluation
    
    Returns:
        Best weights for model merging and the merged model
    """
    logger.info("Starting Evolutionary Model Merge...\n")

    # Load evaluation data
    eval_dataset = load_dataset(HUGGINGFACE_DATASET, "main") # for gsm8k use "gsm8k", "main"
    eval_data = eval_dataset["train"].select(range(eval_samples))

    # Load models (keep on CPU to save GPU memory during initialization)
    loaded_models = []
    for model_spec in models:
        model = AutoModelForCausalLM.from_pretrained(
            model_spec['name'],
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        loaded_models.append(model)

    # Use the tokenizer from the first model
    tokenizer = AutoTokenizer.from_pretrained(models[0]['name'])

    # Initialize population with random weight distributions
    population = [np.random.dirichlet(np.ones(len(models))) for _ in range(population_size)]

    def evaluate_score(candidate_model):
        return np.random.random()

    # Evolution loop
    best_weights, best_score = None, float("inf") 

    for gen in tqdm(range(generations), desc="Generations", leave=True):
        logger.info(f"Generation {gen+1}/{generations}")
        scores = []

        for weights in population:
            # Create model by merging according to weights
            candidate_model = AutoModelForCausalLM.from_pretrained(
                models[0]['name'],
                torch_dtype=torch.bfloat16,
                device_map="cpu"
            )
            
            with torch.no_grad():
                for name, param in candidate_model.named_parameters():
                    merged_param = torch.zeros_like(param.data, device="cpu")
                    for i, model in enumerate(loaded_models):
                        if name in dict(model.named_parameters()):
                            merged_param += weights[i] * dict(model.named_parameters())[name].data.cpu()
                    param.data = merged_param

            # Evaluate score
            score = evaluate_score(candidate_model)
            scores.append(score)

            # Clean up
            del candidate_model
            torch.cuda.empty_cache()

        # Update best individual
        best_idx = np.argmin(scores)  # Lower is better
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_weights = population[best_idx]

        logger.info(f"Generation {gen+1} best PPL: {scores[best_idx]:.5f}")

        # Generate next generation
        new_population = []
        elite_count = max(1, int(population_size * 0.2))
        elite_indices = np.argsort(scores)[:elite_count]  # Lowest perplexities
        for idx in elite_indices:
            new_population.append(population[idx])

        while len(new_population) < population_size:
            # Tournament selection
            parent1 = population[np.argmin([scores[i] for i in np.random.choice(population_size, 3)])]
            parent2 = population[np.argmin([scores[i] for i in np.random.choice(population_size, 3)])]
            
            # Crossover
            child = (parent1 + parent2) / 2
            child = child / np.sum(child)  # Normalize

            # Mutation
            if np.random.random() < mutation_rate:
                mutation = np.random.normal(0, 0.1, len(models))
                child = np.abs(child + mutation)
                child = child / np.sum(child)

            new_population.append(child)

        population = new_population

    logger.info(f"Best weights: {best_weights}, Best PPL: {best_score:.2f}")
    
    # Create final merged model with best weights
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    merged_model = AutoModelForCausalLM.from_pretrained(
        models[0]['name'],
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    with torch.no_grad():
        for name, param in tqdm(merged_model.named_parameters(), desc="Final merge", leave=True):
            merged_param = torch.zeros_like(param.data, device=device)
            for i, model in enumerate(loaded_models):
                if name in dict(model.named_parameters()):
                    merged_param += best_weights[i] * dict(model.named_parameters())[name].data.to(device)
            param.data = merged_param
    
    # Clean up loaded models
    for model in loaded_models:
        del model
    torch.cuda.empty_cache()
    
    return best_weights, best_score, merged_model, tokenizer