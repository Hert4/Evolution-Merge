"""
Configuration module for Evolutionary Model Merge

This module contains default configurations and parameters for the evolutionary merge algorithm.
"""

DEFAULT_PARAMS = {
    'population_size': 10,
    'generations': 5,
    'mutation_rate': 0.2,
    'eval_samples': 50,
    'elite_ratio': 0.2,
    'tournament_size': 3,
    'mutation_strength': 0.1
}

EVAL_SETTINGS = {
    'dataset_name': 'openai/gsm8k',
    'dataset_config': 'main',
    'eval_split': 'train'
}

DEVICE_SETTINGS = {
    'initial_device': 'cpu',  # Initial device for loading models
    'eval_device': 'cuda:0'   # Device for evaluation (if available)
}

LOGGING_SETTINGS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s'
}

MODEL_SETTINGS = {
    'torch_dtype': 'bfloat16',
    'device_map': 'cpu'
}