"""
constants.py - Constants for the social media simulation
"""
MAX_SATISFACTION_CHANGE = 10

# Network parameters
DEFAULT_NETWORK_SIZE = 100
DEFAULT_CONNECTION_PROBABILITY = 0.1
DEFAULT_REWIRING_PROBABILITY = 0.1

# Agent parameters
DEFAULT_BOT_RATIO = 0.2
DEFAULT_HUMAN_SATISFACTION_INIT = 100
DEFAULT_SATISFACTION_THRESHOLD = 0
DEFAULT_POST_FREQUENCY_HUMAN = 0.3
DEFAULT_POST_FREQUENCY_BOT = 0.7

# Interaction parameters
HUMAN_HUMAN_POSITIVE_BIAS = 0.7  # 70% chance of positive interaction between humans
HUMAN_BOT_NEGATIVE_BIAS = 0.8    # 80% chance of negative interaction between human and bot
CONNECTION_FORMATION_CHANCE = 0.2  # Chance to form a new connection after positive interaction
CONNECTION_BREAKING_CHANCE = 0.1   # Chance to break connection after negative interaction
BOT_BLOCKING_CHANCE = 0.3         # Chance to block a bot after very negative interaction

# Simulation parameters
DEFAULT_STEPS = 365  # One year
TOPIC_SHIFT_FREQUENCY = 30  # Monthly topic shifts