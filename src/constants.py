"""
constants.py - Default parameters for social media simulation with 2D topic space
"""

# Initial population - targets for 2012
DEFAULT_INITIAL_HUMANS = 184
DEFAULT_INITIAL_BOTS = 16

# Growth rates - calibrated to reach 2025 targets (~480 humans, ~120 bots by step 50)
DEFAULT_HUMAN_CREATION_RATE = 6.0
DEFAULT_BOT_CREATION_RATE = 2.1

# Bot ban rate multiplier (allows adjusting detection/ban rate)
DEFAULT_BOT_BAN_RATE_MULTIPLIER = 1.0

# Network parameters
DEFAULT_CONNECTION_REWIRING_PROB = 0.1
DEFAULT_NETWORK_STABILITY = 0.9
DEFAULT_TOPIC_SHIFT_FREQUENCY = 5

# Interaction parameters
DEFAULT_HUMAN_HUMAN_POSITIVE_BIAS = 0.7
DEFAULT_HUMAN_BOT_NEGATIVE_BIAS = 0.8
DEFAULT_HUMAN_SATISFACTION_INIT = 100

# Connection parameters
DEFAULT_CONNECTION_FORMATION_CHANCE = 0.1
DEFAULT_CONNECTION_BREAKING_CHANCE = 0.05
DEFAULT_BOT_BLOCKING_CHANCE = 0.3

# Quadrant attractiveness values for humans (percentages from target distribution)
DEFAULT_HUMAN_QUADRANT_ATTRACTIVENESS = {
    'tech_business': 0.15,  # Q1: Serious & Individual
    'politics_news': 0.20,  # Q2: Serious & Societal
    'hobbies': 0.34,        # Q3: Casual & Individual
    'pop_culture': 0.50     # Q4: Casual & Societal
}

# Quadrant attractiveness values for bots (percentages from target distribution)
DEFAULT_BOT_QUADRANT_ATTRACTIVENESS = {
    'tech_business': 0.56,  # Q1: Serious & Individual
    'politics_news': 0.27,  # Q2: Serious & Societal
    'hobbies': 0.05,        # Q3: Casual & Individual
    'pop_culture': 0.19     # Q4: Casual & Societal
}

# Bot type distribution
DEFAULT_BOT_TYPE_WEIGHTS = {
    'spam': 0.5,            # 50% of bots
    'misinformation': 0.25, # 25% of bots
    'astroturfing': 0.25    # 25% of bots
}

# Super user parameters
DEFAULT_SUPER_USER_PROBABILITY = 0.1  # 10% of human users are super users
DEFAULT_SUPER_USER_ACTIVITY_MULTIPLIER = 4.0  # Super users are 4x more active

# Topic mobility parameters
DEFAULT_HUMAN_TOPIC_MOBILITY = 0.1
DEFAULT_BOT_TOPIC_MOBILITY = 0.05

# Echo chamber parameters
DEFAULT_ECHO_CHAMBER_STRENGTH = 0.2  # Probability of processing a human for echo chamber formation per step
DEFAULT_PROXIMITY_CONNECTION_PROBABILITY = 0.1  # Base probability for forming connections with nearby agents