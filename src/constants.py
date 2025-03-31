"""
constants.py - Default parameters for social media simulation with 2D topic space
"""
from datetime import date

# Initial population - targets for 2012
DEFAULT_INITIAL_HUMANS = 184
DEFAULT_INITIAL_BOTS = 16

# Forced feed parameters
DEFAULT_FORCED_FEED_PROBABILITY = 0.65  # % chance of seeing forced bot content
DEFAULT_FORCED_FEED_MAX_POSTS = 5  # Maximum number of forced posts per step

# Growth rates - calibrated to reach 2025 targets (~480 humans, ~120 bots by step 50)
DEFAULT_HUMAN_CREATION_RATE = 8
DEFAULT_BOT_CREATION_RATE = 3

# Bot ban rate multiplier (allows adjusting detection/ban rate)
DEFAULT_BOT_BAN_RATE_MULTIPLIER = 1

# Network parameters
DEFAULT_CONNECTION_REWIRING_PROB = 0.1
DEFAULT_NETWORK_STABILITY = 0.9
DEFAULT_TOPIC_SHIFT_FREQUENCY = 5

# Interaction parameters
DEFAULT_HUMAN_HUMAN_POSITIVE_BIAS = 0.5
DEFAULT_HUMAN_BOT_NEGATIVE_BIAS = 0.8
DEFAULT_HUMAN_SATISFACTION_INIT = 90

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

# Bot type distribution (overall defaults)
DEFAULT_BOT_TYPE_WEIGHTS = {
    'spam': 0.5,            # 50% of bots
    'misinformation': 0.25, # 25% of bots
    'astroturfing': 0.25    # 25% of bots
}

# Quadrant-specific bot type distributions
DEFAULT_BOT_TYPE_BY_QUADRANT = {
    'tech_business': {
        'spam': 0.3,            # 30%
        'misinformation': 0.2,  # 20%
        'astroturfing': 0.5     # 50% - higher in tech/business
    },
    'politics_news': {
        'spam': 0.2,            # 20%
        'misinformation': 0.6,  # 60% - higher in politics/news
        'astroturfing': 0.2     # 20%
    },
    'hobbies': {
        'spam': 0.7,            # 70% - higher in hobbies
        'misinformation': 0.2,  # 20%
        'astroturfing': 0.1     # 10%
    },
    'pop_culture': {
        'spam': 0.6,            # 60% - higher in pop culture
        'misinformation': 0.3,  # 30%
        'astroturfing': 0.1     # 10%
    }
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

# Grid parameters
DEFAULT_GRID_RESOLUTION = 100  # Grid resolution for topic space

# Agent processing parameters
DEFAULT_AGENTS_TO_PROCESS_PER_STEP = 10  # Maximum agents to process per step

# Connection parameters (additional)
DEFAULT_CONNECTION_BREAKING_BASE_PROB = 0.02  # Base probability for breaking connections
DEFAULT_CONNECTION_DECAY_PROB = 0.005  # Probability for random connection decay
DEFAULT_SUPER_USER_CONNECTION_BREAKING_PROB = 0.03  # Probability for super users to break connections
DEFAULT_BOT_BLOCK_SATISFACTION_THRESHOLD = -0.15  # Satisfaction threshold for blocking bots

# Human parameters
DEFAULT_REGULAR_USER_SATISFACTION_THRESHOLD = 0  # Satisfaction threshold for regular users
DEFAULT_SUPER_USER_SATISFACTION_THRESHOLD = -20  # Satisfaction threshold for super users
DEFAULT_HUMAN_IRRITABILITY_RANGE = (0.5, 2)  # Range for human irritability
DEFAULT_HUMAN_AUTHENTICITY_RANGE = (0.5, 2)  # Range for human authenticity
DEFAULT_HUMAN_BASE_ACTIVENESS_RANGE = (0.2, 0.6)  # Range for human base activeness
DEFAULT_HUMAN_BASE_POST_FREQUENCY_RANGE = (0.1, 0.4)  # Range for human base post frequency
DEFAULT_HUMAN_POPULARITY_RANGE = (0.3, 0.95)  # Range for human popularity

# Super user parameters (additional)
DEFAULT_SUPER_USER_TOPIC_MOBILITY_RANGE = (0.01, 0.03)  # Range for super user topic mobility
DEFAULT_SUPER_USER_INFLUENCE_RADIUS_RANGE = (0.2, 0.3)  # Range for super user influence radius
DEFAULT_SUPER_USER_TOPIC_LEADERSHIP_RANGE = (0.5, 0.8)  # Range for super user topic leadership
DEFAULT_REGULAR_USER_TOPIC_MOBILITY_RANGE = (0.08, 0.12)  # Range for regular user topic mobility

# Bot parameters
DEFAULT_BOT_TOPIC_MOBILITY_RANGE = (0.02, 0.1)  # Range for bot topic mobility
DEFAULT_BOT_MALICIOUS_POST_RATE_RANGE = (0.05, 0.9)  # Range for bot malicious post rate

# Bot type quadrant bias
DEFAULT_BOT_TYPE_QUADRANT_BIAS = {
    'spam': {
        'tech_business': 0.2,
        'politics_news': 0.2,
        'pop_culture': 0.5,
        'hobbies': 0.1
    },
    'misinformation': {
        'tech_business': 0.3,
        'politics_news': 0.6,
        'pop_culture': 0.05,
        'hobbies': 0.05
    },
    'astroturfing': {
        'tech_business': 0.7,
        'politics_news': 0.2,
        'pop_culture': 0.05,
        'hobbies': 0.05
    }
}

# Quadrant attractiveness weights
DEFAULT_QUADRANT_ATTRACTIVENESS_WEIGHT = 0.7  # Weight for quadrant attractiveness
DEFAULT_TYPE_QUADRANT_BIAS_WEIGHT = 0.3  # Weight for type quadrant bias

# General parameters
DEFAULT_SIMULATION_START_DATE = date(2022, 1, 1)  # Start date for simulation
DEFAULT_HUMAN_CONNECTION_LIMIT = 8  # Connection limit for humans
DEFAULT_BOT_CONNECTION_LIMIT = 8  # Connection limit for bots