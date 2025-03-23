"""
BotAgent.py - Implementation of bot agents for social media simulation
using Mesa 3.1.4 with 2D topic grid
"""

from SocialMediaAgent import SocialMediaAgent
import numpy as np
from datetime import date, timedelta


class BotAgent(SocialMediaAgent):
    """Bot agent in the social media simulation with 2D topic space."""

    def __init__(self, model):
        # Use weighted choice to determine bot type
        bot_types = ["spam", "misinformation", "astroturfing"]
        bot_weights = [0.5, 0.25, 0.25]  # 10:5:5 ratio

        # Use model's RNG for reproducibility with weighted choice
        self.bot_type = model.random.choices(
            bot_types,
            weights=bot_weights,
            k=1
        )[0]

        # Pass only model to super().__init__ in Mesa 3.1.4
        super().__init__(model=model, post_type=self.bot_type)
        self.agent_type = "bot"

        # Bot characteristics
        self.detection_rate = 0
        self.malicious_post_rate = model.random.uniform(0.05, 0.9)

        # Position in 2D topic grid instead of 5D vector
        # Initialize based on bot type and quadrant attractiveness
        self.initialize_topic_position(model)

        # Low topic mobility for bots (they tend to stay in their preferred topics)
        self.topic_mobility = model.random.uniform(0.02, 0.1)

        # Configure bot behaviors based on type
        self.configure_bot_type()

    def initialize_topic_position(self, model):
        """Initialize topic position based on bot type and quadrant attractiveness."""
        # Get quadrant attractiveness values from model or use defaults
        quadrant_attractiveness = getattr(model, 'bot_quadrant_attractiveness', {
            'tech_business': 0.56,  # Q1: Tech/Business
            'politics_news': 0.27,   # Q2: Politics/News
            'pop_culture': 0.19,     # Q4: Pop Culture
            'hobbies': 0.05          # Q3: Hobbies
        })

        # Adjust attractiveness based on bot type
        type_quadrant_bias = {
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

        # Calculate combined weights
        combined_weights = {}
        for quadrant in quadrant_attractiveness:
            combined_weights[quadrant] = (
                0.7 * quadrant_attractiveness[quadrant] +
                0.3 * type_quadrant_bias[self.bot_type][quadrant]
            )

        # Normalize weights
        weight_sum = sum(combined_weights.values())
        for key in combined_weights:
            combined_weights[key] /= weight_sum

        # Choose target quadrant based on weights
        chosen_quadrant = model.random.choices(
            list(combined_weights.keys()),
            weights=list(combined_weights.values()),
            k=1
        )[0]

        # Initialize position based on chosen quadrant
        if chosen_quadrant == 'tech_business':
            self.topic_position = {
                'x': model.random.uniform(0, 0.5),
                'y': model.random.uniform(0, 0.5)
            }
        elif chosen_quadrant == 'politics_news':
            self.topic_position = {
                'x': model.random.uniform(0, 0.5),
                'y': model.random.uniform(0.5, 1)
            }
        elif chosen_quadrant == 'hobbies':
            self.topic_position = {
                'x': model.random.uniform(0.5, 1),
                'y': model.random.uniform(0, 0.5)
            }
        else:  # pop_culture
            self.topic_position = {
                'x': model.random.uniform(0.5, 1),
                'y': model.random.uniform(0.5, 1)
            }

    def configure_bot_type(self):
        """Configure bot parameters based on type and quadrant."""
        current_quadrant = self.get_current_quadrant()

        # Base values
        base_post_frequency = 0
        base_detection_rate = 0

        if self.bot_type == "misinformation":
            base_post_frequency = self.model.random.uniform(0.2, 0.8)
            base_detection_rate = self.model.random.uniform(0.01, 0.03)
            # Misinformation bots are more active in politics/news
            if current_quadrant == 'politics_news':
                base_post_frequency *= 1.5
                base_detection_rate *= 0.8  # Harder to detect in their preferred quadrant

        elif self.bot_type == "spam":
            base_post_frequency = self.model.random.uniform(0.5, 0.99)
            base_detection_rate = self.model.random.uniform(0.01, 0.07)
            # Spam bots are more active in pop culture
            if current_quadrant == 'pop_culture':
                base_post_frequency *= 1.3
                base_detection_rate *= 0.9

        elif self.bot_type == "astroturfing":
            base_post_frequency = self.model.random.uniform(0.2, 0.8)
            base_detection_rate = self.model.random.uniform(0.01, 0.03)
            # Astroturfing bots are more active in tech/business
            if current_quadrant == 'tech_business':
                base_post_frequency *= 1.5
                base_detection_rate *= 0.8

        self.post_frequency = base_post_frequency
        self.detection_rate = base_detection_rate

    def get_current_quadrant(self):
        """Determine which quadrant the agent is currently in."""
        x, y = self.topic_position['x'], self.topic_position['y']

        if x < 0.5 and y < 0.5:
            return 'tech_business'  # Q1: Serious-Business (Tech/Business)
        elif x < 0.5 and y >= 0.5:
            return 'politics_news'  # Q2: Serious-Societal (Politics/News)
        elif x >= 0.5 and y < 0.5:
            return 'hobbies'        # Q3: Casual-Business (Hobbies)
        else:
            return 'pop_culture'    # Q4: Casual-Societal (Pop Culture)

    def step(self):
        """Bot agent behavior during each step."""
        super().step()

        if not self.active:
            return

        # Update bot parameters based on current quadrant
        self.configure_bot_type()

        # Post with some probability
        self.bot_post()

        # Check if bot gets banned
        self.check_ban()

        # Possibly shift topic position slightly (limited mobility)
        if self.model.random.random() < 0.05:  # 5% chance to shift
            self.shift_topic()

    def check_ban(self):
        """Check if the bot gets banned."""
        # Modify detection rate based on quadrant and time spent
        current_quadrant = self.get_current_quadrant()

        # Bots in hobbies quadrant are less likely to be detected (lower activity)
        detection_modifier = 1.0
        if current_quadrant == 'hobbies':
            detection_modifier = 0.8
        elif current_quadrant == 'politics_news':
            detection_modifier = 1.2  # More scrutiny in political areas

        # Use model's RNG for reproducibility
        if self.model.random.random() < (self.detection_rate * detection_modifier):
            self.deactivate()

    def bot_post(self):
        """Create a post with some probability."""
        # Use model's RNG
        if self.model.random.random() < self.post_frequency:
            self.posted_today = True

            # Quadrant-based post behavior
            current_quadrant = self.get_current_quadrant()

            # Malicious post rate varies by quadrant
            adjusted_malicious_rate = self.malicious_post_rate

            if self.bot_type == "misinformation" and current_quadrant == "politics_news":
                adjusted_malicious_rate *= 1.5
            elif self.bot_type == "astroturfing" and current_quadrant == "tech_business":
                adjusted_malicious_rate *= 1.5

            # Decide if post should be malicious
            if self.model.random.random() < adjusted_malicious_rate:
                self.create_malicious_post()
            else:
                self.attempt_normal_post()
        else:
            self.posted_today = False

    def create_malicious_post(self):
        """Create a malicious post based on bot type."""
        self.post_type = self.bot_type

    def attempt_normal_post(self):
        """Create a normal post to avoid detection."""
        self.post_type = "normal"

    def shift_topic(self):
        """Shift the bot's topic position in 2D space."""
        # Get quadrant attractiveness values from model or use defaults
        quadrant_attractiveness = getattr(self.model, 'bot_quadrant_attractiveness', {
            'tech_business': 0.56,
            'politics_news': 0.27,
            'pop_culture': 0.19,
            'hobbies': 0.05
        })

        # Bot type preferences (from initialize_topic_position)
        type_quadrant_bias = {
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

        # Calculate combined weights
        combined_weights = {}
        for quadrant in quadrant_attractiveness:
            combined_weights[quadrant] = (
                0.7 * quadrant_attractiveness[quadrant] +
                0.3 * type_quadrant_bias[self.bot_type][quadrant]
            )

        # Normalize weights
        weight_sum = sum(combined_weights.values())
        for key in combined_weights:
            combined_weights[key] /= weight_sum

        # Choose target quadrant based on weights
        target_quadrant = self.model.random.choices(
            list(combined_weights.keys()),
            weights=list(combined_weights.values()),
            k=1
        )[0]

        # Set target coordinates based on chosen quadrant
        target_x, target_y = 0.25, 0.25  # Default to Q1 center

        if target_quadrant == 'tech_business':
            target_x, target_y = 0.25, 0.25  # Q1 center
        elif target_quadrant == 'politics_news':
            target_x, target_y = 0.25, 0.75  # Q2 center
        elif target_quadrant == 'hobbies':
            target_x, target_y = 0.75, 0.25  # Q3 center
        elif target_quadrant == 'pop_culture':
            target_x, target_y = 0.75, 0.75  # Q4 center

        # Add small randomness to target
        target_x += self.model.random.uniform(-0.1, 0.1)
        target_y += self.model.random.uniform(-0.1, 0.1)

        # Ensure values stay within [0,1]
        target_x = max(0, min(1, target_x))
        target_y = max(0, min(1, target_y))

        # Move toward target (very gradual movement for bots)
        self.topic_position['x'] += self.topic_mobility * (target_x - self.topic_position['x'])
        self.topic_position['y'] += self.topic_mobility * (target_y - self.topic_position['y'])