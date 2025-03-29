"""
BotAgent.py - Implementation of bot agents for social media simulation
using Mesa 3.1.4 with 2D topic grid
"""

from SocialMediaAgent import SocialMediaAgent
import numpy as np
from datetime import date, timedelta

import constants

class BotAgent(SocialMediaAgent):
    """Bot agent in the social media simulation with 2D topic space."""

    def __init__(self, model, quadrant=None):
        # Use quadrant to determine bot type if available
        bot_types = ["spam", "misinformation", "astroturfing"]

        # Default weights if no quadrant or no quadrant-specific weights
        bot_weights = [0.5, 0.25, 0.25]  # 10:5:5 ratio

        # If a quadrant is specified and the model has quadrant-specific bot type weights
        if quadrant and hasattr(model, 'bot_type_by_quadrant'):
            # Get quadrant-specific bot type weights
            quadrant_weights = model.bot_type_by_quadrant.get(quadrant, {})
            if quadrant_weights:
                # Update bot types and weights based on quadrant-specific distribution
                bot_types = list(quadrant_weights.keys())
                bot_weights = list(quadrant_weights.values())

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
        self.malicious_post_rate = model.random.uniform(*constants.DEFAULT_BOT_MALICIOUS_POST_RATE_RANGE)
        # Position in 2D topic grid instead of 5D vector
        # Initialize based on bot type and quadrant attractiveness
        self.initialize_topic_position(model)

        # Low topic mobility for bots (they tend to stay in their preferred topics)
        self.topic_mobility = model.random.uniform(*constants.DEFAULT_BOT_TOPIC_MOBILITY_RANGE)

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
        type_quadrant_bias = constants.DEFAULT_BOT_TYPE_QUADRANT_BIAS

        # Calculate combined weights
        combined_weights = {}
        for quadrant in quadrant_attractiveness:
            combined_weights[quadrant] = (
                    constants.DEFAULT_QUADRANT_ATTRACTIVENESS_WEIGHT * quadrant_attractiveness[quadrant] +
                    constants.DEFAULT_TYPE_QUADRANT_BIAS_WEIGHT * type_quadrant_bias[self.bot_type][quadrant]
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
        """Create a post with some probability and actively attempt to spread it."""
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

            # ENHANCEMENT: Actively push posts to connected humans
            # This creates a more direct impact mechanism
            self.spread_posts_to_connections()
        else:
            self.posted_today = False

    def spread_posts_to_connections(self):
        """Actively spread posts to human connections to maximize impact."""
        # Only proceed if the bot has posted today
        if not self.posted_today:
            return

        # Get all human connections
        human_connections = []
        for conn_id in self.connections:
            other = self.get_agent_by_id(conn_id)
            if other and other.active and other.agent_type == "human":
                human_connections.append(other)

        # Calculate base impact based on bot type and post type
        base_impact = -1.0  # Default negative impact

        if self.post_type == "normal":
            base_impact = -0.2  # Minimal negative impact for normal posts
        elif self.post_type == "misinformation":
            base_impact = -2.0  # Significant negative impact
        elif self.post_type == "astroturfing":
            base_impact = -1.5  # Moderate-high negative impact
        elif self.post_type == "spam":
            base_impact = -0.8  # Moderate negative impact

        # For each human connection, attempt to affect their satisfaction
        for human in human_connections:
            # Scale impact based on human properties
            scaled_impact = base_impact

            # More authentic humans are less affected
            if hasattr(human, 'authenticity'):
                scaled_impact *= (2.0 - human.authenticity)

            # More irritable humans are more affected
            if hasattr(human, 'irritability'):
                scaled_impact *= human.irritability

            # Super users are more resistant
            if hasattr(human, 'is_super_user') and human.is_super_user:
                scaled_impact *= 0.7

            # Apply the satisfaction impact with some randomness
            # This simulates the human seeing and reacting to the post
            if self.model.random.random() < 0.8:  # 80% chance of seeing the post
                # Add random variance to impact
                variance = self.model.random.uniform(0.8, 1.2)
                final_impact = scaled_impact * variance

                # Update human satisfaction
                if hasattr(human, 'satisfaction'):
                    human.satisfaction += final_impact

                    # Cap satisfaction between 0 and 100
                    human.satisfaction = max(0, min(100, human.satisfaction))

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

    """
    BotAgent.py - Add connection maintenance to BotAgent class
    """

    def maintain_connections(self):
        """
        Aggressively maintain a target number of connections with both bots and humans.
        This method is called during each bot's step to ensure connection targets are met.
        """
        if not self.active:
            return

        # Target number of connections for each type - make this explicit and guaranteed
        target_bot_connections = 5  # Fixed target
        target_human_connections = 5  # Fixed target

        # Count current connections by type
        bot_connections = []
        human_connections = []

        # Analyze existing connections
        for conn_id in self.connections:
            other = self.get_agent_by_id(conn_id)
            if other and other.active:
                if other.agent_type == "bot":
                    bot_connections.append(other)
                elif other.agent_type == "human":
                    human_connections.append(other)

        current_bot_connections = len(bot_connections)
        current_human_connections = len(human_connections)

        # PART 1: Aggressively maintain bot-to-bot connections
        if current_bot_connections < target_bot_connections:
            # Get all possible bot candidates
            all_bots = [agent for agent in self.model.agents
                        if agent.active and
                        agent.agent_type == "bot" and
                        agent.unique_id != self.unique_id and
                        agent.unique_id not in self.connections]

            # Prioritize bots in the same quadrant first
            current_quadrant = self.get_current_quadrant()

            # Separate bots by quadrant match
            same_quadrant_bots = [b for b in all_bots if b.get_current_quadrant() == current_quadrant]
            other_quadrant_bots = [b for b in all_bots if b.get_current_quadrant() != current_quadrant]

            # Combine lists with priority (same quadrant first)
            potential_bots = same_quadrant_bots + other_quadrant_bots

            # If we have bot candidates, add connections until target is reached
            num_to_add = target_bot_connections - current_bot_connections

            for i in range(min(num_to_add, len(potential_bots))):
                bot = potential_bots[i]

                # Make direct connection
                self.add_connection(bot)
                bot.add_connection(self)  # Always make reciprocal for bot-bot
                current_bot_connections += 1

                # Print debug message if desired
                # print(f"Bot {self.unique_id} connected to bot {bot.unique_id}")

        # PART 2: Aggressively maintain bot-to-human connections
        if current_human_connections < target_human_connections:
            # Get all potential human candidates
            all_humans = [agent for agent in self.model.agents
                          if agent.active and
                          agent.agent_type == "human" and
                          agent.unique_id not in self.connections]

            # Prioritize by quadrant preference based on bot type
            current_quadrant = self.get_current_quadrant()
            scored_humans = []

            for human in all_humans:
                human_quadrant = human.get_current_quadrant()
                score = 0

                # Base score
                score = 1

                # Quadrant matching based on bot type
                if self.bot_type == "spam" and human_quadrant in ["pop_culture", "hobbies"]:
                    score += 5
                elif self.bot_type == "misinformation" and human_quadrant == "politics_news":
                    score += 5
                elif self.bot_type == "astroturfing" and human_quadrant == "tech_business":
                    score += 5

                # Super users are high-value targets
                if getattr(human, 'is_super_user', False):
                    score += 10

                # Add to scored list
                scored_humans.append((human, score))

            # Sort by score (highest first)
            scored_humans.sort(key=lambda x: x[1], reverse=True)

            # Get top candidates
            num_to_add = target_human_connections - current_human_connections
            top_humans = [h[0] for h in scored_humans[:num_to_add]]

            # Add connections to selected humans
            for human in top_humans:
                self.add_connection(human)

                # Sometimes make humans connect back (with higher probability for super users)
                if getattr(human, 'is_super_user', False):
                    if self.model.random.random() < 0.7:  # 70% for super users
                        human.add_connection(self)
                else:
                    if self.model.random.random() < 0.4:  # 40% for regular users
                        human.add_connection(self)

                current_human_connections += 1

                # Print debug message if desired
                # print(f"Bot {self.unique_id} connected to human {human.unique_id}")

    def step(self):
        """Bot agent behavior during each step."""
        super().step()

        if not self.active:
            return

        # Update bot parameters based on current quadrant
        self.configure_bot_type()

        # Maintain target connections with both bots and humans
        self.maintain_connections()

        # Post with some probability
        self.bot_post()

        # Check if bot gets banned
        self.check_ban()

        # Possibly shift topic position slightly (limited mobility)
        if self.model.random.random() < 0.05:  # 5% chance to shift
            self.shift_topic()