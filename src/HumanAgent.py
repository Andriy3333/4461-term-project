"""
HumanAgent.py - Implementation of human agents for social media simulation
using Mesa 3.1.4 with 2D topic grid
"""
import constants
from SocialMediaAgent import SocialMediaAgent
import numpy as np
from datetime import date, timedelta


class HumanAgent(SocialMediaAgent):
    """Human agent in the social media simulation with 2D topic space."""

    def __init__(self, model):
        # Pass only model to super().__init__ in Mesa 3.1.4
        super().__init__(model=model, post_type="normal")
        self.agent_type = "human"

        # Determine if this is a super-user (10% chance)
        self.is_super_user = model.random.random() < constants.DEFAULT_SUPER_USER_PROBABILITY

        # Super users get special movement properties
        if self.is_super_user:
            self.topic_mobility = model.random.uniform(*constants.DEFAULT_SUPER_USER_TOPIC_MOBILITY_RANGE)
            self.influence_radius = model.random.uniform(*constants.DEFAULT_SUPER_USER_INFLUENCE_RADIUS_RANGE)
            self.topic_leadership = model.random.uniform(*constants.DEFAULT_SUPER_USER_TOPIC_LEADERSHIP_RANGE)
        else:
            self.topic_mobility = model.random.uniform(*constants.DEFAULT_REGULAR_USER_TOPIC_MOBILITY_RANGE)
            self.influence_radius = 0
            self.topic_leadership = 0

        # Get satisfaction from model if available, otherwise use default 100
        self.satisfaction = getattr(model, "human_satisfaction_init", 100)

        # Super-users are more resistant to leaving
        self.satisfaction_threshold = (constants.DEFAULT_REGULAR_USER_SATISFACTION_THRESHOLD if not self.is_super_user
                                       else constants.DEFAULT_SUPER_USER_SATISFACTION_THRESHOLD)

        # Personality parameters
        self.irritability = model.random.uniform(*constants.DEFAULT_HUMAN_IRRITABILITY_RANGE)
        self.authenticity = model.random.uniform(*constants.DEFAULT_HUMAN_AUTHENTICITY_RANGE)

        # Activity parameters
        self.base_activeness = model.random.uniform(*constants.DEFAULT_HUMAN_BASE_ACTIVENESS_RANGE)
        self.activeness = (self.base_activeness * constants.DEFAULT_SUPER_USER_ACTIVITY_MULTIPLIER
                           if self.is_super_user else self.base_activeness)

        # Position in 2D topic grid instead of 5D vector
        # x-axis: 0 (Serious) to 1 (Casual)
        # y-axis: 0 (Business) to 1 (Societal)
        self.topic_position = {
            'x': model.random.uniform(0, 1),
            'y': model.random.uniform(0, 1)
        }

        # Topic interests (quadrant preferences)
        # Quadrants: Q1 (Tech/Business), Q2 (Politics/News), Q3 (Hobbies), Q4 (Pop Culture)
        # These values affect how likely the agent is to move toward each quadrant
        self.quadrant_preferences = {
            'tech_business': model.random.uniform(0, 1),  # Q1: Tech/Business
            'politics_news': model.random.uniform(0, 1),  # Q2: Politics/News
            'hobbies': model.random.uniform(0, 1),        # Q3: Hobbies
            'pop_culture': model.random.uniform(0, 1)     # Q4: Pop Culture
        }

        # Normalize preferences to sum to 1
        pref_sum = sum(self.quadrant_preferences.values())
        for key in self.quadrant_preferences:
            self.quadrant_preferences[key] /= pref_sum

        # Post frequency
        self.base_post_frequency = model.random.uniform(*constants.DEFAULT_HUMAN_BASE_POST_FREQUENCY_RANGE)
        self.post_frequency = (self.base_post_frequency * constants.DEFAULT_SUPER_USER_ACTIVITY_MULTIPLIER
                               if self.is_super_user else self.base_post_frequency)
        self.popularity = model.random.uniform(*constants.DEFAULT_HUMAN_POPULARITY_RANGE)

    def step(self):
        """Human agent behavior during each step."""
        super().step()

        if not self.active:
            return

        # Move in topic space
        self.move_in_topic_space()

        # Post with some probability
        self.human_post()

        # React to posts from connected agents
        self.react_to_posts()

        # Check if satisfaction is too low to continue
        if self.satisfaction <= self.satisfaction_threshold:
            self.deactivate()

    def human_post(self):
        """Create a post with some probability."""
        if self.should_post():
            self.last_post_date = self.get_current_date(self.model)
            self.posted_today = True
        else:
            self.posted_today = False

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

    def move_in_topic_space(self):
        """Move in 2D topic space with persistent target selection."""

        # Initialize persistent properties if they don't exist
        if not hasattr(self, 'target_commitment'):
            self.target_commitment = 0
            self.current_target_quadrant = None
            self.target_x = None
            self.target_y = None

        # Check if we need to select a new target
        if self.target_commitment <= 0 or self.current_target_quadrant is None:
            # Get quadrant attractiveness values from model
            quadrant_attractiveness = getattr(self.model, 'human_quadrant_attractiveness', {
                'tech_business': 0.15,
                'politics_news': 0.20,
                'hobbies': 0.34,
                'pop_culture': 0.50
            })

            # Calculate movement weights based on preferences and attractiveness
            movement_weights = {}
            for quadrant in self.quadrant_preferences:
                movement_weights[quadrant] = (self.quadrant_preferences[quadrant])

            # Normalize weights
            weight_sum = sum(movement_weights.values())
            for key in movement_weights:
                movement_weights[key] /= weight_sum

            # Choose target quadrant based on weights
            self.current_target_quadrant = self.model.random.choices(
                list(movement_weights.keys()),
                weights=list(movement_weights.values()),
                k=1
            )[0]

            # Set target coordinates based on chosen quadrant
            if self.current_target_quadrant == 'tech_business':
                self.target_x, self.target_y = 0.25, 0.25  # Q1 center
            elif self.current_target_quadrant == 'politics_news':
                self.target_x, self.target_y = 0.25, 0.75  # Q2 center
            elif self.current_target_quadrant == 'hobbies':
                self.target_x, self.target_y = 0.75, 0.25  # Q3 center
            elif self.current_target_quadrant == 'pop_culture':
                self.target_x, self.target_y = 0.75, 0.75  # Q4 center

            # Add randomness to target
            self.target_x += self.model.random.uniform(-0.15, 0.15)
            self.target_y += self.model.random.uniform(-0.15, 0.15)

            # Ensure values stay within [0,1]
            self.target_x = max(0, min(1, self.target_x))
            self.target_y = max(0, min(1, self.target_y))

            # Set commitment period (5-15 steps)
            self.target_commitment = self.model.random.randint(5, 15)

            # Super users have longer commitment periods
            if hasattr(self, 'is_super_user') and self.is_super_user:
                self.target_commitment *= 2

        # Decrement commitment counter
        self.target_commitment -= 1

        # Move toward target (gradual movement)
        movement_speed = getattr(self, 'topic_mobility', 0.1)
        self.topic_position['x'] += movement_speed * (self.target_x - self.topic_position['x'])
        self.topic_position['y'] += movement_speed * (self.target_y - self.topic_position['y'])

    def react_to_posts(self):
        """React to posts with zero-sum allocation between forced and follower feeds."""
        if not self.active:
            return

        if self.model.steps < 0:
            return

        # Get forced feed ratio from constants
        forced_feed_ratio = constants.DEFAULT_FORCED_FEED_PROBABILITY

        # Determine base number of posts to see (based on user's activeness)
        base_post_count = max(3, int(10 * self.activeness))

        # Split post count between sources based on ratio
        forced_feed_count = int(base_post_count * forced_feed_ratio)
        follower_feed_count = base_post_count - forced_feed_count

        # PART 1: Follower feed content
        # Get all active connected agents who posted
        posting_connected_agents = []
        for agent_id in self.connections:
            agent = self.get_agent_by_id(agent_id)
            if agent and agent.active and getattr(agent, "posted_today", False):
                posting_connected_agents.append(agent)

        # Process follower feed interactions
        if posting_connected_agents and follower_feed_count > 0:
            # Limit to available posts
            follower_count = min(follower_feed_count, len(posting_connected_agents))

            # Select random posts from connections
            follower_targets = self.model.random.sample(
                posting_connected_agents,
                follower_count
            )

            # React to each post
            for target in follower_targets:
                self.react_to_post(target)

        # PART 2: Forced feed content
        if forced_feed_count > 0:
            # Get all active bots that posted today (excluding connections)
            active_posting_bots = [agent for agent in self.model.agents
                                   if agent.active and
                                   agent.agent_type == "bot" and
                                   getattr(agent, "posted_today", False) and
                                   agent.unique_id not in self.connections]

            # Process forced feed if bots are available
            if active_posting_bots:
                # Limit to available bots
                bot_count = min(forced_feed_count, len(active_posting_bots))

                # Select random bots to force-expose
                forced_bots = self.model.random.sample(active_posting_bots, bot_count)

                # React to each forced bot post
                for bot in forced_bots:
                    self.react_to_post(bot)

    def react_to_post(self, other_agent):
        """React to a post from another agent with quadrant factors."""
        # Base satisfaction change
        satisfaction_change = 0

        # Get current quadrant
        current_quadrant = self.get_current_quadrant()

        # Human-to-human interaction
        if other_agent.agent_type == "human":
            # Get positive bias value from model or use default
            positive_bias = getattr(self.model, "human_human_positive_bias", 0.7)

            # Adjust the random range based on the bias
            base_change = self.model.random.uniform(
                -0.5 * (1 - positive_bias),
                2.0 * positive_bias
            )

            # Topic similarity increases positive satisfaction
            similarity = self.model.calculate_topic_similarity(self, other_agent)
            topic_effect = (similarity * 2) - 0.5  # Range from -0.5 to 1.5

            # Super users have more influence on satisfaction
            if hasattr(other_agent, 'is_super_user') and other_agent.is_super_user:
                topic_effect *= 1.5

            satisfaction_change = base_change + topic_effect

        elif other_agent.agent_type == "bot":
            # Get negative bias value from model or use default
            negative_bias = getattr(self.model, "human_bot_negative_bias", 0.8)

            # Get bot's current quadrant
            bot_quadrant = other_agent.get_current_quadrant()

            # Quadrant-specific modifiers
            # Bots in politics/news and tech/business have more impact
            quadrant_modifier = 1.0
            if bot_quadrant == 'politics_news':
                quadrant_modifier = 1.5
            elif bot_quadrant == 'tech_business':
                quadrant_modifier = 1.3

            # Different effect based on post type
            if other_agent.post_type == "normal":
                satisfaction_change = self.model.random.uniform(
                    -0.5 * negative_bias * quadrant_modifier,
                    0.2 * (1 - negative_bias)
                )
            elif other_agent.post_type == "misinformation":
                # Increased negative impact
                satisfaction_change = -1.0 * self.authenticity * self.irritability * negative_bias * quadrant_modifier
            elif other_agent.post_type == "astroturfing":
                # Increased negative impact
                satisfaction_change = -1.0 * self.authenticity * self.irritability * negative_bias * quadrant_modifier
            elif other_agent.post_type == "spam":
                # Less negative impact
                satisfaction_change = -0.5 * self.irritability * negative_bias * quadrant_modifier

        # Apply satisfaction change
        self.satisfaction += satisfaction_change * 10  # Scale up for more noticeable effects

        # Cap satisfaction between 0 and 100
        self.satisfaction = max(0, min(100, self.satisfaction))

        # Update connection probability based on interaction
        self.update_connection_probability(other_agent, satisfaction_change)

    def update_connection_probability(self, other_agent, satisfaction_change):
        """Update probability of maintaining connection based on interaction."""
        if other_agent.agent_type == "human" and getattr(other_agent, "is_super_user", False):
            # Higher chance to maintain connections with super users despite negative interactions
            if satisfaction_change < constants.DEFAULT_BOT_BLOCK_SATISFACTION_THRESHOLD: # Significant negative interaction
                if other_agent.unique_id in self.connections:
                    # Reduced chance to break connection (3% vs 5%)
                    if self.model.random.random() < constants.DEFAULT_SUPER_USER_CONNECTION_BREAKING_PROB:
                        self.remove_connection(other_agent)
        # For human-to-human
        if other_agent.agent_type == "human":
            if satisfaction_change < constants.DEFAULT_BOT_BLOCK_SATISFACTION_THRESHOLD:  # Significant negative interaction
                if other_agent.unique_id in self.connections:
                    # Chance to break connection after negative interaction
                    if self.model.random.random() < 0.05:  # Small chance (5%)
                        self.remove_connection(other_agent)

        # For human-to-bot, negative interactions might lead to blocking
        elif other_agent.agent_type == "bot":
            if satisfaction_change < -0.15:
                # Very negative interactions with bots might lead to blocking
                if other_agent.unique_id in self.connections:
                    # Get negative bias value from model or use default
                    negative_bias = getattr(self.model, "human_bot_negative_bias", 0.8)
                    # Chance to block increases with negative bias
                    block_chance = 0.3 * negative_bias
                    # Super users are less likely to block
                    if self.is_super_user:
                        block_chance *= 0.5
                    # Use model's RNG for reproducibility
                    if self.model.random.random() < block_chance:
                        self.remove_connection(other_agent)