"""
HumanAgent.py - Implementation of human agents for social media simulation
using Mesa 3.1.4
"""

from SocialMediaAgent import SocialMediaAgent
import numpy as np
from datetime import date, timedelta


class HumanAgent(SocialMediaAgent):
    """Human agent in the social media simulation."""

    def __init__(self, model):
        # Pass only model to super().__init__ in Mesa 3.1.4
        super().__init__(model=model, post_type="normal")
        self.agent_type = "human"

        # Get satisfaction from model if available, otherwise use default 100
        self.satisfaction = getattr(model, "human_satisfaction_init", 100)

        # Personality parameters that affect interactions - use model's RNG
        self.irritability = model.random.uniform(0.5, 2)  # More irritable users will lose satisfaction quicker
        self.authenticity = model.random.uniform(0.5, 2)  # Higher number indicates distaste for misinformation
        self.activeness = model.random.random()  # Determines exposure to posts

        # Position in the topic space (for network formation)
        # Use model's numpy RNG wrapper
        self.topic_position = model.np_random.normal(0, 0.5, size=5)  # 5-dimensional topic space

        self.post_frequency = model.random.uniform(0.1, 0.6)
        self.popularity = model.random.uniform(0.3, 0.95)

        # Initialize position in topic space
        self.move_in_topic_space()

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
        if self.satisfaction <= 0:
            self.deactivate()

    def human_post(self):
        """Create a post with some probability."""
        if self.should_post():
            self.last_post_date = self.get_current_date(self.model)
            self.posted_today = True
        else:
            self.posted_today = False

    def move_in_topic_space(self):
        """Move in topic space to simulate changing interests."""
        # Small random shift in topic position - use model's numpy RNG wrapper
        shift = self.model.np_random.normal(0, 0.1, size=5)  # 5-dimensional topic space
        self.topic_position = self.topic_position + shift

        # Normalize to stay in similar bounds
        norm = np.linalg.norm(self.topic_position)
        if norm > 2.0:  # Prevent too extreme positions
            self.topic_position = self.topic_position * (2.0 / norm)

    def react_to_posts(self):
        """React to posts from connected or nearby agents."""
        # Get all active connected agents
        connected_agents = []
        for agent_id in self.connections:
            agent = self.get_agent_by_id(agent_id)
            if agent and agent.active:
                connected_agents.append(agent)

        # ECHO CHAMBER MECHANISM DISABLED
        # Also consider nearby agents in topic space that aren't connected yet
        # nearby_agents = self.model.get_nearby_agents(self)

        # ECHO CHAMBER MECHANISM DISABLED
        # Combine the lists, prioritizing connections
        # potential_interactions = connected_agents + [a for a in nearby_agents if a.unique_id not in self.connections]

        # Only consider connected agents for interactions
        potential_interactions = connected_agents

        # Filter out those who didn't post
        potential_interactions = [a for a in potential_interactions if getattr(a, "posted_today", False)]

        # Randomly select a subset to interact with based on activeness
        num_interactions = max(1, int(len(potential_interactions) * self.activeness))
        if potential_interactions and num_interactions > 0:
            # Use model's random for reproducibility
            interaction_targets = self.model.random.sample(
                potential_interactions,
                min(num_interactions, len(potential_interactions))
            )

            # React to each target's post
            for target in interaction_targets:
                self.react_to_post(target)

    def react_to_post(self, other_agent):
        """React to a post from another agent."""
        # Base satisfaction change
        satisfaction_change = 0

        # Human-to-human interaction
        if other_agent.agent_type == "human":
            # Get positive bias value from model or use default
            positive_bias = getattr(self.model, "human_human_positive_bias", 0.7)

            # Adjust the random range based on the bias (higher bias = more positive interactions)
            base_change = self.model.random.uniform(
                -0.5 * (1 - positive_bias),  # Lower bound becomes less negative with higher bias
                2.0 * positive_bias          # Upper bound becomes more positive with higher bias
            )

            # Topic similarity increases positive satisfaction
            similarity = self.model.calculate_topic_similarity(self, other_agent)
            topic_effect = (similarity * 2) - 0.5  # Range from -0.5 to 1.5

            satisfaction_change = base_change + topic_effect

        elif other_agent.agent_type == "bot":
            # Get negative bias value from model or use default
            negative_bias = getattr(self.model, "human_bot_negative_bias", 0.8)

            if other_agent.post_type == "normal":
                satisfaction_change = self.model.random.uniform(
                    0.5 * negative_bias,  # More negative
                    0.2 * (1 - negative_bias)  # Less positive
                )
            elif other_agent.post_type == "misinformation":
                # Increased negative impact
                satisfaction_change = -10 * self.authenticity * self.irritability * negative_bias
            elif other_agent.post_type == "astroturfing":
                # Increased negative impact
                satisfaction_change = -10 * self.authenticity * self.irritability * negative_bias
            elif other_agent.post_type == "spam":
                # Increased negative impact
                satisfaction_change = 1-0 * self.irritability * negative_bias

        # Apply satisfaction change
        self.satisfaction += satisfaction_change

        # Cap satisfaction between 0 and 100
        self.satisfaction = max(0, min(100, self.satisfaction))

        # Update connection probability based on interaction
        self.update_connection_probability(other_agent, satisfaction_change)

    def update_connection_probability(self, other_agent, satisfaction_change):
        """Update probability of maintaining connection based on interaction."""
        # For human-to-human
        if other_agent.agent_type == "human":
            if satisfaction_change < -2:  # Significant negative interaction
                if other_agent.unique_id in self.connections:
                    # Chance to break connection after negative interaction
                    if self.model.random.random() < 0.05:  # Small chance (5%)
                        self.remove_connection(other_agent)

        # For human-to-bot, negative interactions might lead to blocking
        elif other_agent.agent_type == "bot":
            if satisfaction_change < -1.5:
                # Very negative interactions with bots might lead to blocking
                if other_agent.unique_id in self.connections:
                    # Get negative bias value from model or use default
                    negative_bias = getattr(self.model, "human_bot_negative_bias", 0.8)
                    # Chance to block increases with negative bias
                    block_chance = 0.3 * negative_bias
                    # Use model's RNG for reproducibility
                    if self.model.random.random() < block_chance:
                        self.remove_connection(other_agent)