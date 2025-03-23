"""
model.py - Implementation of a social media simulation model with quadrant-based topic space
using Mesa 3.1.4 with proximity-based connections
"""

import mesa
import numpy as np
import networkx as nx
from datetime import date, timedelta
from mesa.datacollection import DataCollector

from HumanAgent import HumanAgent
from BotAgent import BotAgent
import constants


class QuadrantTopicModel(mesa.Model):
    """Social media simulation model with quadrant-based topic space."""

    def __init__(
            self,
            # Initial population parameters
            num_initial_humans=constants.DEFAULT_INITIAL_HUMANS,
            num_initial_bots=constants.DEFAULT_INITIAL_BOTS,

            # Growth rates
            human_creation_rate=constants.DEFAULT_HUMAN_CREATION_RATE,
            bot_creation_rate=constants.DEFAULT_BOT_CREATION_RATE,

            # Ban rates
            bot_ban_rate_multiplier=constants.DEFAULT_BOT_BAN_RATE_MULTIPLIER,

            # Network parameters
            network_stability=constants.DEFAULT_NETWORK_STABILITY,

            # Topic parameters
            topic_shift_frequency=constants.DEFAULT_TOPIC_SHIFT_FREQUENCY,

            # Interaction parameters
            human_human_positive_bias=constants.DEFAULT_HUMAN_HUMAN_POSITIVE_BIAS,
            human_bot_negative_bias=constants.DEFAULT_HUMAN_BOT_NEGATIVE_BIAS,
            human_satisfaction_init=constants.DEFAULT_HUMAN_SATISFACTION_INIT,

            # Seed for reproducibility
            seed=None
    ):
        # Initialize the model with seed
        super().__init__(seed=seed)

        # Create a numpy random generator with the same seed
        self.np_random = np.random.RandomState(seed)

        # Store initial parameters
        self.num_initial_humans = num_initial_humans
        self.num_initial_bots = num_initial_bots
        self.human_creation_rate = human_creation_rate
        self.bot_creation_rate = bot_creation_rate
        self.bot_ban_rate_multiplier = bot_ban_rate_multiplier
        self.network_stability = network_stability
        self.topic_shift_frequency = topic_shift_frequency

        # Interaction parameters
        self.human_human_positive_bias = human_human_positive_bias
        self.human_bot_negative_bias = human_bot_negative_bias
        self.human_satisfaction_init = human_satisfaction_init

        # Quadrant attractiveness values from constants
        self.human_quadrant_attractiveness = constants.DEFAULT_HUMAN_QUADRANT_ATTRACTIVENESS
        self.bot_quadrant_attractiveness = constants.DEFAULT_BOT_QUADRANT_ATTRACTIVENESS

        # Initialize counters and trackers
        self.active_humans = 0
        self.active_bots = 0
        self.deactivated_humans = 0
        self.deactivated_bots = 0

        # Initialize the topic space using cell space
        self.initialize_topic_space()

        # Create initial agents
        self.create_initial_agents()

        # Create initial connections based on proximity
        self.create_initial_connections()

        # Initialize data collector
        self.initialize_data_collector()

    def initialize_topic_space(self):
        """
        Initialize the 2D topic space with custom implementation.

        The topic space is divided into four quadrants with these axes:
        - X-axis: Serious (0) to Casual (1)
        - Y-axis: Individual (0) to Societal (1)

        Resulting in these quadrants:
        - Q1 (0,0 to 0.5,0.5): Tech/Business (Serious & Individual)
        - Q2 (0,0.5 to 0.5,1): Politics/News (Serious & Societal)
        - Q3 (0.5,0 to 1,0.5): Hobbies (Casual & Individual)
        - Q4 (0.5,0.5 to 1,1): Pop Culture (Casual & Societal)
        """
        # Create a grid mapping for agent positions
        # This will be a dictionary where keys are (x, y) tuples (grid cells)
        # and values are lists of agents in that cell
        self.topic_grid = {}

        # Define grid resolution (100x100)
        self.grid_resolution = 100

        # Define the quadrants for quick reference
        self.quadrants = {
            'tech_business': (0, 0, 49, 49),  # x_min, y_min, x_max, y_max for Q1
            'politics_news': (0, 50, 49, 99),  # Q2
            'hobbies': (50, 0, 99, 49),  # Q3
            'pop_culture': (50, 50, 99, 99)  # Q4
        }

    def place_agent_in_topic_space(self, agent):
        """Place an agent in the topic grid based on its position."""
        # Calculate grid cell from agent's topic position
        x_cell = int(agent.topic_position['x'] * (self.grid_resolution - 1))
        y_cell = int(agent.topic_position['y'] * (self.grid_resolution - 1))

        # Get the current agents in that cell or create empty list
        cell_agents = self.topic_grid.get((x_cell, y_cell), [])

        # Add agent to the cell
        cell_agents.append(agent)

        # Update the grid
        self.topic_grid[(x_cell, y_cell)] = cell_agents

        # Store the agent's grid position for quick access
        agent.grid_pos = (x_cell, y_cell)

    def move_agent_in_topic_space(self, agent):
        """Move an agent in the topic grid based on its updated position."""
        # Calculate new grid cell
        x_cell = int(agent.topic_position['x'] * (self.grid_resolution - 1))
        y_cell = int(agent.topic_position['y'] * (self.grid_resolution - 1))

        # If position hasn't changed, do nothing
        if hasattr(agent, 'grid_pos') and agent.grid_pos == (x_cell, y_cell):
            return

        # Otherwise, remove from old position if it exists
        if hasattr(agent, 'grid_pos'):
            old_pos = agent.grid_pos
            if old_pos in self.topic_grid:
                self.topic_grid[old_pos].remove(agent)
                # Clean up empty cells
                if not self.topic_grid[old_pos]:
                    del self.topic_grid[old_pos]

        # Add to new position
        cell_agents = self.topic_grid.get((x_cell, y_cell), [])
        cell_agents.append(agent)
        self.topic_grid[(x_cell, y_cell)] = cell_agents

        # Update the agent's stored position
        agent.grid_pos = (x_cell, y_cell)

    def get_agents_in_radius(self, center_pos, radius):
        """Get all agents within a grid radius of the center position."""
        agents = []
        x_center, y_center = center_pos

        # Search all cells within the square radius
        for x in range(max(0, x_center - radius), min(self.grid_resolution, x_center + radius + 1)):
            for y in range(max(0, y_center - radius), min(self.grid_resolution, y_center + radius + 1)):
                # Check if cell exists in grid
                if (x, y) in self.topic_grid:
                    # Add all agents in the cell
                    agents.extend(self.topic_grid[(x, y)])

        return agents

    def create_initial_agents(self):
        """Create initial human and bot agents with proper quadrant distribution."""
        # Human quadrant target distribution from constants
        human_dist = self.human_quadrant_attractiveness

        # Calculate number of humans per quadrant based on distribution percentages
        humans_per_quadrant = {
            'tech_business': int(self.num_initial_humans * human_dist['tech_business']),
            'politics_news': int(self.num_initial_humans * human_dist['politics_news']),
            'hobbies': int(self.num_initial_humans * human_dist['hobbies']),
            'pop_culture': int(self.num_initial_humans * human_dist['pop_culture'])
        }

        # Ensure we create exactly num_initial_humans by adding any rounding remainder
        remainder = self.num_initial_humans - sum(humans_per_quadrant.values())
        if remainder > 0:
            # Add remainder to the quadrant with highest percentage
            max_quadrant = max(human_dist, key=human_dist.get)
            humans_per_quadrant[max_quadrant] += remainder

        # Bot quadrant target distribution from constants
        bot_dist = self.bot_quadrant_attractiveness

        # Calculate number of bots per quadrant based on distribution percentages
        bots_per_quadrant = {
            'tech_business': int(self.num_initial_bots * bot_dist['tech_business']),
            'politics_news': int(self.num_initial_bots * bot_dist['politics_news']),
            'hobbies': int(self.num_initial_bots * bot_dist['hobbies']),
            'pop_culture': int(self.num_initial_bots * bot_dist['pop_culture'])
        }

        # Ensure we create exactly num_initial_bots by adding any rounding remainder
        remainder = self.num_initial_bots - sum(bots_per_quadrant.values())
        if remainder > 0:
            # Add remainder to the quadrant with highest percentage
            max_quadrant = max(bot_dist, key=bot_dist.get)
            bots_per_quadrant[max_quadrant] += remainder

        # Create humans according to distribution
        for quadrant, count in humans_per_quadrant.items():
            for i in range(count):
                agent = HumanAgent(model=self)
                self.active_humans += 1

                # Set topic position based on quadrant
                self.set_agent_position_in_quadrant(agent, quadrant)

                # Place agent in topic space based on its position
                self.place_agent_in_topic_space(agent)

        # Create bots according to distribution
        for quadrant, count in bots_per_quadrant.items():
            for i in range(count):
                agent = BotAgent(model=self)
                self.active_bots += 1

                # Set topic position based on quadrant
                self.set_agent_position_in_quadrant(agent, quadrant)

                # Place agent in topic space based on its position
                self.place_agent_in_topic_space(agent)

    def set_agent_position_in_quadrant(self, agent, quadrant):
        """Set agent position to be within the specified quadrant with some randomness."""
        # Define quadrant boundaries (min_x, min_y, max_x, max_y)
        quadrant_bounds = {
            'tech_business': (0.0, 0.0, 0.5, 0.5),  # Q1
            'politics_news': (0.0, 0.5, 0.5, 1.0),  # Q2
            'hobbies': (0.5, 0.0, 1.0, 0.5),  # Q3
            'pop_culture': (0.5, 0.5, 1.0, 1.0)  # Q4
        }

        # Get bounds for the specified quadrant
        min_x, min_y, max_x, max_y = quadrant_bounds[quadrant]

        # Set position with some randomness but stay within quadrant
        # Use a Beta distribution to slightly favor positions away from boundaries
        alpha = beta = 2.0  # Parameters for Beta distribution (peaked in middle)

        # Generate positions using beta distribution and scale to quadrant
        x = min_x + (max_x - min_x) * self.random.betavariate(alpha, beta)
        y = min_y + (max_y - min_y) * self.random.betavariate(alpha, beta)

        # Update agent's position
        agent.topic_position = {'x': x, 'y': y}

    def apply_super_user_gravity(self, agent):
        """Apply attraction effect of super users on regular agents."""
        # Skip if agent is a super user - safely check if attribute exists
        if getattr(agent, "is_super_user", False):
            return  # Super users aren't affected by other super users

        # Find nearby super users
        super_users = []
        for other in self.agents:
            if (other.active and
                    getattr(other, "agent_type", "") == "human" and
                    getattr(other, "is_super_user", False)):

                # Calculate distance
                x1, y1 = agent.topic_position['x'], agent.topic_position['y']
                x2, y2 = other.topic_position['x'], other.topic_position['y']
                distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

                # Check if within influence radius
                if distance <= other.influence_radius:
                    super_users.append((other, distance))

        # If no super users in range, return
        if not super_users:
            return

        # Calculate weighted influence of super users
        total_pull_x = 0
        total_pull_y = 0
        total_weight = 0

        for super_user, distance in super_users:
            # Inverse distance weighting
            weight = (1 / max(0.1, distance)) * super_user.topic_leadership

            # Direction vector toward super user
            dir_x = super_user.topic_position['x'] - agent.topic_position['x']
            dir_y = super_user.topic_position['y'] - agent.topic_position['y']

            # Add weighted pull
            total_pull_x += dir_x * weight
            total_pull_y += dir_y * weight
            total_weight += weight

        # Apply the gravitational effect if there's any pull
        if total_weight > 0:
            # Normalize pull
            pull_x = total_pull_x / total_weight
            pull_y = total_pull_y / total_weight

            # Determine pull strength based on agent type
            pull_strength = 0.02  # Base pull strength

            # Bots have type-specific attraction to super users
            if agent.agent_type == "bot":
                bot_type = getattr(agent, "bot_type", "")
                super_user_quadrant = super_users[0][0].get_current_quadrant()

                # Adjust based on bot type and super user's quadrant
                if bot_type == "spam" and super_user_quadrant in ["pop_culture", "hobbies"]:
                    pull_strength *= 1.5
                elif bot_type == "misinformation" and super_user_quadrant == "politics_news":
                    pull_strength *= 2.0
                elif bot_type == "astroturfing" and super_user_quadrant == "tech_business":
                    pull_strength *= 2.0

            # Apply the pull (subtle adjustment to position)
            agent.topic_position['x'] += pull_x * pull_strength
            agent.topic_position['y'] += pull_y * pull_strength

            # Ensure position stays in bounds
            agent.topic_position['x'] = max(0, min(1, agent.topic_position['x']))
            agent.topic_position['y'] = max(0, min(1, agent.topic_position['y']))

    def create_initial_connections(self):
        """Create initial connections based on topic proximity."""
        # Get active agents
        active_agents = [agent for agent in self.agents if agent.active]

        # For each agent, connect to some agents that are close in topic space
        for agent in active_agents:
            # Get nearby agents with a relatively high threshold to ensure some initial connections
            nearby_agents = self.get_nearby_agents(agent, threshold=0.4)

            # Filter out self
            nearby_agents = [other for other in nearby_agents if other.unique_id != agent.unique_id]

            # Only connect to a limited number of nearby agents
            max_connections = 5  # Start with a reasonable number of connections

            if nearby_agents:
                # Select a random subset if there are many nearby agents
                num_to_connect = min(max_connections, len(nearby_agents))
                to_connect = self.random.sample(nearby_agents, num_to_connect)

                # Create connections
                for other in to_connect:
                    agent.add_connection(other)

    def get_agent_by_id(self, agent_id):
        """Retrieve an agent by their unique ID from the model's agents collection."""
        for agent in self.agents:
            if agent.unique_id == agent_id:
                return agent
        return None

    def calculate_topic_similarity(self, agent1, agent2):
        """Calculate similarity in 2D topic space between two agents."""
        # Calculate Euclidean distance in 2D space
        x1, y1 = agent1.topic_position['x'], agent1.topic_position['y']
        x2, y2 = agent2.topic_position['x'], agent2.topic_position['y']

        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Convert distance to similarity (closer = more similar)
        # Distance ranges from 0 to sqrt(2) in a unit square
        max_distance = np.sqrt(2)
        similarity = 1 - (distance / max_distance)

        return similarity

    def get_nearby_agents(self, agent, threshold=0.3):
        """Get agents that are nearby in topic space."""
        nearby_agents = []

        # Get agent's position in grid coordinates
        if not hasattr(agent, 'grid_pos'):
            # If agent doesn't have a grid position, place it first
            self.place_agent_in_topic_space(agent)

        # Calculate search radius based on threshold
        radius = int(threshold * self.grid_resolution * 0.3)  # Scale radius by threshold

        # Get all agents within radius
        neighbors = self.get_agents_in_radius(agent.grid_pos, radius)

        # Filter to active agents that are not the current agent
        for other in neighbors:
            if other.unique_id != agent.unique_id and other.active:
                # Calculate precise similarity
                similarity = self.calculate_topic_similarity(agent, other)
                if similarity > threshold:
                    nearby_agents.append(other)

        return nearby_agents

    def create_new_agents(self):
        """Create new agents based on creation rates."""
        # Track if we added any agents
        agents_added = False

        # Create new humans - Use np_random for Poisson distribution
        num_new_humans = self.np_random.poisson(self.human_creation_rate)
        new_humans = []
        for _ in range(num_new_humans):
            agent = HumanAgent(model=self)
            self.active_humans += 1

            # Place agent in topic space
            self.place_agent_in_topic_space(agent)
            new_humans.append(agent)
            agents_added = True

        # Create new bots - Use np_random for Poisson distribution
        num_new_bots = self.np_random.poisson(self.bot_creation_rate)
        new_bots = []
        for _ in range(num_new_bots):
            agent = BotAgent(model=self)

            # Apply the bot ban rate multiplier to new bots
            if hasattr(agent, 'detection_rate'):
                agent.detection_rate *= self.bot_ban_rate_multiplier

            self.active_bots += 1

            # Place agent in topic space
            self.place_agent_in_topic_space(agent)
            new_bots.append(agent)
            agents_added = True

        # Connect new agents to existing agents based on proximity
        if agents_added:
            new_agents = new_humans + new_bots
            for agent in new_agents:
                # Find nearby agents to connect with
                nearby_agents = self.get_nearby_agents(agent, threshold=0.3)

                # Filter out other new agents to focus on connecting with established agents
                established_nearby = [a for a in nearby_agents if a.unique_id != agent.unique_id]

                # Connect to a limited number of nearby established agents
                if established_nearby:
                    num_to_connect = min(3, len(established_nearby))
                    to_connect = self.random.sample(established_nearby, num_to_connect)

                    for other in to_connect:
                        agent.add_connection(other)

    def update_agent_positions(self):
        """Update agent positions in the topic space based on their topic_position."""
        for agent in self.agents:
            if agent.active:
                # Move agent to new position in grid
                self.move_agent_in_topic_space(agent)

    def update_connections_based_on_proximity(self):
        """Update connections based on topic proximity - replace the rewire_network method."""
        # Get active agents
        active_agents = [agent for agent in self.agents if agent.active]

        # Process only a subset of agents each step to prevent excessive computations
        agents_to_process = self.random.sample(
            active_agents,
            min(20, len(active_agents))  # Process at most 20 agents per step
        )

        for agent in agents_to_process:
            # FORM NEW CONNECTIONS based on proximity
            # Get nearby agents that aren't already connected
            nearby_agents = self.get_nearby_agents(agent, threshold=0.3)
            unconnected_nearby = [
                other for other in nearby_agents
                if other.unique_id != agent.unique_id and other.unique_id not in agent.connections
            ]

            # Connect to nearby agents with probability based on similarity
            for other in unconnected_nearby:
                # Calculate similarity
                similarity = self.calculate_topic_similarity(agent, other)

                # Higher similarity = higher chance to connect
                # Base probability (10%) multiplied by similarity (0-1)
                connect_prob = 0.1 * similarity

                # Adjust for super-users if applicable
                if agent.agent_type == "human" and hasattr(agent, 'is_super_user') and agent.is_super_user:
                    connect_prob *= 1.5
                if other.agent_type == "human" and hasattr(other, 'is_super_user') and other.is_super_user:
                    connect_prob *= 1.5

                # Form connection with calculated probability
                if self.random.random() < connect_prob:
                    agent.add_connection(other)

            # BREAK CONNECTIONS based on distance
            connections_to_check = list(agent.connections)
            for connection_id in connections_to_check:
                other = self.get_agent_by_id(connection_id)
                if other and other.active:
                    # Calculate similarity (inverse of distance)
                    similarity = self.calculate_topic_similarity(agent, other)

                    # Lower similarity = higher chance to break
                    # Base probability (2%) increased as similarity decreases
                    break_prob = 0.02 * (1 - similarity) * 2  # Scale up for more noticeable effect

                    # Super-users are less likely to lose connections
                    if agent.agent_type == "human" and hasattr(agent, 'is_super_user') and agent.is_super_user:
                        break_prob *= 0.7

                    # Break connection with calculated probability
                    if self.random.random() < break_prob:
                        agent.remove_connection(other)

    def update_agent_counts(self):
        """Update counters for active and deactivated agents."""
        active_humans = 0
        active_bots = 0
        deactivated_humans = 0
        deactivated_bots = 0

        for agent in self.agents:
            if getattr(agent, "agent_type", "") == "human":
                if agent.active:
                    active_humans += 1
                else:
                    deactivated_humans += 1
            elif getattr(agent, "agent_type", "") == "bot":
                if agent.active:
                    active_bots += 1
                else:
                    deactivated_bots += 1

        self.active_humans = active_humans
        self.active_bots = active_bots
        self.deactivated_humans = deactivated_humans
        self.deactivated_bots = deactivated_bots

    def get_avg_human_satisfaction(self):
        """Calculate average satisfaction of active human agents."""
        satisfactions = [
            agent.satisfaction for agent in self.agents
            if getattr(agent, "agent_type", "") == "human" and agent.active
        ]

        if satisfactions:
            return sum(satisfactions) / len(satisfactions)
        return 0

    def get_agent_quadrant_distribution(self):
        """Calculate the distribution of agents across quadrants."""
        human_distribution = {
            'tech_business': 0,
            'politics_news': 0,
            'hobbies': 0,
            'pop_culture': 0
        }

        bot_distribution = {
            'tech_business': 0,
            'politics_news': 0,
            'hobbies': 0,
            'pop_culture': 0
        }

        for agent in self.agents:
            if not agent.active:
                continue

            quadrant = agent.get_current_quadrant()

            if agent.agent_type == "human":
                human_distribution[quadrant] += 1
            elif agent.agent_type == "bot":
                bot_distribution[quadrant] += 1

        return human_distribution, bot_distribution

    def initialize_data_collector(self):
        """Initialize the data collector."""
        self.datacollector = DataCollector(
            model_reporters={
                "Active Humans": lambda m: m.active_humans,
                "Active Bots": lambda m: m.active_bots,
                "Deactivated Humans": lambda m: m.deactivated_humans,
                "Deactivated Bots": lambda m: m.deactivated_bots,
                "Average Human Satisfaction": self.get_avg_human_satisfaction,
                "Human Quadrant Distribution": lambda m: m.get_agent_quadrant_distribution()[0],
                "Bot Quadrant Distribution": lambda m: m.get_agent_quadrant_distribution()[1],
                "Human to Bot Ratio": lambda m: m.active_humans / max(1, m.active_bots)
            },
            agent_reporters={
                "Satisfaction": lambda a: getattr(a, "satisfaction", 0),
                "Agent Type": lambda a: getattr(a, "agent_type", ""),
                "Active": lambda a: getattr(a, "active", False),
                "Connections": lambda a: len(getattr(a, "connections", [])),
                "Quadrant": lambda a: getattr(a, "get_current_quadrant", lambda: "")(),
                "Is Super User": lambda a: getattr(a, "is_super_user", False),
                "Topic X": lambda a: getattr(a, "topic_position", {}).get("x", 0),
                "Topic Y": lambda a: getattr(a, "topic_position", {}).get("y", 0)
            }
        )

    def decay_connections(self):
        """Randomly decay some connections to prevent over-connectivity."""
        active_agents = [agent for agent in self.agents if agent.active]

        for agent in active_agents:
            connections_to_remove = []
            for connection_id in agent.connections:
                # 0.5% chance to decay each connection per step
                if self.random.random() < 0.005:
                    connections_to_remove.append(connection_id)

            # Remove the selected connections
            for connection_id in connections_to_remove:
                connected_agent = self.get_agent_by_id(connection_id)
                if connected_agent:
                    agent.remove_connection(connected_agent)

    def form_echo_chamber_connections(self):
        """Form new connections based on topic proximity to create echo chambers."""
        # Get active humans
        active_humans = [agent for agent in self.agents
                         if agent.active and agent.agent_type == "human"]

        # For each human, consider forming connections with nearby humans
        for human in active_humans:
            # Only process a portion of humans each step to prevent excessive connections
            if self.random.random() > 0.2:  # 20% chance to process each human
                continue

            # Get nearby humans in topic space
            nearby_humans = [a for a in self.get_nearby_agents(human, threshold=0.2)
                             if a.agent_type == "human"]

            # Form connections with some probability based on proximity
            for other in nearby_humans:
                # Skip if already connected
                if other.unique_id in human.connections:
                    continue

                # Calculate similarity
                similarity = self.calculate_topic_similarity(human, other)

                # Higher similarity = higher chance to connect
                connect_prob = similarity * 0.1  # Max 10% chance per step

                # Adjust probability for super-users (more likely to form connections)
                if hasattr(human, 'is_super_user') and human.is_super_user:
                    connect_prob *= 1.5

                if hasattr(other, 'is_super_user') and other.is_super_user:
                    connect_prob *= 1.5

                # Form connection with calculated probability
                if self.random.random() < connect_prob:
                    human.add_connection(other)

    def step(self):
        """Advance the model by one step."""
        self.agents.shuffle_do("step")

        # Apply super user gravity to all agents
        for agent in self.agents:
            if agent.active and not getattr(agent, "is_super_user", False):
                self.apply_super_user_gravity(agent)

        # Update agent positions in the topic space
        self.update_agent_positions()

        # Create new agents
        self.create_new_agents()

        # Update connections based on topic proximity
        self.update_connections_based_on_proximity()

        # Form echo chamber connections based on topic proximity
        self.form_echo_chamber_connections()

        # Apply natural connection decay
        self.decay_connections()

        # Update agent counters
        self.update_agent_counts()

        # Update data collector
        self.datacollector.collect(self)