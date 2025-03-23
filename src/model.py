"""
model.py - Implementation of a social media simulation model with quadrant-based topic space
using Mesa 3.1.4
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
            connection_rewiring_prob=constants.DEFAULT_CONNECTION_REWIRING_PROB,
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
        self.connection_rewiring_prob = connection_rewiring_prob
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

        # Create network for agent connections
        self.create_network()

        # Create initial agents
        self.create_initial_agents()

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

    def create_network(self):
        """Create a small world network for agent connections."""
        # Get current count of active agents for network size
        n = max(5, len(self.agents))  # Use current agent count, ensure at least 5 nodes
        k = min(4, n - 1)  # Each node connected to k nearest neighbors

        # Use the model's random number generator for reproducibility
        self.network = nx.watts_strogatz_graph(
            n,
            k,
            self.connection_rewiring_prob,
            seed=self.random.randint(0, 2 ** 32 - 1)
        )

    def create_initial_agents(self):
        """Create initial human and bot agents."""
        # Create humans
        for i in range(self.num_initial_humans):
            agent = HumanAgent(model=self)
            self.active_humans += 1

            # Place agent in topic space based on its position
            self.place_agent_in_topic_space(agent)

        # Create bots
        for i in range(self.num_initial_bots):
            agent = BotAgent(model=self)
            self.active_bots += 1

            # Place agent in topic space based on its position
            self.place_agent_in_topic_space(agent)

        # Create initial connections based on network topology
        self.update_agent_connections()

    def update_agent_connections(self):
        """Update agent connections based on current network topology."""
        # Reset all connections
        for agent in self.agents:
            agent.connections = set()

        # Get active agents
        active_agents = [agent for agent in self.agents if agent.active]

        # Create connections based on network edges
        for edge in self.network.edges():
            source_idx, target_idx = edge

            # Skip if indices are out of range
            if source_idx >= len(active_agents) or target_idx >= len(active_agents):
                continue

            # Get the agents using their indices in the active_agents list
            source_agent = active_agents[source_idx]
            target_agent = active_agents[target_idx]

            # Add connection between the agents
            if source_agent and target_agent:
                source_agent.add_connection(target_agent)

        # Add extra bot-human connections (preferential to super-users)
        bots = [agent for agent in active_agents if agent.agent_type == "bot"]
        humans = [agent for agent in active_agents if agent.agent_type == "human"]

        # Prioritize connections to super-users
        super_users = [h for h in humans if hasattr(h, 'is_super_user') and h.is_super_user]
        regular_users = [h for h in humans if not (hasattr(h, 'is_super_user') and h.is_super_user)]

        # Sort humans by connection count and super-user status
        sorted_humans = sorted(super_users, key=lambda h: len(h.connections)) + \
                        sorted(regular_users, key=lambda h: len(h.connections))

        for bot in bots:
            # Target up to 5 humans with priority to super-users
            target_count = min(5, len(sorted_humans))
            for i in range(target_count):
                if i < len(sorted_humans) and self.random.random() < 0.4:  # 40% chance per human
                    bot.add_connection(sorted_humans[i])

    def rewire_network(self):
        """Rewire the network connections but preserve some existing connections."""
        # Get active agents
        active_agents = [agent for agent in self.agents if agent.active]
        n = len(active_agents)

        if n <= 4:  # Need at least 5 nodes for our approach
            return

        # Store existing connections before rewiring
        existing_connections = {}
        for i, agent in enumerate(active_agents):
            existing_connections[i] = [
                active_agents.index(self.get_agent_by_id(conn_id))
                for conn_id in agent.connections
                if self.get_agent_by_id(conn_id) in active_agents
            ]

        # Create a new small world network
        k = min(4, n - 1)
        self.network = nx.watts_strogatz_graph(
            n, k, self.connection_rewiring_prob,
            seed=self.random.randint(0, 2 ** 32 - 1)
        )

        # Add some of the previous connections back (based on network_stability)
        for i, connections in existing_connections.items():
            for j in connections:
                if i < n and j < n and self.random.random() < self.network_stability:
                    self.network.add_edge(i, j)

        # Update agent connections
        self.update_agent_connections()

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
        for _ in range(num_new_humans):
            agent = HumanAgent(model=self)
            self.active_humans += 1

            # Place agent in topic space
            self.place_agent_in_topic_space(agent)

            agents_added = True

        # Create new bots - Use np_random for Poisson distribution
        num_new_bots = self.np_random.poisson(self.bot_creation_rate)
        for _ in range(num_new_bots):
            agent = BotAgent(model=self)

            # Apply the bot ban rate multiplier to new bots
            if hasattr(agent, 'detection_rate'):
                agent.detection_rate *= self.bot_ban_rate_multiplier

            self.active_bots += 1

            # Place agent in topic space
            self.place_agent_in_topic_space(agent)

            agents_added = True

        # If we added any agents, recreate the network
        if agents_added:
            self.create_network()
            self.update_agent_connections()

    def update_agent_positions(self):
        """Update agent positions in the topic space based on their topic_position."""
        for agent in self.agents:
            if agent.active:
                # Move agent to new position in grid
                self.move_agent_in_topic_space(agent)

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
        # Execute agent steps (using Mesa 3.1.4 syntax)
        self.agents.shuffle_do("step")

        # Update agent positions in the topic space
        self.update_agent_positions()

        # Create new agents
        self.create_new_agents()

        # Form echo chamber connections based on topic proximity
        self.form_echo_chamber_connections()

        # Periodically rewire the network to simulate changing trends
        if self.steps % self.topic_shift_frequency == 0:
            self.rewire_network()

        # Apply natural connection decay
        self.decay_connections()

        # Update agent counters
        self.update_agent_counts()

        # Update data collector
        self.datacollector.collect(self)