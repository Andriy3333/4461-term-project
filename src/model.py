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

            forced_feed_probability=constants.DEFAULT_FORCED_FEED_PROBABILITY,

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

        self.forced_feed_probability = forced_feed_probability

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

        self.bot_type_by_quadrant = constants.DEFAULT_BOT_TYPE_BY_QUADRANT

        # Initialize the topic space using cell space
        self.initialize_topic_space()

        # Create initial agents
        self.create_initial_agents()

        # Create initial connections based on proximity
        self.create_initial_connections()

        # Initialize data collector
        self.initialize_data_collector()

        # Create bot network connections
        self.create_bot_network_connections()

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
        self.grid_resolution = constants.DEFAULT_GRID_RESOLUTION

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
                # Pass the quadrant to BotAgent constructor for quadrant-specific type distribution
                agent = BotAgent(model=self, quadrant=quadrant)
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
        """Create new agents based on creation rates with quadrant distribution control."""
        # Track if we added any agents
        agents_added = False

        # Get current quadrant distribution
        human_dist, bot_dist = self.get_agent_quadrant_distribution()

        # Calculate total counts
        total_humans = sum(human_dist.values())
        total_bots = sum(bot_dist.values())

        # Calculate current distribution percentages
        current_human_distribution = {}
        if total_humans > 0:
            for quadrant in human_dist:
                current_human_distribution[quadrant] = human_dist[quadrant] / total_humans
        else:
            current_human_distribution = self.human_quadrant_attractiveness.copy()

        current_bot_distribution = {}
        if total_bots > 0:
            for quadrant in bot_dist:
                current_bot_distribution[quadrant] = bot_dist[quadrant] / total_bots
        else:
            current_bot_distribution = self.bot_quadrant_attractiveness.copy()

        # Create new humans - Use np_random for Poisson distribution
        num_new_humans = self.np_random.poisson(self.human_creation_rate)
        new_humans = []

        # Create humans with quadrant distribution control
        for _ in range(num_new_humans):
            # Calculate quadrant placement weights based on target vs current distribution
            placement_weights = {}
            for quadrant, target_pct in self.human_quadrant_attractiveness.items():
                current_pct = current_human_distribution.get(quadrant, 0)
                # Higher weight for underpopulated quadrants
                imbalance = target_pct - current_pct
                # Convert imbalance to a positive weight (1.0 is neutral)
                weight = 1.0 + (imbalance * 5.0)  # Scale factor for stronger correction
                # Ensure weight is positive
                placement_weights[quadrant] = max(0.1, weight)

            # Normalize weights
            weight_sum = sum(placement_weights.values())
            for quadrant in placement_weights:
                placement_weights[quadrant] /= weight_sum

            # Choose quadrant based on weights
            target_quadrant = self.random.choices(
                list(placement_weights.keys()),
                weights=list(placement_weights.values()),
                k=1
            )[0]

            # Create agent
            agent = HumanAgent(model=self)
            self.active_humans += 1

            # Set position in target quadrant with some randomness
            self.set_agent_position_in_quadrant(agent, target_quadrant)

            # Set the agent's target quadrant and commitment
            agent.current_target_quadrant = target_quadrant
            agent.target_commitment = self.random.randint(10, 20)  # Higher initial commitment

            # Place agent in topic space
            self.place_agent_in_topic_space(agent)
            new_humans.append(agent)
            agents_added = True

            # Update current distribution for next agent placement
            human_dist[target_quadrant] += 1
            total_humans += 1
            for quadrant in human_dist:
                current_human_distribution[quadrant] = human_dist[quadrant] / total_humans

        # Create new bots - Use np_random for Poisson distribution
        num_new_bots = self.np_random.poisson(self.bot_creation_rate)
        new_bots = []

        # Create bots with quadrant distribution control
        for _ in range(num_new_bots):
            # Calculate quadrant placement weights based on target vs current distribution
            placement_weights = {}
            for quadrant, target_pct in self.bot_quadrant_attractiveness.items():
                current_pct = current_bot_distribution.get(quadrant, 0)
                # Higher weight for underpopulated quadrants
                imbalance = target_pct - current_pct
                # Convert imbalance to a positive weight (1.0 is neutral)
                weight = 1.0 + (imbalance * 5.0)  # Scale factor for stronger correction
                # Ensure weight is positive
                placement_weights[quadrant] = max(0.1, weight)

            # Normalize weights
            weight_sum = sum(placement_weights.values())
            for quadrant in placement_weights:
                placement_weights[quadrant] /= weight_sum

            # Choose quadrant based on weights
            target_quadrant = self.random.choices(
                list(placement_weights.keys()),
                weights=list(placement_weights.values()),
                k=1
            )[0]

            # Create bot with specific quadrant for type distribution
            agent = BotAgent(model=self, quadrant=target_quadrant)

            # Apply the bot ban rate multiplier to new bots
            if hasattr(agent, 'detection_rate'):
                agent.detection_rate *= self.bot_ban_rate_multiplier

            self.active_bots += 1

            # Set position in target quadrant with some randomness
            self.set_agent_position_in_quadrant(agent, target_quadrant)

            # Place agent in topic space
            self.place_agent_in_topic_space(agent)
            new_bots.append(agent)
            agents_added = True

            # Update current distribution for next agent placement
            bot_dist[target_quadrant] += 1
            total_bots += 1
            for quadrant in bot_dist:
                current_bot_distribution[quadrant] = bot_dist[quadrant] / total_bots

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
            min(constants.DEFAULT_AGENTS_TO_PROCESS_PER_STEP, len(active_agents))
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
                connect_prob = constants.DEFAULT_PROXIMITY_CONNECTION_PROBABILITY * similarity

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
                    break_prob = constants.DEFAULT_CONNECTION_BREAKING_BASE_PROB * (1 - similarity) * 2

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
                if self.random.random() < constants.DEFAULT_CONNECTION_DECAY_PROB:
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

    def create_bot_network_connections(self):
        """
        Create network connections between bots in the same quadrant.
        Each bot will connect to 4-5 other bots in their quadrant.
        """
        # Group active bots by quadrant
        bots_by_quadrant = {
            'tech_business': [],
            'politics_news': [],
            'hobbies': [],
            'pop_culture': []
        }

        # Find all active bots and group them by quadrant
        for agent in self.agents:
            if agent.active and agent.agent_type == "bot":
                quadrant = agent.get_current_quadrant()
                bots_by_quadrant[quadrant].append(agent)

        # For each quadrant, create the bot network connections
        for quadrant, bots in bots_by_quadrant.items():
            # Skip if not enough bots in this quadrant
            if len(bots) <= 1:
                continue

            for bot in bots:
                # Determine how many connections to make (4-5)
                num_connections = self.random.randint(4, 5)

                # Make sure we don't try to connect to more bots than available
                num_connections = min(num_connections, len(bots) - 1)

                # Get potential connection targets (all other bots in same quadrant)
                potential_targets = [other for other in bots if other.unique_id != bot.unique_id]

                # If we have enough bots for the desired connections
                if potential_targets and num_connections > 0:
                    # Select random bots to connect with
                    targets = self.random.sample(potential_targets, num_connections)

                    # Create connections
                    for target in targets:
                        bot.add_connection(target)

                        # Add some reciprocal connections (but not all)
                        if self.random.random() < 0.7:  # 70% chance for reciprocal connection
                            target.add_connection(bot)

    def perform_bot_interactions(self):
        """Make bots actively seek out and influence human agents."""
        active_bots = [agent for agent in self.agents if agent.active and agent.agent_type == "bot"]
        active_humans = [agent for agent in self.agents if agent.active and agent.agent_type == "human"]

        if not active_bots or not active_humans:
            return  # No bots or humans to interact

        # For each bot, attempt to influence humans
        for bot in active_bots:
            # Only proceed if bot posted today
            if not getattr(bot, 'posted_today', False):
                continue

            # Determine number of non-connected humans to try to influence
            # based on bot type and quadrant
            influence_count = 2  # Base number

            # Adjust based on bot type
            if bot.bot_type == "misinformation":
                influence_count += 2  # More aggressive influence
            elif bot.bot_type == "astroturfing":
                influence_count += 1  # Moderate influence

            # Get humans who aren't already connected to this bot
            unconnected_humans = [human for human in active_humans
                                  if human.unique_id not in bot.connections]

            # Filter to humans in the same quadrant (for more targeted influence)
            bot_quadrant = bot.get_current_quadrant()
            targeted_humans = [human for human in unconnected_humans
                               if human.get_current_quadrant() == bot_quadrant]

            # If not enough humans in same quadrant, use any unconnected humans
            if len(targeted_humans) < influence_count and unconnected_humans:
                additional_humans = [h for h in unconnected_humans if h not in targeted_humans]
                targeted_humans.extend(additional_humans)

            # Limit to the target influence count
            if targeted_humans and influence_count > 0:
                humans_to_influence = self.random.sample(
                    targeted_humans,
                    min(influence_count, len(targeted_humans))
                )

                # Attempt to influence each human
                for human in humans_to_influence:
                    # Calculate chance of successful influence based on bot type
                    influence_chance = 0.3  # Base chance

                    # Adjust based on bot type
                    if bot.bot_type == "misinformation" and bot_quadrant == "politics_news":
                        influence_chance = 0.5  # Higher in politics for misinfo
                    elif bot.bot_type == "astroturfing" and bot_quadrant == "tech_business":
                        influence_chance = 0.5  # Higher in tech for astroturfing
                    elif bot.bot_type == "spam" and bot_quadrant in ["pop_culture", "hobbies"]:
                        influence_chance = 0.4  # Higher in casual quadrants for spam

                    # Reduce influence chance based on human authenticity
                    if hasattr(human, 'authenticity'):
                        influence_chance *= (2.0 - human.authenticity)

                    # Attempt influence
                    if self.random.random() < influence_chance:
                        # Calculate impact on satisfaction
                        base_impact = -0.5  # Base negative impact

                        # Adjust based on bot post type
                        if bot.post_type == "misinformation":
                            base_impact = -1.0
                        elif bot.post_type == "astroturfing":
                            base_impact = -0.8
                        elif bot.post_type == "spam":
                            base_impact = -0.5

                        # Scale by human irritability
                        if hasattr(human, 'irritability'):
                            base_impact *= human.irritability

                        # Apply impact
                        human.satisfaction += base_impact * 5  # Scale for noticeable effect

                        # Cap satisfaction
                        human.satisfaction = max(0, min(100, human.satisfaction))

                        # With small chance, create connection
                        if self.random.random() < 0.1:  # 10% chance
                            bot.add_connection(human)

                            # Even smaller chance for reciprocal connection
                            if self.random.random() < 0.05:  # 5% chance
                                human.add_connection(bot)

    def enforce_quadrant_distribution(self):
        """
        Enforce the target quadrant distribution by selectively moving agents.
        Called periodically to maintain distribution close to targets.
        """
        # Only enforce every 5 steps to allow natural movement between corrections
        if self.steps % 5 != 0:
            return

        # Get target distribution
        target_distribution = self.human_quadrant_attractiveness

        # Get current distribution
        human_dist, _ = self.get_agent_quadrant_distribution()

        # Calculate total humans for percentage calculation
        total_humans = sum(human_dist.values())
        if total_humans == 0:
            return  # No humans to distribute

        current_distribution = {}
        for quadrant in human_dist:
            current_distribution[quadrant] = human_dist[quadrant] / total_humans

        # Calculate quadrant imbalance
        quadrant_imbalance = {}
        for quadrant in target_distribution:
            target_pct = target_distribution[quadrant]
            current_pct = current_distribution.get(quadrant, 0)
            quadrant_imbalance[quadrant] = current_pct - target_pct

        # Identify overpopulated and underpopulated quadrants
        overpopulated = {q: imbalance for q, imbalance in quadrant_imbalance.items() if imbalance > 0.02}
        underpopulated = {q: -imbalance for q, imbalance in quadrant_imbalance.items() if imbalance < -0.02}

        # If there's nothing to correct, exit
        if not overpopulated or not underpopulated:
            return

        # Sort quadrants by degree of imbalance
        overpopulated_quadrants = sorted(overpopulated.keys(), key=lambda q: overpopulated[q], reverse=True)
        underpopulated_quadrants = sorted(underpopulated.keys(), key=lambda q: underpopulated[q], reverse=True)

        # Get all active human agents
        active_humans = [agent for agent in self.agents if agent.active and agent.agent_type == "human"]

        # For each overpopulated quadrant, move some agents to underpopulated quadrants
        for over_q in overpopulated_quadrants:
            # Calculate how many agents to move (proportional to imbalance)
            imbalance_pct = overpopulated[over_q]
            agents_to_move = int(total_humans * imbalance_pct * 0.2)  # Move 20% of the imbalance
            agents_to_move = max(1, min(5, agents_to_move))  # At least 1, at most 5 agents per step

            # Find agents in this quadrant
            agents_in_quadrant = [agent for agent in active_humans if agent.get_current_quadrant() == over_q]

            # Exclude super users from forced movement, they're community anchors
            non_super_agents = [agent for agent in agents_in_quadrant
                                if not getattr(agent, 'is_super_user', False)]

            # If not enough non-super agents, reduce the move count
            agents_to_move = min(agents_to_move, len(non_super_agents))

            if agents_to_move == 0 or not underpopulated_quadrants:
                continue

            # Select agents to move (prefer those who recently moved or have low commitment)
            agents_to_relocate = []
            if non_super_agents:
                # Sort by commitment - lower commitment means easier to move
                sorted_agents = sorted(non_super_agents,
                                       key=lambda a: getattr(a, 'target_commitment', 0))
                agents_to_relocate = sorted_agents[:agents_to_move]

            # For each agent, assign a new target in an underpopulated quadrant
            for agent in agents_to_relocate:
                # Choose an underpopulated quadrant
                target_quadrant = underpopulated_quadrants[0]

                # Rotate underpopulated quadrants list to distribute agents evenly
                underpopulated_quadrants = underpopulated_quadrants[1:] + [underpopulated_quadrants[0]]

                # Set new target for agent
                agent.current_target_quadrant = target_quadrant

                # Reset commitment to ensure immediate movement
                agent.target_commitment = 0

                # Calculate target coordinates for new quadrant
                if target_quadrant == 'tech_business':
                    target_x, target_y = 0.25, 0.25  # Q1 center
                elif target_quadrant == 'politics_news':
                    target_x, target_y = 0.25, 0.75  # Q2 center
                elif target_quadrant == 'hobbies':
                    target_x, target_y = 0.75, 0.25  # Q3 center
                else:  # pop_culture
                    target_x, target_y = 0.75, 0.75  # Q4 center

                # Add some randomness to avoid clustering
                target_x += self.random.uniform(-0.15, 0.15)
                target_y += self.random.uniform(-0.15, 0.15)

                # Ensure coordinates are within bounds
                target_x = max(0, min(1, target_x))
                target_y = max(0, min(1, target_y))

                # Set new target
                agent.target_x = target_x
                agent.target_y = target_y

    def force_bot_connections(self, min_connections=5):
        """
        Force bot-to-bot connections regardless of other factors.
        This is a direct, brute-force approach that guarantees bot connectivity.
        Call this from the model's step method.

        Parameters:
        -----------
        min_connections : int
            Minimum number of bot-to-bot connections each bot should have
        """
        # Get all active bots
        active_bots = [agent for agent in self.agents
                       if agent.active and agent.agent_type == "bot"]

        # Skip if not enough bots
        if len(active_bots) <= 1:
            return

        # Ensure each bot has minimum connections with other bots
        for bot in active_bots:
            # Count current bot connections
            bot_connections = []
            for conn_id in bot.connections:
                other = self.get_agent_by_id(conn_id)
                if other and other.active and other.agent_type == "bot":
                    bot_connections.append(other)

            current_bot_connections = len(bot_connections)

            # If the bot needs more connections
            if current_bot_connections < min_connections:
                # Get all bots that aren't already connected
                available_bots = [other for other in active_bots
                                  if other.unique_id != bot.unique_id and
                                  other.unique_id not in bot.connections]

                # Sort by quadrant similarity - prefer same quadrant
                bot_quadrant = bot.get_current_quadrant()
                same_quadrant = [other for other in available_bots
                                 if other.get_current_quadrant() == bot_quadrant]
                other_quadrant = [other for other in available_bots
                                  if other.get_current_quadrant() != bot_quadrant]

                # Combine lists with priority (same quadrant first)
                sorted_bots = same_quadrant + other_quadrant

                # Calculate how many connections to add
                connections_needed = min(min_connections - current_bot_connections,
                                         len(sorted_bots))

                # Add connections
                for i in range(connections_needed):
                    other_bot = sorted_bots[i]

                    # Force bidirectional connection
                    bot.connections.add(other_bot.unique_id)
                    other_bot.connections.add(bot.unique_id)


        # Check connectivity after forcing connections
        total_bots = len(active_bots)
        connected_bots = 0

        for bot in active_bots:
            bot_connections = 0
            for conn_id in bot.connections:
                other = self.get_agent_by_id(conn_id)
                if other and other.active and other.agent_type == "bot":
                    bot_connections += 1

            if bot_connections >= min_connections:
                connected_bots += 1


    def step(self):
        """Advance the model by one step."""
        # Step all agents in random order
        self.agents.shuffle_do("step")

        # Apply super user gravity to all agents
        for agent in self.agents:
            if agent.active and not getattr(agent, "is_super_user", False):
                self.apply_super_user_gravity(agent)

        # Enforce quadrant distribution (new)
        self.enforce_quadrant_distribution()

        # Update agent positions in the topic space
        self.update_agent_positions()

        # Create new agents
        self.create_new_agents()

        # Update connections based on topic proximity
        self.update_connections_based_on_proximity()

        # Form echo chamber connections based on topic proximity
        self.form_echo_chamber_connections()

        # Perform active bot interaction with humans
        self.perform_bot_interactions()

        self.force_bot_connections(min_connections=5)

        # Apply natural connection decay
        self.decay_connections()

        # Update agent counters
        self.update_agent_counts()

        # Update data collector
        self.datacollector.collect(self)