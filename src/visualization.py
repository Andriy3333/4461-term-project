"""
quadrant_visualization.py - Solara 1.44.1 visualization for quadrant-based social media simulation
using Mesa 3.1.4 with interactive controls and topic space visualization
"""

import solara
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from functools import partial
import io
from PIL import Image

from model import QuadrantTopicModel
import constants


# Global figure management - clear all matplotlib figures when needed
def clear_all_figures():
    plt.close('all')


class SimulationState:
    """Class to hold simulation state for atomic updates"""
    def __init__(self):
        self.model = None
        self.model_data_list = []
        self.update_counter = 0

    def add_data_row(self, row):
        self.model_data_list.append(row)
        self.update_counter += 1

    def set_model(self, model):
        self.model = model
        self.model_data_list = []
        self.update_counter += 1

    @property
    def steps(self):
        return self.model.steps if self.model else 0

    @property
    def active_humans(self):
        return self.model.active_humans if self.model else 0

    @property
    def active_bots(self):
        return self.model.active_bots if self.model else 0

    def get_avg_human_satisfaction(self):
        return self.model.get_avg_human_satisfaction() if self.model else 0

    def get_human_quadrant_distribution(self):
        if self.model:
            human_dist, _ = self.model.get_agent_quadrant_distribution()
            return human_dist
        return {'tech_business': 0, 'politics_news': 0, 'hobbies': 0, 'pop_culture': 0}

    def get_bot_quadrant_distribution(self):
        if self.model:
            _, bot_dist = self.model.get_agent_quadrant_distribution()
            return bot_dist
        return {'tech_business': 0, 'politics_news': 0, 'hobbies': 0, 'pop_culture': 0}


def topic_space_visualization(model):
    """Create a visualization of agents in the 2D topic space with quadrants"""
    fig, ax = plt.subplots(figsize=(8, 7))

    if not model:
        ax.text(0.5, 0.5, "Topic space visualization will appear after the model is initialized.\n\n"
                          "Press apply changes if changing parameters.",
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        return fig

    # Draw quadrant boundaries
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Label quadrants
    ax.text(0.25, 0.25, "Q1: Tech/Business", ha='center', va='center', fontsize=10, alpha=0.7)
    ax.text(0.25, 0.75, "Q2: Politics/News", ha='center', va='center', fontsize=10, alpha=0.7)
    ax.text(0.75, 0.25, "Q3: Hobbies", ha='center', va='center', fontsize=10, alpha=0.7)
    ax.text(0.75, 0.75, "Q4: Pop Culture", ha='center', va='center', fontsize=10, alpha=0.7)

    # Color blocks for quadrant background
    q1 = Rectangle((0, 0), 0.5, 0.5, color='lightblue', alpha=0.2)
    q2 = Rectangle((0, 0.5), 0.5, 0.5, color='lightgreen', alpha=0.2)
    q3 = Rectangle((0.5, 0), 0.5, 0.5, color='lightyellow', alpha=0.2)
    q4 = Rectangle((0.5, 0.5), 0.5, 0.5, color='mistyrose', alpha=0.2)

    ax.add_patch(q1)
    ax.add_patch(q2)
    ax.add_patch(q3)
    ax.add_patch(q4)

    # Label axes
    ax.set_xlabel("Serious (0) to Casual (1)", fontsize=10)
    ax.set_ylabel("Individual (0) to Societal (1)", fontsize=10)

    # Set axis limits with some padding
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Plot agents by type
    humans_x = []
    humans_y = []
    super_users_x = []
    super_users_y = []

    # Separate bot types
    spam_bots_x = []
    spam_bots_y = []
    misinfo_bots_x = []
    misinfo_bots_y = []
    astro_bots_x = []
    astro_bots_y = []

    # Process all active agents
    for agent in model.agents:
        if not agent.active:
            continue

        x = agent.topic_position['x']
        y = agent.topic_position['y']

        if agent.agent_type == 'human':
            if getattr(agent, 'is_super_user', False):
                super_users_x.append(x)
                super_users_y.append(y)
            else:
                humans_x.append(x)
                humans_y.append(y)
        else:  # Bot
            bot_type = getattr(agent, 'bot_type', 'unknown')
            if bot_type == 'spam':
                spam_bots_x.append(x)
                spam_bots_y.append(y)
            elif bot_type == 'misinformation':
                misinfo_bots_x.append(x)
                misinfo_bots_y.append(y)
            elif bot_type == 'astroturfing':
                astro_bots_x.append(x)
                astro_bots_y.append(y)

    # Plot regular humans
    if humans_x:
        ax.scatter(humans_x, humans_y, color='blue', marker='o', s=30, alpha=0.7, label='Regular Users')

    # Plot super users with a different marker
    if super_users_x:
        ax.scatter(super_users_x, super_users_y, color='darkblue', marker='*', s=60, alpha=0.7, label='Super Users')

    # Plot bots with different markers/colors by type
    if spam_bots_x:
        ax.scatter(spam_bots_x, spam_bots_y, color='red', marker='s', s=30, alpha=0.7, label='Spam Bots')

    if misinfo_bots_x:
        ax.scatter(misinfo_bots_x, misinfo_bots_y, color='darkred', marker='^', s=30, alpha=0.7, label='Misinfo Bots')

    if astro_bots_x:
        ax.scatter(astro_bots_x, astro_bots_y, color='orangered', marker='d', s=30, alpha=0.7, label='Astroturf Bots')

    # Add title and legend
    ax.set_title(f"Topic Space Distribution (Step {model.steps})", fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8)

    # Set grid
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    return fig


def quadrant_distribution_visualization(model):
    """Create bar charts showing agent distribution across quadrants"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if not model:
        ax1.text(0.5, 0.5, "Quadrant distribution will appear after model initialization.",
                ha='center', va='center', fontsize=10)
        ax2.text(0.5, 0.5, "Quadrant distribution will appear after model initialization.",
                ha='center', va='center', fontsize=10)
        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        return fig

    # Get distribution data
    human_dist, bot_dist = model.get_agent_quadrant_distribution()

    # Calculate percentages
    human_total = sum(human_dist.values())
    bot_total = sum(bot_dist.values())

    # Handle edge case of no agents
    if human_total == 0:
        human_pct = {k: 0 for k in human_dist}
    else:
        human_pct = {k: (v/human_total*100) for k, v in human_dist.items()}

    if bot_total == 0:
        bot_pct = {k: 0 for k in bot_dist}
    else:
        bot_pct = {k: (v/bot_total*100) for k, v in bot_dist.items()}

    # Plot data for humans
    quadrants = ['tech_business', 'politics_news', 'hobbies', 'pop_culture']
    pretty_names = ['Tech/Business', 'Politics/News', 'Hobbies', 'Pop Culture']
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'mistyrose']

    # Humans plot (left)
    human_values = [human_pct[q] for q in quadrants]
    human_counts = [human_dist[q] for q in quadrants]

    bars1 = ax1.bar(pretty_names, human_values, color=colors, alpha=0.7)
    ax1.set_ylim(0, 100)
    ax1.set_title(f'Human Distribution (n={human_total})', fontsize=10)
    ax1.set_ylabel('Percentage (%)', fontsize=9)
    ax1.tick_params(axis='x', labelrotation=30, labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)

    # Add count labels to bars
    for bar, count in zip(bars1, human_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontsize=7)

    # Bots plot (right)
    bot_values = [bot_pct[q] for q in quadrants]
    bot_counts = [bot_dist[q] for q in quadrants]

    bars2 = ax2.bar(pretty_names, bot_values, color=colors, alpha=0.7)
    ax2.set_ylim(0, 100)
    ax2.set_title(f'Bot Distribution (n={bot_total})', fontsize=10)
    ax2.set_ylabel('Percentage (%)', fontsize=9)
    ax2.tick_params(axis='x', labelrotation=30, labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)

    # Add count labels to bars
    for bar, count in zip(bars2, bot_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontsize=7)

    # Target line for human distribution
    target_human = [constants.DEFAULT_HUMAN_QUADRANT_ATTRACTIVENESS[q]*100 for q in quadrants]
    ax1.plot(pretty_names, target_human, 'k--', alpha=0.5, label='Target')
    ax1.legend(fontsize=7)

    # Target line for bot distribution
    target_bot = [constants.DEFAULT_BOT_QUADRANT_ATTRACTIVENESS[q]*100 for q in quadrants]
    ax2.plot(pretty_names, target_bot, 'k--', alpha=0.5, label='Target')
    ax2.legend(fontsize=7)

    plt.tight_layout()
    return fig


def satisfaction_visualization(model):
    """Create histogram and line plot for human satisfaction with dynamic scaling"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if not model:
        ax1.text(0.5, 0.5, "Satisfaction data will appear after model initialization.",
                 ha='center', va='center', fontsize=10)
        ax2.text(0.5, 0.5, "Satisfaction data will appear after model initialization.",
                 ha='center', va='center', fontsize=10)
        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        return fig

    # Get satisfaction values from active human agents
    satisfaction_values = [
        agent.satisfaction for agent in model.agents
        if getattr(agent, "agent_type", "") == "human" and agent.active
    ]

    # Calculate quadrant-specific satisfaction
    quadrant_satisfaction = {
        'tech_business': [],
        'politics_news': [],
        'hobbies': [],
        'pop_culture': []
    }

    for agent in model.agents:
        if agent.active and agent.agent_type == 'human':
            quadrant = agent.get_current_quadrant()
            quadrant_satisfaction[quadrant].append(agent.satisfaction)

    # Calculate average satisfaction by quadrant
    quadrant_avg_satisfaction = {}
    for q, values in quadrant_satisfaction.items():
        if values:
            quadrant_avg_satisfaction[q] = sum(values) / len(values)
        else:
            quadrant_avg_satisfaction[q] = 0

    # Create histogram on left plot with adaptive binning and scaling
    if satisfaction_values:
        # Calculate optimal bin count using Freedman-Diaconis rule
        # This scales better with large datasets than fixed bin count
        q75, q25 = np.percentile(satisfaction_values, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(satisfaction_values) ** (1 / 3)) if iqr > 0 else 5
        bin_count = max(10, min(50, int(np.ceil((100) / bin_width)))) if bin_width > 0 else 20

        # Create histogram with dynamic bins
        n, bins, patches = ax1.hist(satisfaction_values, bins=bin_count,
                                    range=(0, 100), alpha=0.7, color='green')

        ax1.set_title(f"Human Satisfaction Distribution (Avg: {np.mean(satisfaction_values):.1f})", fontsize=10)
        ax1.set_xlabel("Satisfaction Level", fontsize=9)
        ax1.set_ylabel("Number of Humans", fontsize=9)
        ax1.set_xlim(0, 100)

        # Set a dynamic y-axis limit based on the histogram height
        # Add 10% padding to the top for readability
        if max(n) > 0:
            ax1.set_ylim(0, max(n) * 1.1)
        else:
            ax1.set_ylim(0, 20)

        # Add grid for easier reading with large datasets
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

    else:
        ax1.text(0.5, 0.5, "No active human agents",
                 ha='center', va='center')
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 20)

    # Create bar chart of satisfaction by quadrant on right plot
    quadrants = ['tech_business', 'politics_news', 'hobbies', 'pop_culture']
    pretty_names = ['Tech/Business', 'Politics/News', 'Hobbies', 'Pop Culture']
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'mistyrose']

    values = [quadrant_avg_satisfaction.get(q, 0) for q in quadrants]
    bars = ax2.bar(pretty_names, values, color=colors, alpha=0.7)

    # Add data labels with adaptive positioning
    for bar, value in zip(bars, values):
        if value > 0:
            # Position labels with smarter vertical offset as values grow
            label_y_pos = value + max(1, min(5, value * 0.03))
            ax2.text(bar.get_x() + bar.get_width() / 2., label_y_pos,
                     f'{value:.1f}', ha='center', va='bottom', fontsize=8)

    ax2.set_title("Average Satisfaction by Quadrant", fontsize=10)
    ax2.set_xlabel("Quadrant", fontsize=9)
    ax2.set_ylabel("Average Satisfaction", fontsize=9)

    # Keep y-axis fixed at 0-100 since satisfaction is on that scale
    # Could be adjusted if values were to exceed 100
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', labelrotation=30, labelsize=8)

    # Add grid for readability
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Add the overall average line
    if satisfaction_values:
        avg = np.mean(satisfaction_values)
        ax2.axhline(avg, color='red', linestyle='--', alpha=0.5,
                    label=f'Overall Avg: {avg:.1f}')
        ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig


def population_metrics_visualization(df):
    """Create visualization of population metrics over time"""
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Population data will appear after running steps.",
                ha='center', va='center', fontsize=10)
        ax.axis('off')
        plt.tight_layout()
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left plot: Population counts over time
    ax1.plot(df['step'], df['Active Humans'], label='Humans', color='blue')
    ax1.plot(df['step'], df['Active Bots'], label='Bots', color='red')
    ax1.set_xlabel('Step', fontsize=9)
    ax1.set_ylabel('Count', fontsize=9)
    ax1.set_title('Population Over Time', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right plot: Satisfaction trend
    ax2.plot(df['step'], df['Average Human Satisfaction'], color='green')
    ax2.set_xlabel('Step', fontsize=9)
    ax2.set_ylabel('Satisfaction Level', fontsize=9)
    ax2.set_title('Average Human Satisfaction Over Time', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def network_visualization(model):
    """Create a simplified visualization of the agent connection network"""
    fig, ax = plt.subplots(figsize=(7, 6))

    if not model or model.steps == 0:
        ax.text(0.5, 0.5, "Network visualization will appear after the first step.",
                ha='center', va='center', fontsize=10)
        ax.axis('off')
        plt.tight_layout()
        return fig

    # Get all active agents with at least one connection
    connected_agents = [agent for agent in model.agents
                       if agent.active and len(agent.connections) > 0]

    # Count connection types for display
    human_human = 0
    human_bot = 0
    bot_bot = 0

    # Create connections dictionary for drawing lines
    connections = []

    for agent in connected_agents:
        for conn_id in agent.connections:
            other = model.get_agent_by_id(conn_id)
            if other and other.active:
                # Only process each connection once
                if agent.unique_id < other.unique_id:
                    connections.append((agent, other))

                    # Count connection types
                    if agent.agent_type == 'human' and other.agent_type == 'human':
                        human_human += 1
                    elif agent.agent_type == 'bot' and other.agent_type == 'bot':
                        bot_bot += 1
                    else:
                        human_bot += 1

    # If there are no connections, show a message
    if not connections:
        ax.text(0.5, 0.5, "No connections between agents yet.",
                ha='center', va='center', fontsize=10)
        ax.axis('off')
        plt.tight_layout()
        return fig

    # Create scatterplot of agents by position with connections
    # Categorize by agent type
    humans_regular = [agent for agent in connected_agents
                    if agent.agent_type == 'human' and not getattr(agent, 'is_super_user', False)]
    humans_super = [agent for agent in connected_agents
                   if agent.agent_type == 'human' and getattr(agent, 'is_super_user', False)]
    bots_spam = [agent for agent in connected_agents
                if agent.agent_type == 'bot' and getattr(agent, 'bot_type', '') == 'spam']
    bots_misinfo = [agent for agent in connected_agents
                  if agent.agent_type == 'bot' and getattr(agent, 'bot_type', '') == 'misinformation']
    bots_astro = [agent for agent in connected_agents
                if agent.agent_type == 'bot' and getattr(agent, 'bot_type', '') == 'astroturfing']

    # Draw quadrant boundaries
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Label quadrants
    ax.text(0.25, 0.25, "Q1: Tech/Business", ha='center', va='center', fontsize=8, alpha=0.7)
    ax.text(0.25, 0.75, "Q2: Politics/News", ha='center', va='center', fontsize=8, alpha=0.7)
    ax.text(0.75, 0.25, "Q3: Hobbies", ha='center', va='center', fontsize=8, alpha=0.7)
    ax.text(0.75, 0.75, "Q4: Pop Culture", ha='center', va='center', fontsize=8, alpha=0.7)

    # Draw connections first (behind nodes)
    for agent1, agent2 in connections:
        x1, y1 = agent1.topic_position['x'], agent1.topic_position['y']
        x2, y2 = agent2.topic_position['x'], agent2.topic_position['y']

        # Determine connection type for color
        if agent1.agent_type == 'human' and agent2.agent_type == 'human':
            color = 'blue'
            alpha = 0.3
        elif agent1.agent_type == 'bot' and agent2.agent_type == 'bot':
            color = 'red'
            alpha = 0.3
        else:
            color = 'purple'
            alpha = 0.3

        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.5)

    # Plot different agent types
    # Regular humans
    if humans_regular:
        x = [agent.topic_position['x'] for agent in humans_regular]
        y = [agent.topic_position['y'] for agent in humans_regular]
        ax.scatter(x, y, color='blue', marker='o', s=30, alpha=0.7, label='Regular Users')

    # Super users
    if humans_super:
        x = [agent.topic_position['x'] for agent in humans_super]
        y = [agent.topic_position['y'] for agent in humans_super]
        ax.scatter(x, y, color='darkblue', marker='*', s=80, alpha=0.7, label='Super Users')

    # Spam bots
    if bots_spam:
        x = [agent.topic_position['x'] for agent in bots_spam]
        y = [agent.topic_position['y'] for agent in bots_spam]
        ax.scatter(x, y, color='red', marker='s', s=30, alpha=0.7, label='Spam Bots')

    # Misinformation bots
    if bots_misinfo:
        x = [agent.topic_position['x'] for agent in bots_misinfo]
        y = [agent.topic_position['y'] for agent in bots_misinfo]
        ax.scatter(x, y, color='darkred', marker='^', s=30, alpha=0.7, label='Misinfo Bots')

    # Astroturfing bots
    if bots_astro:
        x = [agent.topic_position['x'] for agent in bots_astro]
        y = [agent.topic_position['y'] for agent in bots_astro]
        ax.scatter(x, y, color='orangered', marker='d', s=30, alpha=0.7, label='Astroturf Bots')

    # Add legend and title
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8)
    ax.set_title(f"Agent Network (Step {model.steps})", fontsize=12)

    # Add connection type summary
    summary = f"Human-Human: {human_human}, Human-Bot: {human_bot}, Bot-Bot: {bot_bot}"
    ax.text(0.5, -0.1, summary, ha='center', va='center',
            transform=ax.transAxes, fontsize=8)

    # Set limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("Serious (0) to Casual (1)", fontsize=8)
    ax.set_ylabel("Individual (0) to Societal (1)", fontsize=8)

    plt.tight_layout()
    return fig


def bot_types_visualization(model):
    """Create visualization of bot types and their distribution"""
    fig, ax = plt.subplots(figsize=(6, 4))

    if not model:
        ax.text(0.5, 0.5, "Bot type data will appear after model initialization.",
                ha='center', va='center', fontsize=10)
        ax.axis('off')
        plt.tight_layout()
        return fig

    # Count bot types
    spam_bots = 0
    misinfo_bots = 0
    astro_bots = 0

    for agent in model.agents:
        if agent.active and agent.agent_type == 'bot':
            bot_type = getattr(agent, 'bot_type', '')
            if bot_type == 'spam':
                spam_bots += 1
            elif bot_type == 'misinformation':
                misinfo_bots += 1
            elif bot_type == 'astroturfing':
                astro_bots += 1

    # Create pie chart
    bot_types = ['Spam', 'Misinformation', 'Astroturfing']
    counts = [spam_bots, misinfo_bots, astro_bots]
    colors = ['red', 'darkred', 'orangered']

    # If there are no bots, show a message
    if sum(counts) == 0:
        ax.text(0.5, 0.5, "No active bots in the simulation.",
                ha='center', va='center', fontsize=10)
        ax.axis('off')
        plt.tight_layout()
        return fig

    # Calculate percentages for display
    total = sum(counts)
    percentages = [(count/total*100) for count in counts]
    labels = [f"{bot_type}\n{count} ({pct:.1f}%)" for bot_type, count, pct in zip(bot_types, counts, percentages)]

    ax.pie(counts, labels=labels, colors=colors, autopct='',
          startangle=90, wedgeprops={'alpha': 0.7})

    ax.set_title(f"Bot Types Distribution (n={total})", fontsize=12)

    plt.tight_layout()
    return fig


@solara.component
def QuadrantDashboard():
    """Main dashboard component for quadrant-based social media simulation"""
    # Model parameters with state
    num_initial_humans, set_num_initial_humans = solara.use_state(constants.DEFAULT_INITIAL_HUMANS)
    num_initial_bots, set_num_initial_bots = solara.use_state(constants.DEFAULT_INITIAL_BOTS)
    human_creation_rate, set_human_creation_rate = solara.use_state(constants.DEFAULT_HUMAN_CREATION_RATE)
    bot_creation_rate, set_bot_creation_rate = solara.use_state(constants.DEFAULT_BOT_CREATION_RATE)
    bot_ban_rate_multiplier, set_bot_ban_rate_multiplier = solara.use_state(constants.DEFAULT_BOT_BAN_RATE_MULTIPLIER)

    # Topic and interaction parameters
    human_human_positive_bias, set_human_human_positive_bias = solara.use_state(constants.DEFAULT_HUMAN_HUMAN_POSITIVE_BIAS)
    human_bot_negative_bias, set_human_bot_negative_bias = solara.use_state(constants.DEFAULT_HUMAN_BOT_NEGATIVE_BIAS)
    human_satisfaction_init, set_human_satisfaction_init = solara.use_state(constants.DEFAULT_HUMAN_SATISFACTION_INIT)

    # Network parameters
    network_stability, set_network_stability = solara.use_state(constants.DEFAULT_NETWORK_STABILITY)
    topic_shift_frequency, set_topic_shift_frequency = solara.use_state(constants.DEFAULT_TOPIC_SHIFT_FREQUENCY)

    # Seed for reproducibility
    seed, set_seed = solara.use_state(42)

    # Flag to indicate parameters have changed
    params_changed, set_params_changed = solara.use_state(False)

    # Create a unified simulation state
    sim_state, set_sim_state = solara.use_state(SimulationState())

    # Function to update a parameter and mark as changed
    def update_param(value, setter):
        setter(value)
        set_params_changed(True)

    # Function to create a new model with current parameters
    def create_new_model():
        clear_all_figures()

        # Create the model with current parameters
        new_model = QuadrantTopicModel(
            num_initial_humans=num_initial_humans,
            num_initial_bots=num_initial_bots,
            human_creation_rate=human_creation_rate,
            bot_creation_rate=bot_creation_rate,
            bot_ban_rate_multiplier=bot_ban_rate_multiplier,
            network_stability=network_stability,
            topic_shift_frequency=topic_shift_frequency,
            human_human_positive_bias=human_human_positive_bias,
            human_bot_negative_bias=human_bot_negative_bias,
            human_satisfaction_init=human_satisfaction_init,
            seed=seed
        )

        # Create a new simulation state
        new_state = SimulationState()
        new_state.set_model(new_model)

        # Set the new simulation state and clear changed flag
        set_sim_state(new_state)
        set_params_changed(False)

    # Initialize the model if it's None
    if sim_state.model is None:
        create_new_model()

    # Function to run a single step
    def step():
        if sim_state.model:
            # Clear figures
            clear_all_figures()

            # Create a new state object to update atomically
            new_state = SimulationState()
            new_state.model = sim_state.model
            new_state.model_data_list = sim_state.model_data_list.copy()

            # Step the model
            new_state.model.step()

            # Get current data as a dictionary
            df_row = new_state.model.datacollector.get_model_vars_dataframe().iloc[-1:].to_dict('records')[0]
            df_row['step'] = new_state.model.steps

            # Add the new data row
            new_state.add_data_row(df_row)

            # Update the state just once
            set_sim_state(new_state)

    # Function to run multiple steps
    def run_multiple_steps(num_steps):
        if sim_state.model:
            # Clear figures once
            clear_all_figures()

            # Make a copy of the state
            new_state = SimulationState()
            new_state.model = sim_state.model
            new_state.model_data_list = sim_state.model_data_list.copy()

            # Run multiple steps
            for _ in range(num_steps):
                # Step the model
                new_state.model.step()

                # Add data for this step
                df_row = new_state.model.datacollector.get_model_vars_dataframe().iloc[-1:].to_dict('records')[0]
                df_row['step'] = new_state.model.steps
                new_state.add_data_row(df_row)

            # Update the state once at the end
            set_sim_state(new_state)

    # Convert model_data_list to DataFrame for plotting
    def get_model_dataframe():
        if sim_state.model_data_list:
            return pd.DataFrame(sim_state.model_data_list)
        return pd.DataFrame()

    # Use solara's memoization to prevent rebuilding the dataframe on every render
    df = solara.use_memo(
        get_model_dataframe,
        [len(sim_state.model_data_list), sim_state.update_counter]
    )

    # Memoize visualization functions
    topic_space_viz = solara.use_memo(
        lambda: topic_space_visualization(sim_state.model),
        [sim_state.steps, sim_state.update_counter]
    )

    quadrant_dist_viz = solara.use_memo(
        lambda: quadrant_distribution_visualization(sim_state.model),
        [sim_state.steps, sim_state.update_counter]
    )

    satisfaction_viz = solara.use_memo(
        lambda: satisfaction_visualization(sim_state.model),
        [sim_state.steps, sim_state.update_counter]
    )

    population_viz = solara.use_memo(
        lambda: population_metrics_visualization(df),
        [len(df), sim_state.update_counter]
    )

    network_viz = solara.use_memo(
        lambda: network_visualization(sim_state.model),
        [sim_state.steps, sim_state.update_counter]
    )

    bot_types_viz = solara.use_memo(
        lambda: bot_types_visualization(sim_state.model),
        [sim_state.steps, sim_state.update_counter]
    )

    # Create the dashboard layout using simpler Solara layout
    with solara.Column():
        # Title
        solara.Title("Social Media Simulation with Quadrant-Based Topic Space")
        solara.Markdown("This simulation models social media platform dynamics in a 2D topic space with bots and human users.")

        # First row - Parameters and Controls
        with solara.Columns([1, 1, 2]):
            # Left column - Parameters
            with solara.Card("Simulation Parameters"):
                # Warning if parameters have changed
                if params_changed:
                    solara.Info("⚠️ Parameters have changed.", style={"color": "orange", "fontWeight": "bold"})
                    solara.Button(
                        label="Apply Changes",
                        on_click=create_new_model
                    )

                # Population parameters
                solara.Markdown("### Initial Population")
                solara.SliderInt(
                    label="Initial Humans",
                    min=50,
                    max=500,
                    step=10,
                    value=num_initial_humans,
                    on_value=lambda v: update_param(v, set_num_initial_humans)
                )

                solara.SliderInt(
                    label="Initial Bots",
                    min=10,
                    max=200,
                    step=5,
                    value=num_initial_bots,
                    on_value=lambda v: update_param(v, set_num_initial_bots)
                )

                # Growth rates
                solara.Markdown("### Growth Rates")
                solara.SliderFloat(
                    label="Human Creation Rate",
                    min=0,
                    max=10,
                    step=0.1,
                    value=human_creation_rate,
                    on_value=lambda v: update_param(v, set_human_creation_rate)
                )

                solara.SliderFloat(
                    label="Bot Creation Rate",
                    min=0,
                    max=10,
                    step=0.1,
                    value=bot_creation_rate,
                    on_value=lambda v: update_param(v, set_bot_creation_rate)
                )

                solara.SliderFloat(
                    label="Bot Ban Rate Multiplier",
                    min=0.1,
                    max=2.0,
                    step=0.1,
                    value=bot_ban_rate_multiplier,
                    on_value=lambda v: update_param(v, set_bot_ban_rate_multiplier)
                )

                # Network parameters (shortened section)
                solara.Markdown("### Other Parameters")
                # Seed input
                seed_text, set_seed_text = solara.use_state(str(seed))

                def on_seed_change(value):
                    set_seed_text(value)
                    try:
                        seed_value = int(value)
                        update_param(seed_value, set_seed)
                    except ValueError:
                        pass

                solara.InputText(
                    label="Random Seed",
                    value=str(seed),
                    on_value=on_seed_change
                )

            # Middle column - Simulation Control and Status
            with solara.Column():
                # Simulation control
                with solara.Card("Simulation Control"):
                    # Buttons in a row
                    with solara.Row():
                        solara.Button(
                            label="Step",
                            on_click=step
                        )

                        solara.Button(
                            label="Run 5 Steps",
                            on_click=lambda: run_multiple_steps(5),
                            style={"backgroundColor": "#2196F3", "marginLeft": "4px"}
                        )

                    # Second row of buttons
                    with solara.Row():
                        solara.Button(
                            label="Run 10 Steps",
                            on_click=lambda: run_multiple_steps(10),
                            style={"backgroundColor": "#9C27B0", "marginTop": "4px"}
                        )

                        solara.Button(
                            label="Run 50 Steps",
                            on_click=lambda: run_multiple_steps(50),
                            style={"backgroundColor": "#4CAF50", "marginTop": "4px", "marginLeft": "4px"}
                        )

                # Current state
                if sim_state.model:
                    with solara.Card("Current State"):
                        solara.Markdown(f"""
                        **Step:** {sim_state.steps}
                        
                        **Active Humans:** {sim_state.active_humans}
                        
                        **Active Bots:** {sim_state.active_bots}
                        
                        **Human:Bot Ratio:** {sim_state.active_humans / max(1, sim_state.active_bots):.2f}
                        
                        **Avg Satisfaction:** {sim_state.get_avg_human_satisfaction():.1f}
                        """)

                # Bot types distribution
                if sim_state.model:
                    with solara.Card("Bot Types"):
                        solara.FigureMatplotlib(bot_types_viz)

            # Right column - Topic Space Visualization
            with solara.Card("Topic Space"):
                solara.FigureMatplotlib(topic_space_viz)

        # Second row - Quadrant Distribution and Satisfaction
        with solara.Columns([1, 1]):
            # Left column - Quadrant Distribution
            with solara.Card("Quadrant Distribution"):
                solara.FigureMatplotlib(quadrant_dist_viz)

            # Right column - Satisfaction
            with solara.Card("Satisfaction Analysis"):
                solara.FigureMatplotlib(satisfaction_viz)

        # Third row - Population Metrics and Network
        with solara.Columns([1, 1]):
            # Left column - Population Metrics
            with solara.Card("Population Metrics"):
                solara.FigureMatplotlib(population_viz)

            # Right column - Network
            with solara.Card("Agent Network"):
                solara.FigureMatplotlib(network_viz)


@solara.component
def Page():
    """Main page component"""
    QuadrantDashboard()


# When running with `solara run quadrant_visualization.py`, this will be used
if __name__ == "__main__":
    # Let Solara CLI handle the running
    pass