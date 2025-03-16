"""
visualization.py - Solara visualization for social media simulation
using Mesa 3.1.4 and Solara 1.44.1 with batch step buttons
"""

import solara
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial

from model import SmallWorldNetworkModel


# Global figure management - clear all matplotlib figures when needed
def clear_all_figures():
    plt.close('all')


def network_visualization(model):
    """Creates a network visualization of the social media model or placeholder for step 0."""
    # Create a figure with a specific figure number to avoid duplicates
    fig, ax = plt.subplots(figsize=(5, 5))

    # If we're at step 0, show a placeholder message instead of the network
    if model.steps == 0:
        ax.text(0.5, 0.5, "Network visualization will appear after the first step.\n\n"
                          "Press apply changes if changing parameters.",
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        return fig

    # Otherwise, create the normal network visualization
    G = nx.Graph()

    # Get all active agents
    all_active_agents = [agent for agent in model.agents if agent.active]

    # Count agents by type and connection status
    all_humans = [a for a in all_active_agents if a.agent_type == 'human']
    all_bots = [a for a in all_active_agents if a.agent_type == 'bot']

    connected_humans = [a for a in all_humans if len(a.connections) > 0]
    connected_bots = [a for a in all_bots if len(a.connections) > 0]

    # Add nodes - but only for agents that have connections
    connected_agents = [agent for agent in all_active_agents if len(agent.connections) > 0]

    for agent in connected_agents:
        G.add_node(agent.unique_id,
                   agent_type=agent.agent_type,
                   satisfaction=getattr(agent, "satisfaction", 0))

    # Add edges from connections
    for agent in connected_agents:
        for connection_id in agent.connections:
            if G.has_node(connection_id):  # Make sure the connection exists
                G.add_edge(agent.unique_id, connection_id)

    # Position nodes using a layout algorithm
    pos = nx.spring_layout(G, seed=model.random.randint(0, 2 ** 32 - 1))

    # Get node colors based on agent type
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        agent_type = G.nodes[node]['agent_type']
        if agent_type == 'human':
            node_colors.append('blue')
            node_sizes.append(40)  # Smaller node size
        else:  # bot
            node_colors.append('red')
            node_sizes.append(40)  # Smaller node size

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, ax=ax)

    # Add legend with counts
    ax.plot([0], [0], 'o', color='blue',
            label=f'Connected Humans: {len(connected_humans)}/{len(all_humans)}')
    ax.plot([0], [0], 'o', color='red',
            label=f'Connected Bots: {len(connected_bots)}/{len(all_bots)}')
    ax.legend(fontsize=8)

    ax.set_title(f"Social Network (Step {model.steps})", fontsize=10)
    ax.axis('off')

    # Add tight layout to make better use of space
    plt.tight_layout()
    return fig


# Function to visualize satisfaction distribution
def satisfaction_histogram(model):
    """Creates a histogram of human satisfaction levels"""
    # Smaller figure size
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Get satisfaction values from active human agents
    satisfaction_values = [
        agent.satisfaction for agent in model.agents
        if getattr(agent, "agent_type", "") == "human" and agent.active
    ]

    if satisfaction_values:
        # Create histogram
        ax.hist(satisfaction_values, bins=10, range=(0, 100), alpha=0.7, color='green')
        ax.set_title(f"Human Satisfaction (Step {model.steps})", fontsize=10)
        ax.set_xlabel("Satisfaction Level", fontsize=8)
        ax.set_ylabel("Number of Humans", fontsize=8)
        ax.set_xlim(0, 100)

        # Set a fixed y-axis limit based on the number of active humans
        # This ensures the scale doesn't change between steps
        max_humans = model.active_humans
        # Use a slightly higher value to account for potential growth
        y_max = max(20, int(max_humans * 1.2))
        ax.set_ylim(0, y_max)

        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, "No active human agents",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=8)
        # Set default y-limit even when no agents
        ax.set_ylim(0, 20)

    # Add tight layout to make better use of space
    plt.tight_layout()
    return fig


# Create a function for population over time plot
def population_plot(df):
    """Creates a line plot showing population over time"""
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(df['step'], df['Active Humans'], label='Humans')
    ax.plot(df['step'], df['Active Bots'], label='Bots')
    ax.set_xlabel('Step', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    return fig


# Create a function for satisfaction over time plot
def satisfaction_plot(df):
    """Creates a line plot showing satisfaction over time"""
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(df['step'], df['Average Human Satisfaction'], color='green')
    ax.set_xlabel('Step', fontsize=8)
    ax.set_ylabel('Satisfaction Level', fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    return fig


# Create a class to hold simulation state for atomic updates
class SimulationState:
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


# Create a Solara dashboard component
@solara.component
def SocialMediaDashboard():
    """Main dashboard component for the social media simulation"""
    # Model parameters with state
    num_initial_humans, set_num_initial_humans = solara.use_state(200)
    num_initial_bots, set_num_initial_bots = solara.use_state(40)
    human_creation_rate, set_human_creation_rate = solara.use_state(1)
    bot_creation_rate, set_bot_creation_rate = solara.use_state(3)

    # Hidden parameters - not displayed in UI but used in model
    connection_rewiring_prob, set_connection_rewiring_prob = solara.use_state(0.1)
    topic_shift_frequency, set_topic_shift_frequency = solara.use_state(30)
    human_satisfaction_init, set_human_satisfaction_init = solara.use_state(100)

    # Network & Interactions parameters
    human_human_positive_bias, set_human_human_positive_bias = solara.use_state(0.65)
    human_bot_negative_bias, set_human_bot_negative_bias = solara.use_state(0.65)
    seed, set_seed = solara.use_state(42)

    # Flag to indicate parameters have changed
    params_changed, set_params_changed = solara.use_state(False)

    # Create a unified simulation state to prevent multiple rerenders
    sim_state, set_sim_state = solara.use_state(SimulationState())

    # Function to update a parameter and mark as changed
    def update_param(value, setter):
        setter(value)
        set_params_changed(True)

    # Function to create a new model with current parameters
    def create_new_model():
        print(f"\n=== Creating new model with parameters ===")
        print(f"Initial Humans: {num_initial_humans}")
        print(f"Initial Bots: {num_initial_bots}")
        print(f"Human Creation Rate: {human_creation_rate}")
        print(f"Bot Creation Rate: {bot_creation_rate}")
        print(f"Connection Rewiring: {connection_rewiring_prob}")
        print(f"Topic Shift Frequency: {topic_shift_frequency}")
        print(f"Human-Human Bias: {human_human_positive_bias}")
        print(f"Human-Bot Bias: {human_bot_negative_bias}")
        print(f"Initial Satisfaction: {human_satisfaction_init}")
        print(f"Seed: {seed}")

        new_model = SmallWorldNetworkModel(
            num_initial_humans=num_initial_humans,
            num_initial_bots=num_initial_bots,
            human_creation_rate=human_creation_rate,
            bot_creation_rate=bot_creation_rate,
            connection_rewiring_prob=connection_rewiring_prob,
            topic_shift_frequency=topic_shift_frequency,
            human_human_positive_bias=human_human_positive_bias,
            human_bot_negative_bias=human_bot_negative_bias,
            human_satisfaction_init=human_satisfaction_init,
            seed=seed
        )
        return new_model

    # Function to initialize the model
    def initialize_model():
        # Close all matplotlib figures
        clear_all_figures()

        # Create a new simulation state with a new model
        new_state = SimulationState()
        new_state.set_model(create_new_model())

        # Set the new simulation state
        set_sim_state(new_state)

        # Clear the params_changed flag
        set_params_changed(False)

    # Initialize the model if it's None
    if sim_state.model is None:
        initialize_model()

    # Function to run a single step
    def step():
        if sim_state.model:
            # Clear all figures
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
        # Make a copy of the current state so we only update once at the end
        if sim_state.model:
            # Clear figures once
            clear_all_figures()

            # Make a deep copy of the state
            new_state = SimulationState()
            new_state.model = sim_state.model  # This is a reference, not a copy
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

    # Convert model_data_list to DataFrame for plotting using memoization
    def get_model_dataframe():
        if sim_state.model_data_list:
            return pd.DataFrame(sim_state.model_data_list)
        return pd.DataFrame()

    # Use solara's memoization to prevent rebuilding the dataframe on every render
    df = solara.use_memo(
        get_model_dataframe,
        [len(sim_state.model_data_list), sim_state.update_counter]
    )

    # Memoize our visualization functions to prevent duplicate renders
    get_network_viz = solara.use_memo(
        lambda: network_visualization(sim_state.model) if sim_state.model else None,
        [sim_state.steps, sim_state.update_counter]
    )

    get_satisfaction_hist = solara.use_memo(
        lambda: satisfaction_histogram(sim_state.model) if sim_state.model else None,
        [sim_state.steps, sim_state.update_counter]
    )

    get_population_plot = solara.use_memo(
        lambda: population_plot(df) if not df.empty else None,
        [len(df), sim_state.update_counter]
    )

    get_satisfaction_plot = solara.use_memo(
        lambda: satisfaction_plot(df) if not df.empty else None,
        [len(df), sim_state.update_counter]
    )

    # Create the dashboard layout
    with solara.Column():
        # First row - Initial Parameters, Graph, and Histogram
        with solara.Row():
            # Initial parameters column (left)
            with solara.Column(classes=["w-1/4"]):
                with solara.Card(title="Initial Parameters"):
                    # Add a warning if parameters have changed
                    if params_changed:
                        with solara.Row():
                            solara.Text("Parameters have changed.")
                            solara.Button(
                                label="Apply Changes",
                                on_click=initialize_model
                            )

                    # Initial Population and Growth Rates
                    solara.Markdown("### Initial Population")
                    solara.Text(f"Initial Humans: {num_initial_humans}")
                    solara.SliderInt(
                        label="Initial Humans",
                        min=100,
                        max=500,
                        value=num_initial_humans,
                        on_value=lambda v: update_param(v, set_num_initial_humans)
                    )

                    solara.Text(f"Initial Bots: {num_initial_bots}")
                    solara.SliderInt(
                        label="Initial Bots",
                        min=20,
                        max=500,
                        value=num_initial_bots,
                        on_value=lambda v: update_param(v, set_num_initial_bots)
                    )

                    solara.Markdown("### Growth Rates")
                    solara.Text(f"Human Creation Rate: {human_creation_rate}")
                    solara.SliderFloat(
                        label="Human Creation Rate",
                        min=0,
                        max=10,
                        step=1,
                        value=human_creation_rate,
                        on_value=lambda v: update_param(v, set_human_creation_rate)
                    )

                    solara.Text(f"Bot Creation Rate: {bot_creation_rate}")
                    solara.SliderFloat(
                        label="Bot Creation Rate",
                        min=1,
                        max=10,
                        step=1,
                        value=bot_creation_rate,
                        on_value=lambda v: update_param(v, set_bot_creation_rate)
                    )

            # Social Network Graph (middle)
            with solara.Column(classes=["w-3/8"]):
                if sim_state.model and get_network_viz is not None:
                    solara.FigureMatplotlib(get_network_viz)

            # Satisfaction Histogram (right)
            with solara.Column(classes=["w-3/8"]):
                if sim_state.model and get_satisfaction_hist is not None:
                    solara.FigureMatplotlib(get_satisfaction_hist)

        # Second row - Network Parameters and Simulation Controls
        with solara.Row():
            # Network & Interactions plus remaining parameters - wider now (left)
            with solara.Column(classes=["w-3/4"]):
                with solara.Card(title="Network & Interactions"):
                    # More space for bias sliders by using 50% width columns
                    with solara.Row():
                        # Column 1 - Human-Human Positive Bias (wider)
                        with solara.Column(classes=["w-1/2"]):
                            # Text above slider
                            solara.Text(f"Human-Human Positive Bias: {human_human_positive_bias:.2f}")
                            # Slider with empty label
                            solara.SliderFloat(
                                label=" ",  # Empty space as label to satisfy the requirement
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=human_human_positive_bias,
                                on_value=lambda v: update_param(v, set_human_human_positive_bias)
                            )

                        # Column 2 - Human-Bot Negative Bias (wider)
                        with solara.Column(classes=["w-1/2"]):
                            # Text above slider
                            solara.Text(f"Human-Bot Negative Bias: {human_bot_negative_bias:.2f}")
                            # Slider with empty label
                            solara.SliderFloat(
                                label=" ",  # Empty space as label to satisfy the requirement
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=human_bot_negative_bias,
                                on_value=lambda v: update_param(v, set_human_bot_negative_bias)
                            )

                    # Seed input in a separate row for more space
                    with solara.Row(classes=["mt-4"]):
                        with solara.Column(classes=["w-full"]):
                            # Text above input
                            solara.Text("Random Seed:")

                            # Text field for seed value
                            seed_text, set_seed_text = solara.use_state(str(seed))

                            def on_seed_change(value):
                                set_seed_text(value)
                                try:
                                    # Try to convert to integer
                                    seed_value = int(value)
                                    update_param(seed_value, set_seed)
                                except ValueError:
                                    # If not a valid integer, don't update
                                    pass

                            # Input with empty label
                            solara.InputText(
                                label=" ",  # Empty space as label to satisfy the requirement
                                value=seed_text,
                                on_value=on_seed_change
                            )

            # Simulation Control and Current State - right side
            with solara.Column(classes=["w-1/4"]):
                # Enhanced simulation controls with batch step buttons
                with solara.Card(title="Simulation Control"):
                    with solara.Row():
                        # Regular Step button
                        solara.Button(
                            label="Step",
                            on_click=step,
                            classes=["mr-2"]
                        )

                        # Run button - run 10 steps at once
                        solara.Button(
                            label="Run 10 Steps",
                            on_click=lambda: run_multiple_steps(10),
                            style={"background": "#38a169"}
                        )

                    with solara.Row(classes=["mt-2"]):
                        # Run 50 steps button
                        solara.Button(
                            label="Run 50 Steps",
                            on_click=lambda: run_multiple_steps(50),
                            style={"background": "#2b6cb0"}
                        )

                        # Run 100 steps button
                        solara.Button(
                            label="Run 100 Steps",
                            on_click=lambda: run_multiple_steps(100),
                            style={"background": "#805ad5"},
                            classes=["ml-2"]
                        )

                # Current state display
                if sim_state.model:
                    with solara.Card(title="Current State"):
                        with solara.Row():
                            solara.Info(
                                f"Step: {sim_state.steps} | "
                                f"Active Humans: {sim_state.active_humans} | "
                                f"Active Bots: {sim_state.active_bots} | "
                                f"Avg Satisfaction: {sim_state.get_avg_human_satisfaction():.1f}"
                            )

        # Third row - Time series plots
        with solara.Row():
            with solara.Column(classes=["w-1/2"]):
                if sim_state.model and not df.empty and get_population_plot is not None:
                    # Line plots of key metrics using the dedicated function
                    with solara.Card(title="Population Over Time"):
                        solara.FigureMatplotlib(get_population_plot)

            with solara.Column(classes=["w-1/2"]):
                if sim_state.model and not df.empty and get_satisfaction_plot is not None:
                    # Satisfaction over time using the dedicated function
                    with solara.Card(title="Satisfaction Over Time"):
                        solara.FigureMatplotlib(get_satisfaction_plot)


# Main app
@solara.component
def Page():
    SocialMediaDashboard()


# When running with `solara run visualization.py`, this will be used
if __name__ == "__main__":
    # No need to call solara.run() as the CLI will handle that
    pass