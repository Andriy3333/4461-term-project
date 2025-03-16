"""
visualization.py - Solara visualization for social media simulation
using Mesa 3.1.4 and Solara 1.44.1
"""

import solara
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import threading
import io
import base64
from functools import partial

from model import SmallWorldNetworkModel


# Global figure management - clear all matplotlib figures when needed
def clear_all_figures():
    plt.close('all')


def network_visualization(model):
    """Creates a network visualization of the social media model"""
    # Create a figure with a specific figure number to avoid duplicates
    fig, ax = plt.subplots(figsize=(5, 5))
    G = nx.Graph()

    # Add nodes - but only for agents that have connections
    active_agents = [agent for agent in model.agents
                     if agent.active and len(agent.connections) > 0]

    for agent in active_agents:
        G.add_node(agent.unique_id,
                   agent_type=agent.agent_type,
                   satisfaction=getattr(agent, "satisfaction", 0))

    # Add edges from connections
    for agent in active_agents:
        for connection_id in agent.connections:
            if G.has_node(connection_id):  # Make sure the connection exists
                G.add_edge(agent.unique_id, connection_id)

    # Position nodes using a layout algorithm
    pos = nx.spring_layout(G, seed=model.random.randint(0, 2**32-1))

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

    # Add legend
    ax.plot([0], [0], 'o', color='blue', label='Human')
    ax.plot([0], [0], 'o', color='red', label='Bot')
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

    def add_data_rows(self, rows):
        self.model_data_list.extend(rows)
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
    num_initial_humans, set_num_initial_humans = solara.use_state(400)
    num_initial_bots, set_num_initial_bots = solara.use_state(100)
    human_creation_rate, set_human_creation_rate = solara.use_state(5)
    bot_creation_rate, set_bot_creation_rate = solara.use_state(20)
    connection_rewiring_prob, set_connection_rewiring_prob = solara.use_state(0.1)
    topic_shift_frequency, set_topic_shift_frequency = solara.use_state(30)
    human_human_positive_bias, set_human_human_positive_bias = solara.use_state(0.7)
    human_bot_negative_bias, set_human_bot_negative_bias = solara.use_state(0.8)
    human_satisfaction_init, set_human_satisfaction_init = solara.use_state(100)
    seed, set_seed = solara.use_state(42)

    # Flag to indicate parameters have changed
    params_changed, set_params_changed = solara.use_state(False)

    # Simulation control values - unified state
    is_running, set_is_running = solara.use_state(False)
    step_size, set_step_size = solara.use_state(1)

    # Create a unified simulation state to prevent multiple rerenders
    sim_state, set_sim_state = solara.use_state(SimulationState())

    # Debug counter to track renders - uncomment if needed
    # render_count, set_render_count = solara.use_state(0)
    # print(f"Dashboard rendering: {render_count}")
    # solara.use_effect(lambda: set_render_count(render_count + 1), [])

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
    def run_steps():
        if sim_state.model:
            # Clear all figures
            clear_all_figures()

            # Create a new state object to update atomically
            new_state = SimulationState()
            new_state.model = sim_state.model
            new_state.model_data_list = sim_state.model_data_list.copy()

            # Run all steps before updating state
            new_data_rows = []

            for _ in range(step_size):
                # Step the model
                new_state.model.step()

                # Collect data
                df_row = new_state.model.datacollector.get_model_vars_dataframe().iloc[-1:].to_dict('records')[0]
                df_row['step'] = new_state.model.steps
                new_data_rows.append(df_row)

            # Add all new data rows at once
            new_state.add_data_rows(new_data_rows)

            # Update the state just once
            set_sim_state(new_state)

    # Reset button function
    def reset():
        # Close all plots to ensure cleanup
        clear_all_figures()
        # Stop auto-stepping
        set_is_running(False)
        # Initialize a new model (which creates a new simulation state)
        initialize_model()

    # Background auto-stepping using a side effect
    def auto_step_effect():
        # Only set up auto-stepping when running is true
        if not is_running:
            return None

        # Use a list to hold the timer reference for proper cleanup
        timer_ref = [None]

        # Function to execute in a timer
        def timer_callback():
            # This runs in a background thread
            if is_running:
                # Create a flag to indicate that we need to update the UI
                run_steps_flag = [True]

                # We need to trigger a UI update from the main thread
                # Use an event to synchronize
                def trigger_update():
                    if run_steps_flag[0]:  # Only run if the flag is still True
                        run_steps()
                        run_steps_flag[0] = False  # Set the flag to False to avoid multiple runs

                # Schedule the update for the main thread
                solara.use_timer(0.1, trigger_update, once=True)

                # Schedule the next step after a delay
                timer_ref[0] = threading.Timer(1.0, timer_callback)
                timer_ref[0].start()

        # Start the timer
        timer_ref[0] = threading.Timer(1.0, timer_callback)
        timer_ref[0].start()

        # Return a proper cleanup function
        def cleanup():
            if timer_ref[0]:
                timer_ref[0].cancel()

        return cleanup

    # Set up the auto-stepping effect when is_running changes
    solara.use_effect(auto_step_effect, [is_running, step_size])

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
                        solara.Text("Parameters have changed. Click 'Initialize' to apply.")

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
                        min=0,
                        max=500,
                        value=num_initial_bots,
                        on_value=lambda v: update_param(v, set_num_initial_bots)
                    )

                    solara.Markdown("### Growth Rates")
                    solara.Text(f"Human Creation Rate: {human_creation_rate}")
                    solara.SliderFloat(
                        label="Human Creation Rate",
                        min=0,
                        max=15,
                        step=1,
                        value=human_creation_rate,
                        on_value=lambda v: update_param(v, set_human_creation_rate)
                    )

                    solara.Text(f"Bot Creation Rate: {bot_creation_rate}")
                    solara.SliderFloat(
                        label="Bot Creation Rate",
                        min=1,
                        max=30,
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
            # Network & Interactions - much wider now (left)
            with solara.Column(classes=["w-3/4"]):
                with solara.Card(title="Network & Interactions"):
                    with solara.Row():
                        # Column 1 of parameters
                        with solara.Column(classes=["w-1/3"]):
                            solara.Text(f"Connection Rewiring: {connection_rewiring_prob:.2f}")
                            solara.SliderFloat(
                                label="Connection Rewiring",
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=connection_rewiring_prob,
                                on_value=lambda v: update_param(v, set_connection_rewiring_prob)
                            )

                            solara.Text(f"Topic Shift Frequency: {topic_shift_frequency}")
                            solara.SliderInt(
                                label="Topic Shift Frequency",
                                min=1,
                                max=100,
                                value=topic_shift_frequency,
                                on_value=lambda v: update_param(v, set_topic_shift_frequency)
                            )

                        # Column 2 of parameters
                        with solara.Column(classes=["w-1/3"]):
                            solara.Text(f"Human-Human Positive Bias: {human_human_positive_bias:.2f}")
                            solara.SliderFloat(
                                label="Human-Human Positive Bias",
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=human_human_positive_bias,
                                on_value=lambda v: update_param(v, set_human_human_positive_bias)
                            )

                            solara.Text(f"Human-Bot Negative Bias: {human_bot_negative_bias:.2f}")
                            solara.SliderFloat(
                                label="Human-Bot Negative Bias",
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=human_bot_negative_bias,
                                on_value=lambda v: update_param(v, set_human_bot_negative_bias)
                            )

                        # Column 3 of parameters
                        with solara.Column(classes=["w-1/3"]):
                            solara.Text(f"Initial Human Satisfaction: {human_satisfaction_init}")
                            solara.SliderInt(
                                label="Initial Human Satisfaction",
                                min=0,
                                max=100,
                                value=human_satisfaction_init,
                                on_value=lambda v: update_param(v, set_human_satisfaction_init)
                            )

                            solara.Text(f"Random Seed: {seed}")
                            solara.SliderInt(
                                label="Random Seed",
                                min=0,
                                max=1000,
                                value=seed,
                                on_value=lambda v: update_param(v, set_seed)
                            )

            # Simulation Controls and Current State - moved far right
            with solara.Column(classes=["w-1/4"]):
                # Simulation controls
                with solara.Card(title="Simulation Controls"):
                    # First row of controls
                    with solara.Row():
                        with solara.Column(classes=["w-1/3"]):
                            solara.Button(
                                label="Initialize" if not params_changed else "Initialize (Apply Changes)",
                                on_click=reset
                            )
                        with solara.Column(classes=["w-1/3"]):
                            solara.Button(label="Step", on_click=step)
                        with solara.Column(classes=["w-1/3"]):
                            solara.Button(
                                label="Run" if not is_running else "Pause",
                                on_click=lambda: set_is_running(not is_running)
                            )

                    # Second row with steps slider
                    with solara.Row():
                        solara.Text(f"Steps per Click: {step_size}")
                        solara.SliderInt(
                            label="Steps per Click",
                            min=1,
                            max=30,
                            value=step_size,
                            on_value=lambda v: set_step_size(v)
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