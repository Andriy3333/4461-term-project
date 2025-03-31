"""
run.py - Comprehensive data collection for the social media simulation
"""

import os
import csv
import json
import networkx as nx
from model import QuadrantTopicModel
import constants

# Create directory for results if it doesn't exist
os.makedirs('results', exist_ok=True)

def run_simulation(steps=50, seed=42):
    """Run a single simulation with comprehensive data collection"""
    print(f"Running simulation for {steps} steps with seed {seed}...")

    # Create results directory for this run
    run_dir = f'results/run_seed{seed}'
    os.makedirs(run_dir, exist_ok=True)

    # Create subdirectories for different data types
    os.makedirs(f'{run_dir}/steps', exist_ok=True)
    os.makedirs(f'{run_dir}/networks', exist_ok=True)
    os.makedirs(f'{run_dir}/agents', exist_ok=True)

    # Create model with default parameters
    model = QuadrantTopicModel(seed=seed)

    # Ensure data is collected for initial state
    model.datacollector.collect(model)

    # Create CSV file for comprehensive model-level time series data
    with open(f'{run_dir}/model_timeseries.csv', 'w', newline='') as f:
        writer = csv.writer(f)

        # Create comprehensive header row
        headers = [
            'Step',
            # Population counts
            'Active_Humans', 'Active_Bots', 'Deactivated_Humans', 'Deactivated_Bots',
            # Satisfaction
            'Avg_Human_Satisfaction', 'Human_Bot_Ratio',
            # Quadrant distribution - humans (counts)
            'Humans_Tech_Business', 'Humans_Politics_News', 'Humans_Hobbies', 'Humans_Pop_Culture',
            # Quadrant distribution - bots (counts)
            'Bots_Tech_Business', 'Bots_Politics_News', 'Bots_Hobbies', 'Bots_Pop_Culture',
            # Quadrant distribution - humans (percentages)
            'Humans_Tech_Business_Pct', 'Humans_Politics_News_Pct', 'Humans_Hobbies_Pct', 'Humans_Pop_Culture_Pct',
            # Quadrant distribution - bots (percentages)
            'Bots_Tech_Business_Pct', 'Bots_Politics_News_Pct', 'Bots_Hobbies_Pct', 'Bots_Pop_Culture_Pct',
            # Super users
            'Super_Users_Count', 'Super_Users_Pct',
            # Bot types
            'Spam_Bots', 'Misinfo_Bots', 'Astroturf_Bots',
            # Network metrics
            'Total_Connections', 'Avg_Connections_Per_Human', 'Avg_Connections_Per_Bot',
            'Human_Human_Connections', 'Human_Bot_Connections', 'Bot_Bot_Connections'
        ]
        writer.writerow(headers)

        # Process and save data for each step
        for step in range(steps + 1):  # +1 to include initial state
            if step > 0:
                # Run simulation step
                model.step()
                print(f"  Step {step}/{steps}. Humans: {model.active_humans}, Bots: {model.active_bots}")

            # Get quadrant distribution
            human_dist, bot_dist = model.get_agent_quadrant_distribution()

            # Calculate percentages
            total_humans = max(1, sum(human_dist.values()))  # Avoid division by zero
            total_bots = max(1, sum(bot_dist.values()))      # Avoid division by zero

            human_pct = {k: (v/total_humans*100) for k, v in human_dist.items()}
            bot_pct = {k: (v/total_bots*100) for k, v in bot_dist.items()}

            # Count super users
            super_users = sum(1 for agent in model.agents
                             if agent.active and
                             agent.agent_type == 'human' and
                             getattr(agent, 'is_super_user', False))

            super_user_pct = (super_users / max(1, model.active_humans)) * 100

            # Count bot types
            spam_bots = 0
            misinfo_bots = 0
            astroturf_bots = 0

            for agent in model.agents:
                if agent.active and agent.agent_type == 'bot':
                    bot_type = getattr(agent, 'bot_type', '')
                    if bot_type == 'spam':
                        spam_bots += 1
                    elif bot_type == 'misinformation':
                        misinfo_bots += 1
                    elif bot_type == 'astroturfing':
                        astroturf_bots += 1

            # Network metrics
            total_connections = 0
            human_human_connections = 0
            human_bot_connections = 0
            bot_bot_connections = 0
            human_connections = []
            bot_connections = []

            # Count connection types
            for agent in model.agents:
                if agent.active:
                    conn_count = len(agent.connections)
                    total_connections += conn_count

                    if agent.agent_type == 'human':
                        human_connections.append(conn_count)
                    else:
                        bot_connections.append(conn_count)

                    for conn_id in agent.connections:
                        other = model.get_agent_by_id(conn_id)
                        if other and other.active and other.unique_id > agent.unique_id:  # Count each connection only once
                            if agent.agent_type == 'human' and other.agent_type == 'human':
                                human_human_connections += 1
                            elif agent.agent_type == 'bot' and other.agent_type == 'bot':
                                bot_bot_connections += 1
                            else:
                                human_bot_connections += 1

            # Calculate averages
            avg_human_connections = sum(human_connections) / max(1, len(human_connections))
            avg_bot_connections = sum(bot_connections) / max(1, len(bot_connections))

            # Create row with all data for this step
            row = [
                step,
                # Population counts
                model.active_humans, model.active_bots, model.deactivated_humans, model.deactivated_bots,
                # Satisfaction
                model.get_avg_human_satisfaction(), model.active_humans / max(1, model.active_bots),
                # Quadrant distribution - humans (counts)
                human_dist['tech_business'], human_dist['politics_news'],
                human_dist['hobbies'], human_dist['pop_culture'],
                # Quadrant distribution - bots (counts)
                bot_dist['tech_business'], bot_dist['politics_news'],
                bot_dist['hobbies'], bot_dist['pop_culture'],
                # Quadrant distribution - humans (percentages)
                human_pct['tech_business'], human_pct['politics_news'],
                human_pct['hobbies'], human_pct['pop_culture'],
                # Quadrant distribution - bots (percentages)
                bot_pct['tech_business'], bot_pct['politics_news'],
                bot_pct['hobbies'], bot_pct['pop_culture'],
                # Super users
                super_users, super_user_pct,
                # Bot types
                spam_bots, misinfo_bots, astroturf_bots,
                # Network metrics
                total_connections // 2, avg_human_connections, avg_bot_connections,  # // 2 because we counted each twice
                human_human_connections, human_bot_connections, bot_bot_connections
            ]
            writer.writerow(row)

            # Save detailed agent data for this step
            save_agent_data(model, f'{run_dir}/agents/step_{step}.csv')

            # Save network structure for key steps (initial, middle, final, and every 10 steps)
            if step == 0 or step == steps or step % 10 == 0:
                save_network_data(model, f'{run_dir}/networks/network_step_{step}.gexf')

            # Save comprehensive step data snapshot
            save_step_snapshot(model, step, f'{run_dir}/steps/step_{step}.json')

    print(f"Simulation complete. Comprehensive results saved to '{run_dir}' directory.")
    return model

def save_agent_data(model, filename):
    """Save detailed data for all agents to a CSV file"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Create comprehensive header for agent data
        writer.writerow([
            'AgentID', 'Type', 'Active', 'Creation_Step', 'Deactivation_Step',
            'Satisfaction', 'Connections_Count', 'Quadrant',
            'Is_SuperUser', 'Bot_Type', 'Topic_X', 'Topic_Y',
            'Post_Frequency', 'Activeness', 'Irritability', 'Authenticity'
        ])

        # Write data for each agent
        for agent in model.agents:
            # Common properties
            agent_id = agent.unique_id
            agent_type = agent.agent_type
            active = agent.active
            creation_date = getattr(agent, 'creation_date', None)
            deactivation_date = getattr(agent, 'deactivation_date', None)

            # Try to extract creation/deactivation steps from dates
            creation_step = None
            deactivation_step = None
            if hasattr(agent, 'creation_date') and agent.creation_date:
                try:
                    creation_step = (agent.creation_date - model.steps).days
                except:
                    pass

            if hasattr(agent, 'deactivation_date') and agent.deactivation_date:
                try:
                    deactivation_step = (agent.deactivation_date - model.steps).days
                except:
                    pass

            # Type-specific properties
            if agent_type == 'human':
                satisfaction = getattr(agent, 'satisfaction', None)
                is_super_user = getattr(agent, 'is_super_user', False)
                bot_type = 'N/A'
                activeness = getattr(agent, 'activeness', None)
                irritability = getattr(agent, 'irritability', None)
                authenticity = getattr(agent, 'authenticity', None)
            else:  # bot
                satisfaction = 'N/A'
                is_super_user = 'N/A'
                bot_type = getattr(agent, 'bot_type', None)
                activeness = 'N/A'
                irritability = 'N/A'
                authenticity = 'N/A'

            # Common additional properties
            connections_count = len(getattr(agent, 'connections', []))
            quadrant = agent.get_current_quadrant()
            topic_x = agent.topic_position.get('x', None)
            topic_y = agent.topic_position.get('y', None)
            post_frequency = getattr(agent, 'post_frequency', None)

            # Write row for this agent
            writer.writerow([
                agent_id, agent_type, active, creation_step, deactivation_step,
                satisfaction, connections_count, quadrant,
                is_super_user, bot_type, topic_x, topic_y,
                post_frequency, activeness, irritability, authenticity
            ])

def save_network_data(model, filename):
    """Save network structure to a file in GEXF format for visualization in tools like Gephi"""
    # Create a networkx graph
    G = nx.Graph()

    # Add nodes (agents)
    for agent in model.agents:
        if agent.active:
            # Add node with attributes
            G.add_node(
                agent.unique_id,
                agent_type=agent.agent_type,
                quadrant=agent.get_current_quadrant(),
                topic_x=agent.topic_position.get('x', 0),
                topic_y=agent.topic_position.get('y', 0),
                satisfaction=getattr(agent, 'satisfaction', 0) if agent.agent_type == 'human' else 0,
                is_super_user=getattr(agent, 'is_super_user', False) if agent.agent_type == 'human' else False,
                bot_type=getattr(agent, 'bot_type', '') if agent.agent_type == 'bot' else ''
            )

    # Add edges (connections)
    for agent in model.agents:
        if agent.active:
            for conn_id in agent.connections:
                # Only add edge if the connected agent exists and is active
                other = model.get_agent_by_id(conn_id)
                if other and other.active:
                    # Add edge with weight
                    G.add_edge(agent.unique_id, conn_id, weight=1)

    # Write to file
    nx.write_gexf(G, filename)

def save_step_snapshot(model, step, filename):
    """Save a comprehensive snapshot of the model state at this step"""
    # Create a dictionary with all relevant model state
    snapshot = {
        'step': step,
        'population': {
            'active_humans': model.active_humans,
            'active_bots': model.active_bots,
            'deactivated_humans': model.deactivated_humans,
            'deactivated_bots': model.deactivated_bots,
            'avg_satisfaction': model.get_avg_human_satisfaction(),
            'human_bot_ratio': model.active_humans / max(1, model.active_bots)
        },
        'quadrant_distribution': {
            'human': model.get_agent_quadrant_distribution()[0],
            'bot': model.get_agent_quadrant_distribution()[1]
        },
        'parameters': {
            'human_human_positive_bias': model.human_human_positive_bias,
            'human_bot_negative_bias': model.human_bot_negative_bias,
            'bot_ban_rate_multiplier': model.bot_ban_rate_multiplier,
            'human_creation_rate': model.human_creation_rate,
            'bot_creation_rate': model.bot_creation_rate,
            'network_stability': model.network_stability
        },
        'agents': []
    }

    # Add data for each agent
    for agent in model.agents:
        agent_data = {
            'id': agent.unique_id,
            'type': agent.agent_type,
            'active': agent.active,
            'topic_position': agent.topic_position,
            'quadrant': agent.get_current_quadrant(),
            'connections': list(agent.connections)
        }

        # Add type-specific properties
        if agent.agent_type == 'human':
            agent_data.update({
                'satisfaction': getattr(agent, 'satisfaction', None),
                'is_super_user': getattr(agent, 'is_super_user', False),
                'activeness': getattr(agent, 'activeness', None),
                'irritability': getattr(agent, 'irritability', None),
                'authenticity': getattr(agent, 'authenticity', None)
            })
        else:  # bot
            agent_data.update({
                'bot_type': getattr(agent, 'bot_type', None),
                'detection_rate': getattr(agent, 'detection_rate', None),
                'malicious_post_rate': getattr(agent, 'malicious_post_rate', None)
            })

        snapshot['agents'].append(agent_data)

    # Write to file
    with open(filename, 'w') as f:
        json.dump(snapshot, f, indent=2)

def run_parameter_experiment(param_name, param_values, steps=50):
    """Run comprehensive experiments with different parameter values"""
    print(f"Running comprehensive experiment for parameter: {param_name}")
    print(f"Values: {param_values}")

    # Create a directory for this experiment
    exp_dir = f'results/{param_name}_experiment'
    os.makedirs(exp_dir, exist_ok=True)

    # Create a summary file for experiment
    with open(f'{exp_dir}/summary.csv', 'w', newline='') as summary_file:
        summary_writer = csv.writer(summary_file)

        # Write header row
        summary_writer.writerow([
            param_name,
            # Population
            'Final_Humans', 'Final_Bots', 'Final_Satisfaction', 'Human_Bot_Ratio',
            # Human distribution percentages
            'Tech_Business_Humans%', 'Politics_News_Humans%', 'Hobbies_Humans%', 'Pop_Culture_Humans%',
            # Bot distribution percentages
            'Tech_Business_Bots%', 'Politics_News_Bots%', 'Hobbies_Bots%', 'Pop_Culture_Bots%',
            # Bot types
            'Spam_Bots%', 'Misinfo_Bots%', 'Astroturf_Bots%',
            # Network metrics
            'Avg_Connections_Per_Human', 'Human_Human_Connections%', 'Human_Bot_Connections%'
        ])

        # Run a simulation for each parameter value
        for i, value in enumerate(param_values):
            print(f"\nRun {i+1}/{len(param_values)}: {param_name}={value}")

            # Create run directory for this parameter value
            value_dir = f'{exp_dir}/value_{value}'
            os.makedirs(value_dir, exist_ok=True)
            os.makedirs(f'{value_dir}/steps', exist_ok=True)

            # Create params dictionary with just the parameter we're varying
            params = {param_name: value, 'seed': 42}

            # Create and run model
            model = QuadrantTopicModel(**params)

            # Ensure data is collected for initial state
            model.datacollector.collect(model)

            # Save initial state
            save_step_snapshot(model, 0, f'{value_dir}/steps/step_0.json')

            # Create CSV file for time series data
            with open(f'{value_dir}/timeseries.csv', 'w', newline='') as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow([
                    'Step', 'Active_Humans', 'Active_Bots', 'Avg_Satisfaction',
                    'Human_Bot_Ratio', 'Tech_Business_Humans%', 'Politics_News_Humans%',
                    'Hobbies_Humans%', 'Pop_Culture_Humans%'
                ])

                # Write initial state
                human_dist, _ = model.get_agent_quadrant_distribution()
                total_humans = max(1, sum(human_dist.values()))
                human_pct = {k: (v/total_humans*100) for k, v in human_dist.items()}

                writer.writerow([
                    0, model.active_humans, model.active_bots,
                    model.get_avg_human_satisfaction(),
                    model.active_humans / max(1, model.active_bots),
                    human_pct['tech_business'], human_pct['politics_news'],
                    human_pct['hobbies'], human_pct['pop_culture']
                ])

                # Run for specified steps
                for step in range(steps):
                    model.step()

                    # Get updated distribution
                    human_dist, _ = model.get_agent_quadrant_distribution()
                    total_humans = max(1, sum(human_dist.values()))
                    human_pct = {k: (v/total_humans*100) for k, v in human_dist.items()}

                    # Write state after each step
                    writer.writerow([
                        step+1, model.active_humans, model.active_bots,
                        model.get_avg_human_satisfaction(),
                        model.active_humans / max(1, model.active_bots),
                        human_pct['tech_business'], human_pct['politics_news'],
                        human_pct['hobbies'], human_pct['pop_culture']
                    ])

                    # Save state snapshots at key steps
                    if (step+1) % 10 == 0 or step == steps-1:
                        save_step_snapshot(model, step+1, f'{value_dir}/steps/step_{step+1}.json')

                    # Print progress
                    if (step+1) % 10 == 0 or step == steps-1:
                        print(f"  Step {step+1}/{steps}. Humans: {model.active_humans}, Bots: {model.active_bots}")

            # Calculate final metrics for summary
            human_dist, bot_dist = model.get_agent_quadrant_distribution()

            # Calculate percentages
            total_humans = max(1, sum(human_dist.values()))
            total_bots = max(1, sum(bot_dist.values()))

            human_pct = {k: (v/total_humans*100) for k, v in human_dist.items()}
            bot_pct = {k: (v/total_bots*100) for k, v in bot_dist.items()}

            # Count bot types
            spam_bots = 0
            misinfo_bots = 0
            astroturf_bots = 0

            for agent in model.agents:
                if agent.active and agent.agent_type == 'bot':
                    bot_type = getattr(agent, 'bot_type', '')
                    if bot_type == 'spam':
                        spam_bots += 1
                    elif bot_type == 'misinformation':
                        misinfo_bots += 1
                    elif bot_type == 'astroturfing':
                        astroturf_bots += 1

            # Calculate bot type percentages
            spam_pct = (spam_bots / max(1, model.active_bots)) * 100
            misinfo_pct = (misinfo_bots / max(1, model.active_bots)) * 100
            astroturf_pct = (astroturf_bots / max(1, model.active_bots)) * 100

            # Calculate network metrics
            human_connections = []
            human_human_conn = 0
            human_bot_conn = 0
            total_conn = 0

            for agent in model.agents:
                if agent.active:
                    conn_count = len(agent.connections)
                    total_conn += conn_count

                    if agent.agent_type == 'human':
                        human_connections.append(conn_count)

                        for conn_id in agent.connections:
                            other = model.get_agent_by_id(conn_id)
                            if other and other.active:
                                if other.agent_type == 'human':
                                    human_human_conn += 1
                                else:
                                    human_bot_conn += 1

            # Calculate averages and percentages
            avg_human_conn = sum(human_connections) / max(1, len(human_connections))
            human_human_pct = (human_human_conn / max(1, total_conn)) * 100
            human_bot_pct = (human_bot_conn / max(1, total_conn)) * 100

            # Write summary data
            summary_writer.writerow([
                value,
                # Population
                model.active_humans, model.active_bots,
                model.get_avg_human_satisfaction(),
                model.active_humans / max(1, model.active_bots),
                # Human distribution percentages
                human_pct['tech_business'], human_pct['politics_news'],
                human_pct['hobbies'], human_pct['pop_culture'],
                # Bot distribution percentages
                bot_pct['tech_business'], bot_pct['politics_news'],
                bot_pct['hobbies'], bot_pct['pop_culture'],
                # Bot types
                spam_pct, misinfo_pct, astroturf_pct,
                # Network metrics
                avg_human_conn, human_human_pct, human_bot_pct
            ])

    print(f"\nExperiment complete. Comprehensive results saved to '{exp_dir}' directory.")


if __name__ == "__main__":
    # Run three separate simulations with different forced_feed_probability values
    for feed_prob in [0.65, 0.7, 0.75]:
        print(f"\nRunning simulation with forced_feed_probability = {feed_prob}")

        # Create a descriptive directory name that clearly indicates the parameter value
        run_dir = f'results/feed_prob_{feed_prob}'
        os.makedirs(run_dir, exist_ok=True)

        # Create subdirectories for different data types
        os.makedirs(f'{run_dir}/steps', exist_ok=True)
        os.makedirs(f'{run_dir}/networks', exist_ok=True)
        os.makedirs(f'{run_dir}/agents', exist_ok=True)

        # Create model with current forced_feed_probability value
        model = QuadrantTopicModel(forced_feed_probability=feed_prob, seed=42)

        # Ensure data is collected for initial state
        model.datacollector.collect(model)

        # Create CSV file for comprehensive model-level time series data
        with open(f'{run_dir}/model_timeseries.csv', 'w', newline='') as f:
            writer = csv.writer(f)

            # Create comprehensive header row (same as in run_simulation function)
            headers = [
                'Step',
                # Population counts
                'Active_Humans', 'Active_Bots', 'Deactivated_Humans', 'Deactivated_Bots',
                # Satisfaction
                'Avg_Human_Satisfaction', 'Human_Bot_Ratio',
                # Quadrant distribution - humans (counts)
                'Humans_Tech_Business', 'Humans_Politics_News', 'Humans_Hobbies', 'Humans_Pop_Culture',
                # Quadrant distribution - bots (counts)
                'Bots_Tech_Business', 'Bots_Politics_News', 'Bots_Hobbies', 'Bots_Pop_Culture',
                # Quadrant distribution - humans (percentages)
                'Humans_Tech_Business_Pct', 'Humans_Politics_News_Pct', 'Humans_Hobbies_Pct', 'Humans_Pop_Culture_Pct',
                # Quadrant distribution - bots (percentages)
                'Bots_Tech_Business_Pct', 'Bots_Politics_News_Pct', 'Bots_Hobbies_Pct', 'Bots_Pop_Culture_Pct',
                # Super users
                'Super_Users_Count', 'Super_Users_Pct',
                # Bot types
                'Spam_Bots', 'Misinfo_Bots', 'Astroturf_Bots',
                # Network metrics
                'Total_Connections', 'Avg_Connections_Per_Human', 'Avg_Connections_Per_Bot',
                'Human_Human_Connections', 'Human_Bot_Connections', 'Bot_Bot_Connections'
            ]
            writer.writerow(headers)

            # Process and save data for each step (initial state + 50 steps)
            for step in range(51):  # 0 to 50 inclusive
                # For steps after the initial state, run simulation
                if step > 0:
                    model.step()
                    print(f"  Step {step}/50. Humans: {model.active_humans}, Bots: {model.active_bots}")

                # Get quadrant distribution
                human_dist, bot_dist = model.get_agent_quadrant_distribution()

                # Calculate percentages
                total_humans = max(1, sum(human_dist.values()))
                total_bots = max(1, sum(bot_dist.values()))

                human_pct = {k: (v / total_humans * 100) for k, v in human_dist.items()}
                bot_pct = {k: (v / total_bots * 100) for k, v in bot_dist.items()}

                # Count super users
                super_users = sum(1 for agent in model.agents
                                  if agent.active and
                                  agent.agent_type == 'human' and
                                  getattr(agent, 'is_super_user', False))

                super_user_pct = (super_users / max(1, model.active_humans)) * 100

                # Count bot types
                spam_bots = 0
                misinfo_bots = 0
                astroturf_bots = 0

                for agent in model.agents:
                    if agent.active and agent.agent_type == 'bot':
                        bot_type = getattr(agent, 'bot_type', '')
                        if bot_type == 'spam':
                            spam_bots += 1
                        elif bot_type == 'misinformation':
                            misinfo_bots += 1
                        elif bot_type == 'astroturfing':
                            astroturf_bots += 1

                # Network metrics
                total_connections = 0
                human_human_connections = 0
                human_bot_connections = 0
                bot_bot_connections = 0
                human_connections = []
                bot_connections = []

                # Count connection types
                for agent in model.agents:
                    if agent.active:
                        conn_count = len(agent.connections)
                        total_connections += conn_count

                        if agent.agent_type == 'human':
                            human_connections.append(conn_count)
                        else:
                            bot_connections.append(conn_count)

                        for conn_id in agent.connections:
                            other = model.get_agent_by_id(conn_id)
                            if other and other.active and other.unique_id > agent.unique_id:  # Count each connection only once
                                if agent.agent_type == 'human' and other.agent_type == 'human':
                                    human_human_connections += 1
                                elif agent.agent_type == 'bot' and other.agent_type == 'bot':
                                    bot_bot_connections += 1
                                else:
                                    human_bot_connections += 1

                # Calculate averages
                avg_human_connections = sum(human_connections) / max(1, len(human_connections))
                avg_bot_connections = sum(bot_connections) / max(1, len(bot_connections))

                # Create row with all data for this step
                row = [
                    step,
                    # Population counts
                    model.active_humans, model.active_bots, model.deactivated_humans, model.deactivated_bots,
                    # Satisfaction
                    model.get_avg_human_satisfaction(), model.active_humans / max(1, model.active_bots),
                    # Quadrant distribution - humans (counts)
                    human_dist['tech_business'], human_dist['politics_news'],
                    human_dist['hobbies'], human_dist['pop_culture'],
                    # Quadrant distribution - bots (counts)
                    bot_dist['tech_business'], bot_dist['politics_news'],
                    bot_dist['hobbies'], bot_dist['pop_culture'],
                    # Quadrant distribution - humans (percentages)
                    human_pct['tech_business'], human_pct['politics_news'],
                    human_pct['hobbies'], human_pct['pop_culture'],
                    # Quadrant distribution - bots (percentages)
                    bot_pct['tech_business'], bot_pct['politics_news'],
                    bot_pct['hobbies'], bot_pct['pop_culture'],
                    # Super users
                    super_users, super_user_pct,
                    # Bot types
                    spam_bots, misinfo_bots, astroturf_bots,
                    # Network metrics
                                                        total_connections // 2, avg_human_connections,
                    avg_bot_connections,
                    human_human_connections, human_bot_connections, bot_bot_connections
                ]
                writer.writerow(row)

                # Save detailed agent data for this step
                save_agent_data(model, f'{run_dir}/agents/step_{step}.csv')

                # Save network structure for key steps
                if step == 0 or step == 50 or step % 10 == 0:
                    save_network_data(model, f'{run_dir}/networks/network_step_{step}.gexf')

                # Save comprehensive step data snapshot
                save_step_snapshot(model, step, f'{run_dir}/steps/step_{step}.json')

        print(f"Simulation for forced_feed_probability={feed_prob} complete. Results saved to '{run_dir}' directory.")

    print("\nAll simulations complete.")