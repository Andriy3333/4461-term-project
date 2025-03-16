"""
run_sweep.py - Simple script to run a parameter sweep and save results
"""

import os
import json
import pandas as pd
import time
import datetime
from model import SmallWorldNetworkModel


def run_sweep(config_file, output_dir="results", steps=365, runs_per_config=3):
    """
    Run a parameter sweep based on a config file

    Args:
        config_file: Path to JSON config file with parameter grid
        output_dir: Directory to save results
        steps: Number of steps to run each simulation
        runs_per_config: Number of runs per parameter configuration
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load parameter grid from config file
    with open(config_file, 'r') as f:
        parameter_grid = json.load(f)

    # Generate all parameter combinations
    import itertools

    # Get all parameter names and values
    param_names = list(parameter_grid.keys())
    param_values = list(itertools.product(*[parameter_grid[name] for name in param_names]))

    # Generate parameter dictionaries for all combinations
    all_params = []
    for values in param_values:
        params = {name: value for name, value in zip(param_names, values)}
        all_params.append(params)

    print(f"Running {len(all_params)} parameter configurations with {runs_per_config} runs each")
    print(f"Total runs: {len(all_params) * runs_per_config}")

    # Timestamp for this sweep
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(output_dir, f"sweep_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    # Save the parameter grid
    with open(os.path.join(sweep_dir, "parameters.json"), 'w') as f:
        json.dump(parameter_grid, f, indent=2)

    # Run all simulations and collect results
    all_results = []

    for config_idx, params in enumerate(all_params):
        print(f"\nConfiguration {config_idx + 1}/{len(all_params)}")
        print(f"Parameters: {params}")

        # Run multiple times with different seeds
        for run in range(runs_per_config):
            run_seed = config_idx * runs_per_config + run
            params_with_seed = params.copy()
            params_with_seed['seed'] = run_seed

            print(f"  Run {run + 1}/{runs_per_config} (Seed: {run_seed})")

            # Create model with these parameters
            model = SmallWorldNetworkModel(**params_with_seed)

            # Run for specified number of steps
            for i in range(steps):
                if i % 100 == 0:
                    print(f"    Step {i}/{steps}")
                model.step()

            # Get results from the datacollector
            model_data = model.datacollector.get_model_vars_dataframe()

            # Save the model data for this run
            run_id = f"config_{config_idx}_run_{run}"
            model_data.to_csv(os.path.join(sweep_dir, f"{run_id}.csv"))

            # Store final results
            final_results = {
                'config_idx': config_idx,
                'run': run,
                'run_id': run_id,
                'seed': run_seed,
                'active_humans': model.active_humans,
                'active_bots': model.active_bots,
                'avg_human_satisfaction': model.get_avg_human_satisfaction()
            }
            # Add parameters to results
            final_results.update(params)

            all_results.append(final_results)

    # Create a summary dataframe of all runs
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(sweep_dir, "summary.csv"), index=False)

    print(f"\nSweep complete! Results saved to {sweep_dir}")
    return sweep_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a parameter sweep")
    parser.add_argument('--config', type=str, default='sweep_config.json',
                        help="Path to parameter sweep config file")
    parser.add_argument('--steps', type=int, default=365,
                        help="Number of steps per simulation")
    parser.add_argument('--runs', type=int, default=3,
                        help="Number of runs per parameter configuration")
    parser.add_argument('--output', type=str, default='results',
                        help="Output directory")

    args = parser.parse_args()

    run_sweep(
        config_file=args.config,
        output_dir=args.output,
        steps=args.steps,
        runs_per_config=args.runs
    )