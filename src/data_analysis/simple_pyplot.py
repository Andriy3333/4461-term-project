"""
simple_pyplot.py - Direct matplotlib visualization for social media bot simulation data
No dependency on Solara - just matplotlib and pandas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def main():
    print("Loading data from paste.txt...")

    # Check if file exists
    if not os.path.exists('paste.txt'):
        print("Error: File 'paste.txt' not found.")
        return

    try:
        # Attempt to load the data
        df = pd.read_csv('paste.txt', sep='\t')
        print(f"Successfully loaded data with shape: {df.shape}")
        print(f"Columns found: {list(df.columns)}")

        # Create a figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Use step as x-axis if available, otherwise use index
        x_values = df.index
        if 'step' in df.columns:
            x_values = df['step']

        # Plot 1: Population data
        print("Creating population plot...")
        try:
            # Try to plot the expected columns
            active_humans_col = [col for col in df.columns if 'active' in col.lower() and 'human' in col.lower()]
            active_bots_col = [col for col in df.columns if 'active' in col.lower() and 'bot' in col.lower()]

            if active_humans_col and active_bots_col:
                ax1.plot(x_values, df[active_humans_col[0]], label='Active Humans', color='blue', linewidth=2)
                ax1.plot(x_values, df[active_bots_col[0]], label='Active Bots', color='red', linewidth=2)
            else:
                # Just plot the first two numeric columns if the expected columns aren't found
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) >= 2:
                    ax1.plot(x_values, df[numeric_cols[0]], label=numeric_cols[0], linewidth=2)
                    ax1.plot(x_values, df[numeric_cols[1]], label=numeric_cols[1], linewidth=2)

            ax1.set_title('Population Over Time', fontsize=14)
            ax1.set_ylabel('Count', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
        except Exception as e:
            print(f"Error creating population plot: {str(e)}")
            ax1.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax1.transAxes, fontsize=12)

        # Plot 2: Satisfaction data
        print("Creating satisfaction plot...")
        try:
            # Try to plot satisfaction
            satisfaction_col = [col for col in df.columns if 'satisf' in col.lower() or 'average' in col.lower()]

            if satisfaction_col:
                ax2.plot(x_values, df[satisfaction_col[0]], color='green', linewidth=2)
                ax2.set_title(f'{satisfaction_col[0]} Over Time', fontsize=14)
            else:
                # Use the last numeric column as a fallback
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) >= 3:
                    ax2.plot(x_values, df[numeric_cols[2]], color='green', linewidth=2)
                    ax2.set_title(f'{numeric_cols[2]} Over Time', fontsize=14)

            ax2.set_ylabel('Value', fontsize=12)
            ax2.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error creating satisfaction plot: {str(e)}")
            ax2.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes, fontsize=12)

        # Plot 3: Ratio plot
        print("Creating ratio plot...")
        try:
            # Try to calculate human-to-bot ratio
            active_humans_col = [col for col in df.columns if 'active' in col.lower() and 'human' in col.lower()]
            active_bots_col = [col for col in df.columns if 'active' in col.lower() and 'bot' in col.lower()]

            if active_humans_col and active_bots_col:
                ratio = df[active_humans_col[0]] / df[active_bots_col[0]]
                ax3.plot(x_values, ratio, color='purple', linewidth=2)
                ax3.set_title('Human-to-Bot Ratio Over Time', fontsize=14)
                ax3.set_ylabel('Ratio (Humans/Bots)', fontsize=12)
            else:
                # Plot a third metric as fallback
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) >= 4:
                    ax3.plot(x_values, df[numeric_cols[3]], color='purple', linewidth=2)
                    ax3.set_title(f'{numeric_cols[3]} Over Time', fontsize=14)
                    ax3.set_ylabel('Value', fontsize=12)

            ax3.set_xlabel('Simulation Step', fontsize=12)
            ax3.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error creating ratio plot: {str(e)}")
            ax3.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax3.transAxes, fontsize=12)

        # Finalize and save the figure
        plt.tight_layout()

        # Save the figure
        output_file = 'simulation_results.png'
        print(f"Saving visualization to {output_file}...")
        plt.savefig(output_file, dpi=300)

        print(f"Done! Visualization saved to {output_file}")

        # Show plot if not running in a headless environment
        plt.show()

    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main()