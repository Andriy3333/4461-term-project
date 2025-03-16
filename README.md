# Social Media Simulation of Human Satisfaction with Bot Interaction

By Yinkan Chen, Andriy Sapeha, Hongji Tang

## A. Overview of Current Implementation

This project simulates the effects of bot interactions on human user satisfaction in social media environments. As social media platforms become increasingly populated with AI-driven bots, we observe a phenomenon known as "enshittification" - a decline in user experience, growing distrust, and eventual user disengagement. Our simulation explores how bots contribute to this decline by interacting with both human users and other bots, ultimately shaping platform dynamics in a way that erodes trust and satisfaction.

The current implementation features an agent-based model using Mesa 3.1.4 with:

1. **Two agent types:**

   - **Human Agents**: Users with satisfaction levels, topic interests, and interaction patterns
   - **Bot Agents**: Automated accounts of different types (spam, misinformation, astroturfing) that interact with humans and other bots

2. **Small-world network structure:**

   - Represents social connections between users
   - Periodically rewired to simulate changing trends and topics

3. **Interactive visualization:**

   - Built with Solara 1.44.1
   - Network graph visualization showing human and bot connections
   - Satisfaction histogram and trend analysis
   - Population dynamics visualization

4. **Core simulation mechanics:**
   - Human-to-human interactions (generally positive)
   - Human-to-bot interactions (generally negative)
   - Bot detection and removal system
   - Satisfaction-based user retention
   - Dynamic network evolution

In our current simulation, bots and humans interact in a network structure where humans have satisfaction levels that decrease upon negative interactions with bots. Each agent type has specific behavior patterns, and the simulation tracks various metrics over time, including average human satisfaction, active population counts, and network structure changes.

## B. How to Run the Simulation

### Requirements

This project requires Python 3.13 and the packages inside requirements.txt.

### Installation

1. Clone this repository:

   ```
   git clone https://github.com/Andriy3333/4461-term-project.git
   cd 4461-term-project/src
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip3 install -r requirements.txt
   ```

### Running the Simulation

There are two ways to run the simulation:

#### 1. Using the Solara-based visualization (recommended):

```
solara run visualization.py
```

This will start a web server, and you should see output with a URL (typically http://localhost:8765). Open this URL in your web browser to interact with the simulation dashboard.

IMPORTANT:

1. When changing parameters please click initialize after.
2. Only use the single step function to advance the simulation.

- Controls for setting simulation parameters
- Network visualization
- Satisfaction histogram
- Population over time charts

#### 2. Run and then check the data:

The sweep_config.json file defines the parameter combinations to explore:
{
"num_initial_humans": [400],
"num_initial_bots": [100],
"human_human_positive_bias": [0.7],
"human_bot_negative_bias": [0.7],
"human_creation_rate": [5.0],
"bot_creation_rate": [20],
"connection_rewiring_prob": [0.1],
"topic_shift_frequency": [30],
"human_satisfaction_init": [100]
}
You can modify this file to test different parameter values by adding more options to each array. For example, to test multiple bot creation rates:
"bot_creation_rate": [10, 20, 30]
This would run the simulation with all three bot creation rate values while keeping other parameters constant.
Running Sweeps
To run a parameter sweep from the terminal:

# Run with default settings

python run_sweep.py

# Run with custom configuration file

python run_sweep.py --config custom_sweep.json

# Run with custom steps and runs per configuration

python run_sweep.py --steps 500 --runs 5

# Run with custom output directory

python run_sweep.py --output my_results
The sweep results will be saved in the specified output directory (defaults to results/), including:

A timestamped subdirectory for each sweep (e.g., sweep_20250315_120000/)
Individual CSV files for each run with detailed time series data
A summary.csv file with the final state of all runs
A copy of the parameters.json used for the sweep

Analyzing Results
After running a sweep, you can analyze the results using your preferred data analysis tools. The summary CSV file contains key metrics from each run, making it easy to compare different parameter configurations.

### Parameter Explanation

- **Initial Population**: Starting number of human and bot agents
- **Growth Rates**: Rate at which new humans and bots are added per step
- **Connection Rewiring**: Probability of connections being rewired (higher values = more dynamic networks)
- **Topic Shift Frequency**: How often major topic shifts occur (in steps)
- **Human-Human Positive Bias**: Likelihood of positive interactions between humans
- **Human-Bot Negative Bias**: Likelihood of negative interactions between humans and bots
- **Initial Human Satisfaction**: Starting satisfaction level for human agents

## C. Limitations and Planned Improvements

### Current Limitations

1. **Echo Chamber Mechanism**: The echo chamber formation mechanism has been disabled in the current prototype as it was causing humans and bots to become isolated from each other, as noted by our developer: "It was ruining the simulation as no matter the parameters given, the humans would eventually move into smaller and smaller groups while the bots were left on the periphery."

2. **Parameter Calibration**: Finding the right balance between parameters has been challenging. As our developer noted: "Parameter tuning proved to be another time-consuming obstacle. Finding the right balance between human satisfaction decay rates, bot creation frequencies, and network rewiring probabilities took many iterations." The current bot growth rate appears too high, and the human growth rate too slow compared to real Twitter statistics.

3. **Specific Parameter Values**: While we have an idea of what ratios should be used, specific starting numbers haven't been fully determined yet, making it difficult to match the simulation to real-world data.

4. **Implementation Refinement**: Some systems like the bot banning mechanism don't appear to be fully implemented yet in the current prototype.

### Planned Improvements

1. **Parameter Tuning**: Adjust parameters to achieve a more realistic simulation where bot growth and human satisfaction decay are balanced appropriately. Parameters need to be based on confirmed Twitter statistics.

2. **Echo Chamber Mechanism**: Revisit the echo chamber system implementation to prevent the isolation issue, making it possible for bots and humans to meaningfully interact while still forming realistic community structures.

3. **Bot Ban System Implementation**: Properly implement the system for banning bots that display suspicious behavior, which currently appears to be missing in the prototype.

4. **Growth Rate Calibration**: Adjust human growth rate to better replicate Twitter's historical growth from approximately 200 million monthly active users in 2013 to 600 million in 2025, with nodes being proportional to those numbers.

5. **Visualization Enhancements**: Polish the finer details of the simulation and agent visualization for proper results and visual representation of social media networks.

6. **System Refinement**: Some systems may need more time to be fully realized or reimagined and implemented for proper results.
