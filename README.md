
# Social Media Simulation of Human Satisfaction with Bot Interaction

By Yinkan Chen, Andriy Sapeha, Hongji Tang

## A. Overview of the Phenomenon Modeled

This project simulates the effects of AI-driven bots on human user satisfaction in social media environments. As social media platforms become increasingly populated with bots, we observe a phenomenon known as "enshittification" - a decline in user experience, growing distrust, and eventual user disengagement. Our simulation explores how bots contribute to this decline by interacting with both human users and other bots, ultimately shaping platform dynamics in ways that can erode trust and satisfaction.

Our agent-based model features two primary agent types:
- **Human Agents**: Users with satisfaction levels, topic interests, and varying interaction patterns
- **Bot Agents**: Automated accounts of different types (spam, misinformation, astroturfing) with unique behavioral patterns

The simulation takes place in a 2D topic space with four quadrants representing different discussion areas:
- Q1 (0-0.5, 0-0.5): Tech/Business (Serious & Individual)
- Q2 (0-0.5, 0.5-1): Politics/News (Serious & Societal)
- Q3 (0.5-1, 0-0.5): Hobbies (Casual & Individual)
- Q4 (0.5-1, 0.5-1): Pop Culture (Casual & Societal)

Agents move through this space based on preferences and interact with each other, with different interaction types affecting human satisfaction. The model incorporates realistic distributions of both humans and bots across topic areas based on research findings, with bots being concentrated in Tech/Business (56%) and Politics/News (27%) quadrants, while humans favor Pop Culture (50%) and Hobbies (34%).

## B. How to Run the Simulation

### Requirements

This project requires Python 3.13 and the packages specified in requirements.txt.

### Installation

1. Clone the repository or navigate to the project directory:

   ```
   cd 4461-term-project/src
   ```

2. Install the required packages:
   ```
   pip3 install -r requirements.txt
   ```

### Running the Simulation

#### Option 1: Interactive Visualization (Recommended)

The Solara-based visualization provides an interactive dashboard with real-time charts and controls:

```
solara run visualization.py
```

This will start a web server and display a URL (typically http://localhost:8765). Open this URL in your web browser to interact with the simulation dashboard.

**Important:**
1. After changing parameters, click "Apply Changes" to update the simulation.
2. Use the step buttons (Step, Run 5 Steps, Run 10 Steps, Run 50 Steps) to advance the simulation.

The dashboard includes:
- Parameter controls (population, growth rates, interaction biases)
- Topic space visualization showing agent distribution
- Satisfaction histograms and trends
- Population metrics over time
- Network visualization
- Bot type distribution

#### Option 2: Headless Simulation with Data Collection

Run a non-interactive simulation and collect comprehensive data:

```
python run.py
```

This will execute a simulation with default parameters and save detailed results to the `results/` directory, including:
- Time series data in CSV format
- Network structures in GEXF format (viewable in tools like Gephi)
- Detailed agent snapshots
- Summary statistics

## C. Key Findings and Observations

Our simulation yielded several interesting observations about the relationship between bot presence and human satisfaction on social media platforms:

1. **Initial Impact**: The most significant drop in average user satisfaction occurs in the early stages of bot introduction. In our primary simulation, average satisfaction dropped from 90 to 67.8 during the first 10 steps, suggesting an initial shock effect.

2. **Stabilization Over Time**: Contrary to expectations, satisfaction levels tend to stabilize and even slightly increase over time despite growing bot populations. This suggests that more tolerant users remain on the platform while less tolerant users leave, creating a selection effect.

3. **Quadrant-Specific Effects**: Satisfaction levels vary significantly by topic area. Tech/Business and Politics/News quadrants, which have higher bot concentrations, consistently show lower average satisfaction compared to Hobbies and Pop Culture quadrants.

4. **Bot Type Impact**: Different bot types affect satisfaction differently. Misinformation and astroturfing bots, concentrated in the "serious" quadrants (Tech/Business and Politics/News), cause more negative impacts than spam bots typically found in casual areas.

5. **Super User Influence**: Super users (the 10% of human users who are most active) show higher resistance to negative bot interactions and play important roles in maintaining community structures.

6. **Network Formation**: The simulation reveals interesting network patterns where some users become isolated while others form dense connection clusters, sometimes bridging between quadrants.

While our model confirms that increased bot presence correlates with some decrease in satisfaction, the relationship is more complex than a simple negative correlation. The development of user tolerance, self-selection of more resilient users, and quadrant-specific effects all contribute to the overall dynamics of the system.
