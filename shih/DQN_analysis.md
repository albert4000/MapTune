# Coding Plan: DQN Training Analysis Integration

## Phase 1: Environment and Library Setup
1. Add required libraries at the beginning of `batched_DQN.py` for data logging and visualization.
   - `import matplotlib.pyplot as plt`
   - `import os`
   - `import json`
2. Create a dynamic directory to store the analysis results based on the design name.
   - Extract design name from `design = sys.argv[-2]` (e.g., removing paths or extensions if necessary).
   - Create directory `log_dir = f"analysis_results_{design_name}"`.
   - `os.makedirs(log_dir, exist_ok=True)`

## Phase 2: Initialize Tracking Data Structures
1. Modify the `train_agent` function to include tracking variables before the episode loop starts.
   - `reward_history = []`
   - `delay_history = []`
   - `area_history = []`
   - `q_value_history = []`
   - `loss_history = []`
   - `action_distribution = []`

## Phase 3: Modify Agent to Expose Q-values and Loss
1. Update `DQNAgent.update_batch` to return the loss value.
   - Return `loss.item()`.
2. Update `DQNAgent.select_action` to return both the chosen action and its corresponding Q-value.

## Phase 4: Data Collection in Training Loop
1. Inside the `for episode in range(num_episodes):` loop:
   - Initialize `episode_q_values = []`, `episode_losses = []`, and `episode_actions = np.zeros(action_size)`.
2. Inside the `while not done:` loop:
   - Record the chosen `action` by incrementing `episode_actions[action] += 1`.
   - Append the retrieved Q-value to `episode_q_values`.
   - Append the returned loss from `agent.update_batch(batch)` into `episode_losses`.
3. At the end of each episode (when `done` is True):
   - Append the final `reward` to `reward_history`.
   - Append the final `delay` to `delay_history`.
   - Append the final `area` to `area_history`.
   - Calculate the mean of `episode_q_values` and append to `q_value_history`.
   - Calculate the mean of `episode_losses` (if not empty) and append to `loss_history`.
   - Append `episode_actions` to `action_distribution`.

## Phase 5: Data Visualization and Export
1. Implement `save_analysis_results(log_dir, ...)` function to be called after `train_agent`.
2. Learning Curve Plot (Reward):
   - Plot `reward_history` with a moving average.
   - Save as `f"{log_dir}/reward_curve.png"`.
3. Delay and Area Curves:
   - Create a plot with two subplots (one for `delay_history`, one for `area_history`).
   - Draw horizontal lines representing the baseline (`max_delay`, `max_area`) for comparison.
   - Save as `f"{log_dir}/delay_area_curves.png"`.
4. Q-value and Loss Convergence Plot:
   - Plot `q_value_history` and `loss_history` in subplots.
   - Save as `f"{log_dir}/convergence_metrics.png"`.
5. Action Distribution Analysis:
   - Save `action_distribution` as `f"{log_dir}/action_distribution.npy"`.
   - Generate a heatmap for the action distribution. Save as `f"{log_dir}/action_heatmap.png"`.
6. Raw Data Export:
   - Dump all history lists into `f"{log_dir}/training_data.json"`.