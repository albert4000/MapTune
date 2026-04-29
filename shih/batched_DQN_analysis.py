import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import re
import sys
import random
from collections import deque
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import json
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class GateSelectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, genlib_origin, lib_path, design, total_gates, sample_gate, max_delay, max_area):
        super().__init__()
        self.genlib_origin = genlib_origin
        self.lib_path = lib_path
        self.design = design
        self.total_gates = total_gates
        self.sample_gate = sample_gate
        self.max_delay = max_delay
        self.max_area = max_area
        self.action_space = spaces.Discrete(self.total_gates)
        self.observation_space = spaces.MultiBinary(self.total_gates)
        self.state = np.zeros(self.total_gates, dtype=int)
        self.selection_count = 0

    def step(self, action):
        if self.state[action] == 0 and self.selection_count < self.sample_gate:
            self.state[action] = 1
            self.selection_count += 1

        done = self.selection_count == self.sample_gate
        reward = 0
        delay = 1
        area = 1
        next_state = self.state.copy()

        if done:
            delay, area = self.technology_mapper(list(np.where(self.state == 1)[0]))
            reward = self.calculate_reward(delay, area)
            next_state = self.reset()
        return next_state, reward, done, delay, area

    def technology_mapper(self, partial_cell_library):
        with open(self.genlib_origin, 'r') as f:
            f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
        
        with open(self.genlib_origin, 'r') as f:
            f_keep = [line.strip() for line in f if line.startswith("GATE BUF") or line.startswith("GATE INV") or line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]

        lines_partial = [f_lines[i] for i in partial_cell_library] + f_keep
        design_basename = os.path.basename(self.design)
        
        # Add Process ID (PID) to avoid conflicts during parallel execution
        pid = os.getpid()
        output_genlib_file = f"{self.lib_path}{design_basename}_{pid}_{len(lines_partial)}_dqn_samplelib.genlib"
        lib_origin = self.genlib_origin[:-7] + '.lib'
        temp_blif = f"temp_blifs/{design_basename}_{pid}_dqn_temp.blif"
        
        with open(output_genlib_file, 'w') as out_gen:
            for line in lines_partial:
                out_gen.write(line + '\n')

        abc_cmd = f"abc -c 'read {output_genlib_file}; read {self.design}; map -a; write {temp_blif}; read {lib_origin}; read -m {temp_blif}; ps; topo; upsize; dnsize; stime;'"
        try:
            res = subprocess.check_output(abc_cmd, shell=True, text=True)
            match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", res)
            match_a = re.search(r"Area\s*=\s*([\d.]+)", res)
            delay = float(match_d.group(1)) if match_d else float('inf')
            area = float(match_a.group(1)) if match_a else float('inf')
        except subprocess.CalledProcessError as e:
            print("Failed to execute ABC:", e)
            delay, area = float('inf'), float('inf')

        # Clean up temporary genlib to save disk space during parallel runs
        if os.path.exists(output_genlib_file):
            os.remove(output_genlib_file)

        return delay, area
    
    def save_best_genlib(self, partial_cell_library, output_filename):
        with open(self.genlib_origin, 'r') as f:
            f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
        
        with open(self.genlib_origin, 'r') as f:
            f_keep = [line.strip() for line in f if line.startswith("GATE BUF") or line.startswith("GATE INV") or line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]

        lines_partial = [f_lines[i] for i in partial_cell_library] + f_keep
        
        with open(output_filename, 'w') as out_gen:
            for line in lines_partial:
                out_gen.write(line + '\n')

    def calculate_reward(self, delay, area):
        if delay == float('inf') or area == float('inf'):
            return float('-inf')
        
        normalized_delay = delay / self.max_delay
        normalized_area = area / self.max_area
        return -np.sqrt(normalized_delay * normalized_area)

    def reset(self):
        self.state = np.zeros(self.total_gates, dtype=int)
        self.selection_count = 0
        return self.state

    def render(self, mode='human'):
        print(f"Selected Gates: {np.where(self.state == 1)[0]}")

    def close(self):
        pass

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = 0.99

    def select_action(self, state, epsilon=0.2):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            max_q_value = q_values.max().item()

        if np.random.rand() < epsilon:
            action = np.random.randint(0, len(state))
        else:
            action = q_values.argmax().item()
            
        return action, max_q_value

    def update_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        current_qs = self.model(states).gather(1, actions).squeeze(1)
        next_qs = self.model(next_states).max(1)[0]
        expected_qs = rewards + self.gamma * (1 - dones) * next_qs

        loss = F.mse_loss(current_qs, expected_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# 更改函數定義，增加 best_actions 參數
def save_analysis_results(log_dir, results, baseline_delay, baseline_area, best_metrics, best_actions):
    json_data = {
        'reward_history': results['reward'],
        'delay_history': results['delay'],
        'area_history': results['area'],
        'q_value_history': results['q_value'],
        'loss_history': results['loss']
    }
    with open(os.path.join(log_dir, 'training_data.json'), 'w') as f:
        json.dump(json_data, f)
        
    np.save(os.path.join(log_dir, 'action_distribution.npy'), np.array(results['action_dist']))

    # 輸出 Summary Report
    baseline_adp = baseline_delay * baseline_area
    report_path = os.path.join(log_dir, 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write("=== Design Optimization Summary Report ===\n")
        f.write(f"Baseline Delay: {baseline_delay:.2f} ps\n")
        f.write(f"Baseline Area: {baseline_area:.2f}\n")
        f.write(f"Baseline ADP: {baseline_adp:.2f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Delay Found: {best_metrics['best_delay']:.2f} ps\n")
        f.write(f"Best Area Found: {best_metrics['best_area']:.2f}\n")
        f.write(f"Best ADP Found: {best_metrics['best_adp']:.2f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Selected Gate Indices: {sorted(list(best_actions))}\n")

    def moving_average(a, n=50):
        if len(a) < n:
            return np.array(a)
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    # --- 1. Reward Curve Plot ---
    plt.figure()
    plt.plot(results['reward'], alpha=0.3, label='Raw Reward')
    
    ma_reward = moving_average(results['reward'])
    if len(results['reward']) >= 50:
        plt.plot(range(49, len(results['reward'])), ma_reward, color='red', label='Moving Average (50)')
    
    # rewards_arr = np.array(results['reward'])
    # top_5_indices = rewards_arr.argsort()[-5:][::-1]
    
    # plt.scatter(top_5_indices, rewards_arr[top_5_indices], color='green', zorder=5, label='Top 5 Rewards')
    
    # max_idx = top_5_indices[0]
    # max_val = rewards_arr[max_idx]
    # y_offset = abs(max_val) * 0.05 if max_val != 0 else 1.0
    # plt.annotate(f'Max: {max_val:.2f}',
    #              xy=(max_idx, max_val),
    #              xytext=(max_idx, max_val + y_offset),
    #              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
    #              horizontalalignment='center')

    plt.title('Learning Curve (Reward)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'reward_curve.png'))
    plt.close()

    # --- 2. Delay, Area, and ADP Curves ---
    # 計算歷史 ADP
    adp_history = [d * a for d, a in zip(results['delay'], results['area'])]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    ax1.plot(results['delay'], alpha=0.5, label='Delay')
    ax1.axhline(y=baseline_delay, color='r', linestyle='--', label='Baseline Delay')
    ax1.set_title('Delay per Episode')
    ax1.set_ylabel('Delay (ps)')
    ax1.legend()

    ax2.plot(results['area'], alpha=0.5, label='Area')
    ax2.axhline(y=baseline_area, color='r', linestyle='--', label='Baseline Area')
    ax2.set_title('Area per Episode')
    ax2.set_ylabel('Area')
    ax2.legend()

    ax3.plot(adp_history, alpha=0.5, label='ADP')
    ax3.axhline(y=baseline_adp, color='r', linestyle='--', label='Baseline ADP')
    ax3.set_title('Area-Delay Product (ADP) per Episode')
    ax3.set_ylabel('ADP')
    ax3.set_xlabel('Episode')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'delay_area_curves.png'))
    plt.close()

    # --- 3. Convergence Metrics ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(results['q_value'])
    ax1.set_title('Average Q-Value per Episode')
    ax1.set_ylabel('Q-Value')

    ax2.plot(results['loss'])
    ax2.set_title('Average Loss per Episode')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Episode')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'convergence_metrics.png'))
    plt.close()

    # --- 4. Action Distribution Heatmap ---
    plt.figure(figsize=(30, 20))
    plt.imshow(np.array(results['action_dist']).T, aspect='auto', cmap='viridis', origin='lower', vmax=6)
    plt.colorbar(label='Selection Frequency')
    plt.title('Action Selection Distribution over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Gate Index')
    plt.savefig(os.path.join(log_dir, 'action_heatmap.png'))
    plt.close()

def train_agent(num_episodes, agent, env, batch_size, buffer_size, log_dir, baseline_delay, baseline_area):
    replay_buffer = deque(maxlen=buffer_size)
    highest_reward = float('-inf')

    # 紀錄最佳數值
    best_metrics = {
        'best_delay': float('inf'),
        'best_area': float('inf'),
        'best_adp': float('inf')
    }
    
    # 新增一個變數來儲存最佳的選擇組合
    global_best_actions = set()

    tracking_data = {
        'reward': [], 'delay': [], 'area': [],
        'q_value': [], 'loss': [], 'action_dist': []
    }

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        episode_q_values = []
        episode_losses = []
        selected_actions_this_episode = set()

        while not done:
            action, q_value = agent.select_action(state)
            selected_actions_this_episode.add(action)
            episode_q_values.append(q_value)

            next_state, reward, done, delay, area = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                # 排除 mapping 失敗的情況並更新最佳紀錄
                if delay != float('inf') and area != float('inf'):
                    adp = delay * area
                    if delay < best_metrics['best_delay']:
                        best_metrics['best_delay'] = delay
                    if area < best_metrics['best_area']:
                        best_metrics['best_area'] = area
                    if adp < best_metrics['best_adp']:
                        best_metrics['best_adp'] = adp

                # 當發現更高的 Reward 時，儲存這個組合並輸出 genlib
                if reward > highest_reward:
                    highest_reward = reward
                    best_result = (delay, area)
                    global_best_actions = selected_actions_this_episode.copy()
                    print('Current Best Result: ', best_result)
                    
                    # 將最佳的元件庫輸出到分析資料夾中
                    best_genlib_path = os.path.join(log_dir, 'best_solution.genlib')
                    env.save_best_genlib(list(global_best_actions), best_genlib_path)

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                loss = agent.update_batch(batch)
                episode_losses.append(loss)

        # 確保在 episode 結束後，頻率只會紀錄 1 次
        episode_actions = np.zeros(env.action_space.n)
        for act in selected_actions_this_episode:
            episode_actions[act] = 1

        plot_delay = delay if delay != float('inf') else baseline_delay * 1.5
        plot_area = area if area != float('inf') else baseline_area * 1.5

        tracking_data['reward'].append(reward)
        tracking_data['delay'].append(plot_delay)
        tracking_data['area'].append(plot_area)
        tracking_data['q_value'].append(np.mean(episode_q_values) if episode_q_values else 0.0)
        tracking_data['loss'].append(np.mean(episode_losses) if episode_losses else 0.0)
        tracking_data['action_dist'].append(episode_actions.tolist())

        print(f"Episode {episode + 1}, Highest Reward = {highest_reward}")

    # 傳遞 global_best_actions 給寫檔函數
    save_analysis_results(log_dir, tracking_data, baseline_delay, baseline_area, best_metrics, global_best_actions)

if __name__ == '__main__':
    genlib_origin = sys.argv[-1]
    lib_origin = genlib_origin[:-7] + '.lib'
    design = sys.argv[-2]
    sample_gate = int(sys.argv[-3])
    
    design_basename = os.path.basename(design).split('.')[0]
    lib_name = os.path.basename(genlib_origin).replace(".genlib", "")
    
    # If running the same design with different parameters, append PID to log_dir
    pid = os.getpid()
    log_dir = f"analysis_results_{design_basename}_{lib_name}_{pid}"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("temp_blifs", exist_ok=True)
    os.makedirs("gen_newlibs", exist_ok=True)

    # Add PID to temp blif to prevent overlap during baseline calculation
    temp_blif = f"temp_blifs/{design_basename}_{pid}_dqn_temp.blif"
    lib_path = "gen_newlibs/"
    
    abc_cmd = "read %s;read %s; map -a; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
    try:
        res = subprocess.check_output(('abc', '-c', abc_cmd), text=True)
        match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
        match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
        max_delay = float(match_d.group(1))
        max_area = float(match_a.group(1))
    except Exception as e:
        print("Failed to run baseline:", e)
        max_delay, max_area = 1.0, 1.0

    print('Baseline Delay: ', max_delay)
    print('Baseline Area: ', max_area)

    with open(genlib_origin, 'r') as f:
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]

    total_gates = len(f_lines)
    state_size = total_gates
    action_size = total_gates
    num_episodes = 100
    batch_size = 10
    buffer_size = 10000

    env = GateSelectionEnv(genlib_origin, lib_path, design, total_gates, sample_gate, max_delay, max_area)
    agent = DQNAgent(state_size, action_size)

    start = time.time()
    train_agent(num_episodes, agent, env, batch_size, buffer_size, log_dir, max_delay, max_area)
    end = time.time()
    
    # Clean up baseline temp blif
    if os.path.exists(temp_blif):
        os.remove(temp_blif)
        
    print('Total time: ', end - start)