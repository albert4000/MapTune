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
import shutil
import csv

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
        
        # Parse library once to avoid I/O overhead during iterations
        self.f_lines = []
        self.f_keep = []
        with open(self.genlib_origin, 'r') as f:
            for line in f:
                if line.startswith("GATE"):
                    tokens = line.split()
                    if len(tokens) > 1:
                        gate_name = tokens[1].lower()
                        # Retain constants, buffers, and inverters
                        if "buf" in gate_name or "inv" in gate_name or "const" in gate_name:
                            self.f_keep.append(line.strip())
                        else:
                            self.f_lines.append(line.strip())

    def step(self, action):
        if self.state[action] == 0 and self.selection_count < self.sample_gate:
            self.state[action] = 1
            self.selection_count += 1

        done = self.selection_count == self.sample_gate
        reward = 0
        delay = float('inf')
        area = float('inf')
        out_genlib = ""
        out_blif = ""
        next_state = self.state.copy()

        if done:
            delay, area, out_genlib, out_blif = self.technology_mapper(list(np.where(self.state == 1)[0]))
            reward = self.calculate_reward(delay, area)
            next_state = self.reset()
            
        return next_state, reward, done, delay, area, out_genlib, out_blif

    def technology_mapper(self, partial_cell_library):
        lines_partial = [self.f_lines[i] for i in partial_cell_library] + self.f_keep
        
        design_basename = os.path.basename(self.design)
        output_genlib_file = self.lib_path + design_basename + "_" + str(len(lines_partial)) + "_ddqn_samplelib.genlib"
        lib_origin = self.genlib_origin[:-7] + '.lib'
        temp_blif = "temp_blifs/" + design_basename[:-6] + "_ddqn_temp.blif"
        
        with open(output_genlib_file, 'w') as out_gen:
            for line in lines_partial:
                out_gen.write(line + '\n')

        # abc_cmd = f"abc -c 'read {output_genlib_file}; read {self.design}; map -a; write {temp_blif}; read {lib_origin}; read -m {temp_blif}; ps; topo; upsize; dnsize; stime;'"
        # abc_cmd = f"abc -c 'read {output_genlib_file}; read {self.design}; map; write {temp_blif}; read {lib_origin}; read -m {temp_blif}; ps; topo; upsize; dnsize; stime;'"
        abc_cmd = f"abc -c 'read {output_genlib_file}; read {self.design}; map; write {temp_blif}; read {lib_origin}; read -m {temp_blif}; ps; topo; stime;'"
        try:
            res = subprocess.check_output(abc_cmd, shell=True, text=True)
            match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", res)
            match_a = re.search(r"Area\s*=\s*([\d.]+)", res)
            delay = float(match_d.group(1)) if match_d else float('inf')
            area = float(match_a.group(1)) if match_a else float('inf')
        except subprocess.CalledProcessError as e:
            print("Failed to execute ABC:", e)
            delay, area = float('inf'), float('inf')

        return delay, area, output_genlib_file, temp_blif

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

class DDQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DDQNAgent:
    def __init__(self, state_size, action_size, learning_rate, tau):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_network = DDQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DDQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.gamma = 0.99
        self.tau = tau

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, len(state))
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.online_network(state)
                return q_values.argmax().item()

    def update_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        current_qs = self.online_network(states).gather(1, actions).squeeze(1)
        next_state_actions = self.online_network(next_states).argmax(1).unsqueeze(1)
        next_qs = self.target_network(next_states).gather(1, next_state_actions).squeeze(1)
        expected_qs = rewards + self.gamma * (1 - dones) * next_qs

        loss = F.mse_loss(current_qs, expected_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.online_network, self.target_network, self.tau)

    def soft_update(self, online_network, target_network, tau):
        for target_param, online_param in zip(target_network.parameters(), online_network.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

def train_agent(num_episodes, agent, env, batch_size, buffer_size, epsilon, paths, hyperparams):
    replay_buffer = deque(maxlen=buffer_size)
    highest_reward = float('-inf')

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        current_gate_selection = []

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, delay, area, out_genlib, out_blif = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                current_gate_selection = list(np.where(state == 1)[0])
                if reward > highest_reward:
                    highest_reward = reward
                    best_result = (delay, area)
                    print(f'Current Best Result: Delay: {delay}, Area: {area}, Reward: {reward}')
                    
                    # Overwrite best files
                    if os.path.exists(out_genlib):
                        shutil.copy(out_genlib, paths['genlib_path'])
                    if os.path.exists(out_blif):
                        shutil.copy(out_blif, paths['blif_path'])
                    
                    with open(paths['txt_path'], 'w') as f:
                        f.write(f"Selected Gates Indices: {current_gate_selection}\n")
                        gate_names = [env.f_lines[i].split()[1] for i in current_gate_selection]
                        f.write(f"Selected Gates Names: {gate_names}\n")

                    with open(paths['csv_path'], 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Metric", "Value"])
                        writer.writerow(["Best_Delay", delay])
                        writer.writerow(["Best_Area", area])
                        writer.writerow(["Best_Reward", reward])
                        writer.writerow(["Baseline_Delay", hyperparams['baseline_delay']])
                        writer.writerow(["Baseline_Area", hyperparams['baseline_area']])
                        writer.writerow(["---", "---"])
                        writer.writerow(["Hyperparameter", "Value"])
                        for k, v in hyperparams.items():
                            if k not in ['baseline_delay', 'baseline_area']:
                                writer.writerow([k, v])

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                agent.update_batch(batch)

        print(f"Episode {episode + 1}, Highest Reward = {highest_reward}")

if __name__ == "__main__":
    genlib_origin = sys.argv[-1]
    design = sys.argv[-2]
    sample_gate = int(sys.argv[-3])
    
    lib_origin = genlib_origin[:-7] + '.lib'
    lib_name = os.path.basename(genlib_origin).replace('.genlib', '')
    circuit_name = os.path.basename(design).replace('.bench', '').replace('.blif', '')
    rlmethodname = "ddqn"
    
    # Create directories
    os.makedirs("temp_blifs", exist_ok=True)
    os.makedirs("gen_newlibs", exist_ok=True)
    best_output_dir = f"output/best/{lib_name}/{circuit_name}/"
    os.makedirs(best_output_dir, exist_ok=True)

    paths = {
        'genlib_path': os.path.join(best_output_dir, f"best_{lib_name}_{rlmethodname}.genlib"),
        'blif_path': os.path.join(best_output_dir, f"best_{circuit_name}_{rlmethodname}.blif"),
        'csv_path': os.path.join(best_output_dir, f"metrics_{rlmethodname}.csv"),
        'txt_path': os.path.join(best_output_dir, f"best_gates_{rlmethodname}.txt")
    }

    temp_blif = "temp_blifs/" + circuit_name + "_ddqn_temp.blif"
    lib_path = "gen_newlibs/"
    
    # Get Baseline
    # abc_cmd = "read %s;read %s; map -a; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
    # abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
    abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
    res = subprocess.check_output(('abc', '-c', abc_cmd))
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    max_delay = float(match_d.group(1))
    max_area = float(match_a.group(1))
    print('Baseline Delay: ', max_delay)
    print('Baseline Area: ', max_area)

    # Hyperparameters mapping
    hyperparams = {
        'baseline_delay': max_delay,
        'baseline_area': max_area,
        'sample_gate': sample_gate,
        'num_episodes': 5000,
        'batch_size': 10,
        'buffer_size': 10000,
        'learning_rate': 0.001,
        'tau': 0.01,
        'epsilon': 0.2
    }

    # Calculate total gates correctly
    total_gates = 0
    with open(genlib_origin, 'r') as f:
        for line in f:
            if line.startswith("GATE"):
                gate_name = line.split()[1].lower()
                if not ("buf" in gate_name or "inv" in gate_name or "const" in gate_name):
                    total_gates += 1

    state_size = total_gates
    action_size = total_gates

    env = GateSelectionEnv(genlib_origin, lib_path, design, total_gates, sample_gate, max_delay, max_area)
    agent = DDQNAgent(state_size, action_size, learning_rate=hyperparams['learning_rate'], tau=hyperparams['tau'])
    
    start = time.time()
    train_agent(hyperparams['num_episodes'], agent, env, hyperparams['batch_size'], hyperparams['buffer_size'], hyperparams['epsilon'], paths, hyperparams)
    end = time.time()
    runtime = end - start
    print('Total time: ', runtime)