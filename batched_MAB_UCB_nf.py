import random
import math
import sys
import os 
import numpy as np
import subprocess
from subprocess import PIPE
import re
import time
import shutil
import csv

genlib_origin = sys.argv[-1]
lib_origin = genlib_origin[:-7] + '.lib'
design = sys.argv[-2]
sample_gate = int(sys.argv[-3])
design_basename = os.path.basename(design)
circuit_name = os.path.splitext(design_basename)[0]
temp_blif = os.path.join("temp_blifs", f"{circuit_name}_bs_ucb_temp.blif")
lib_path = "gen_newlibs/" 
rlmethodname = "bs_ucb"
lib_name = os.path.basename(genlib_origin).replace('.genlib', '')

os.makedirs("temp_blifs", exist_ok=True)
os.makedirs("gen_newlibs", exist_ok=True)
best_output_dir = os.path.join("output", "best", lib_name, circuit_name)
os.makedirs(best_output_dir, exist_ok=True)

paths = {
    'genlib_path': os.path.join(best_output_dir, f"best_{lib_name}_{rlmethodname}.genlib"),
    'blif_path': os.path.join(best_output_dir, f"best_{circuit_name}_{rlmethodname}.blif"),
    'csv_path': os.path.join(best_output_dir, f"metrics_{rlmethodname}.csv"),
    'txt_path': os.path.join(best_output_dir, f"best_gates_{rlmethodname}.txt")
}

start=time.time()
# abc_cmd = "read %s;read %s; map -a; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
# abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
res = subprocess.check_output(('abc', '-c', abc_cmd))
match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
# Baseline
max_delay = float(match_d.group(1))
max_area = float(match_a.group(1))

print("Baseline Delay:", max_delay)
print("Baseline Area:", max_area)

# Mapper call
def technology_mapper(genlib_origin, partial_cell_library):
    with open(genlib_origin, 'r') as f:
        #f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    f.close()
    with open(genlib_origin, 'r') as f:
        #f_keep = [line.strip() for line in f if any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        f_keep = [line.strip() for line in f if line.startswith("GATE BUF") or line.startswith("GATE INV") or line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    f.close()
    lines_partial = [f_lines[i] for i in partial_cell_library]
    lines_partial = lines_partial + f_keep
    output_genlib_file = os.path.join(lib_path, f"{design_basename}_{len(lines_partial)}_bs_ucb_samplelib.genlib")
    with open(output_genlib_file, 'w') as out_gen:
        for line in lines_partial:
            out_gen.write(line + '\n')
    out_gen.close() 

    # abc_cmd = "read %s;read %s; map -a; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (output_genlib_file, design, temp_blif, lib_origin, temp_blif)
    # abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (output_genlib_file, design, temp_blif, lib_origin, temp_blif)
    abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; stime; " % (output_genlib_file, design, temp_blif, lib_origin, temp_blif)
    res = subprocess.check_output(('abc', '-c', abc_cmd))
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    if match_d and match_a:
        delay = float(match_d.group(1))
        area = float(match_a.group(1))
    else:
        delay, area = float("NaN"),float("NaN")
    return delay, area, output_genlib_file, temp_blif

# Reward calculation
def calculate_reward(max_delay, max_area, delay, area):
    normalized_delay = delay / max_delay
    normalized_area = area / max_area

    return -np.sqrt(normalized_delay * normalized_area) 

class UCB_MAB:
    def __init__(self, num_arms, c, sample_gate, batch_size):
        self.num_arms = num_arms
        self.c = c  # Exploration parameter for UCB
        self.q_values = [0.0] * num_arms
        self.counts = [0] * num_arms
        self.sample_gate = sample_gate
        self.batch_size = batch_size

    def select_batch_actions(self):
        batches = []
        for _ in range(self.batch_size):
            selected_cells = set()
            total_counts = sum(self.counts)
            ucb_values = [0.0 if count == 0 else self.q_values[arm] + self.c * math.sqrt(math.log(total_counts) / count)
                          for arm, count in enumerate(self.counts)]
            # Ensure every arm is selected at least once
            for arm in range(self.num_arms):
                if self.counts[arm] == 0:
                    selected_cells.add(arm)
                    ucb_values[arm] = float('-inf')  # Make sure not to select again in this batch
                    if len(selected_cells) == self.sample_gate:
                        break
            # Select remaining cells based on UCB values
            while len(selected_cells) < self.sample_gate:
                selected_cell = max(range(self.num_arms), key=lambda x: ucb_values[x])
                selected_cells.add(selected_cell)
                ucb_values[selected_cell] = float('-inf')  # Make sure not to select again in this batch
            batches.append(list(selected_cells))
        return batches

    def update_batch(self, batch_actions, rewards):
        for selected_arm, reward in zip(batch_actions, rewards):
            for arm in selected_arm:
                self.counts[arm] += 1
                self.q_values[arm] = (self.q_values[arm] * (self.counts[arm] - 1) + reward) / self.counts[arm]

# Initialization
num_cells_select = sample_gate
with open(genlib_origin, 'r') as f:
        #f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
f.close()
num_arms=len(f_lines)
c=2
batch_size=10
mab = UCB_MAB(num_arms, c, sample_gate, batch_size)  
best_cells = None
best_result = (float('inf'), float('inf'))  
best_reward = -float('inf') 

# Main Loop
num_iterations = 100

for i in range(num_iterations):  # Adjust iterations based on batch size
    print(f"Batch iteration: {i}")
    batch_actions = mab.select_batch_actions()
    batch_rewards = []
    for selected_cells in batch_actions:
        delay, area, out_genlib, out_blif = technology_mapper(genlib_origin, selected_cells)
        if np.isnan(delay) or np.isnan(area):
            reward = -float('inf')
        else:
            reward = calculate_reward(max_delay, max_area, delay, area)
        if reward > best_reward:
            best_reward = reward
            best_result = (delay, area)
            best_cells = selected_cells

            if os.path.exists(out_genlib):
                shutil.copy(out_genlib, paths['genlib_path'])
            if os.path.exists(out_blif):
                shutil.copy(out_blif, paths['blif_path'])

            with open(paths['txt_path'], 'w') as f:
                f.write(f"Selected Gates Indices: {best_cells}\n")
                gate_names = [f_lines[i].split()[1] for i in best_cells]
                f.write(f"Selected Gates Names: {gate_names}\n")

            with open(paths['csv_path'], 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Best_Delay", delay])
                writer.writerow(["Best_Area", area])
                writer.writerow(["Best_Reward", reward])
                writer.writerow(["Baseline_Delay", max_delay])
                writer.writerow(["Baseline_Area", max_area])
                writer.writerow(["---", "---"])
                writer.writerow(["Hyperparameter", "Value"])
                writer.writerow(["sample_gate", sample_gate])
                writer.writerow(["num_iterations", num_iterations])
                writer.writerow(["batch_size", batch_size])
                writer.writerow(["c", c])

        batch_rewards.append(reward)
    print("Current best reward: ", best_reward)
    print("Current best result: ", best_result)
        # Update best results tracking here as needed
    mab.update_batch(batch_actions, batch_rewards)
end=time.time()
runtime=end-start

print("Best Delay:", best_result[0])
print("Best Area:", best_result[1])
print("Best Reward:", best_reward)
print("Total time:", runtime)
