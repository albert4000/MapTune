"""
Constraint-based gate selection (one cell per logic group + optional None per group).
Sequential DQN: one group per env step; invalid actions are masked in Q selection / bootstrap.
See MapTune/DOC/plan.md.
"""
import csv
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces


# First boolean output assignment on a GATE line (Y/Z/CON/SN/Q/QN).
_BOOL_ASSIGN_RE = re.compile(
    r"\b(Y|y|Z|z|CON|con|SN|sn|Q|q|QN|qn)=([^;]+);"
)

# Truth-table canonicalization: same TT under sorted-variable enumeration => semantic equivalence.
_MAX_TT_VARS = 14


def _normalize_bool_expr(expr: str) -> str:
    """Syntactic fallback key when parsing or TT enumeration is unsafe."""
    s = expr.strip()
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _gate_name_from_line(line: str) -> str:
    parts = line.split()
    return parts[1].lower() if len(parts) > 1 else ""


def _gate_name_token(line: str) -> str:
    """Exact genlib cell name (second field), for duplicate-name / multi-output bundling."""
    parts = line.split()
    return parts[1] if len(parts) > 1 else ""


def _is_sequential_cell(name: str) -> bool:
    return "dff" in name or "dhl" in name


def _is_fixed_keep(name: str) -> bool:
    return "buf" in name or "inv" in name or "const" in name


def _first_bool_expr(line: str) -> Optional[str]:
    m = _BOOL_ASSIGN_RE.search(line)
    return m.group(2).strip() if m else None


class _GenlibBoolParser:
    """Parse genlib Boolean (AND=*, OR=+, NOT=!) into ('|',a,b), ('&',a,b), ('!',a), ('v',name)."""

    __slots__ = ("s", "i")

    def __init__(self, s: str):
        self.s = "".join(s.split())
        self.i = 0

    def _peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def _get(self) -> str:
        c = self._peek()
        self.i += 1
        return c

    def parse(self):
        e = self.parse_or()
        if self.i != len(self.s):
            raise ValueError(f"trailing junk at {self.i}: {self.s[self.i :]}")
        return e

    def parse_or(self):
        e = self.parse_and()
        while self._peek() == "+":
            self.i += 1
            e = ("|", e, self.parse_and())
        return e

    def parse_and(self):
        e = self.parse_unary()
        while self._peek() == "*":
            self.i += 1
            e = ("&", e, self.parse_unary())
        return e

    def parse_unary(self):
        if self._peek() == "!":
            self.i += 1
            return ("!", self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        c = self._peek()
        if c == "(":
            self.i += 1
            e = self.parse_or()
            if self._get() != ")":
                raise ValueError("expected )")
            return e
        if c.isalpha() or c == "_":
            return ("v", self._read_ident())
        raise ValueError(f"unexpected {c!r} at {self.i}")

    def _read_ident(self) -> str:
        start = self.i
        while self.i < len(self.s):
            c = self.s[self.i]
            if c.isalnum() or c == "_":
                self.i += 1
            else:
                break
        if self.i == start:
            raise ValueError("expected identifier")
        return self.s[start : self.i]


def _collect_vars(node) -> List[str]:
    if node[0] == "v":
        return [node[1]]
    if node[0] == "!":
        return _collect_vars(node[1])
    if node[0] in ("&", "|"):
        return _collect_vars(node[1]) + _collect_vars(node[2])
    raise ValueError(node)


def _eval_bool(node, env: Dict[str, int]) -> int:
    k = node[0]
    if k == "v":
        return int(env[node[1]])
    if k == "!":
        return 1 - _eval_bool(node[1], env)
    if k == "&":
        return _eval_bool(node[1], env) & _eval_bool(node[2], env)
    if k == "|":
        return _eval_bool(node[1], env) | _eval_bool(node[2], env)
    raise ValueError(node)


def _truth_table_semantic_key(expr: str) -> Optional[str]:
    """
    Canonical key: sorted var names + output column for all assignments (lexicographic on vars).
    Detects semantic equivalence (e.g. (A*B) vs (B*A)) for genlib-style identifiers.
    """
    try:
        ast = _GenlibBoolParser(expr).parse()
    except Exception:
        return None
    vs = sorted(set(_collect_vars(ast)))
    if not vs:
        try:
            b = _eval_bool(ast, {})
            return f"const:{b}"
        except Exception:
            return None
    n = len(vs)
    if n > _MAX_TT_VARS:
        return None
    bits: List[int] = []
    for mask in range(1 << n):
        env = {vs[i]: (mask >> i) & 1 for i in range(n)}
        bits.append(_eval_bool(ast, env))
    return f"tt{n}:" + "".join(str(b) for b in bits)


def _semantic_group_key(line: str) -> str:
    expr = _first_bool_expr(line)
    if expr is None:
        return "noparse:" + _normalize_bool_expr(line[:120])
    sk = _truth_table_semantic_key(expr)
    if sk is not None:
        return sk
    return "syn:" + _normalize_bool_expr(expr)


def build_constraint_groups(
    genlib_path: str,
) -> Tuple[List[str], Dict[str, List[List[str]]]]:
    """
    Returns (f_keep, gate_groups) where each gate_groups[key] is a list of *variants*.
    Each variant is a list of genlib lines to emit together (singleton for typical cells,
    multi-line for one physical cell with multiple GATE rows / outputs).
    """
    f_keep: List[str] = []
    combinational: List[str] = []

    with open(genlib_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("GATE"):
                continue
            lname = _gate_name_from_line(line)
            if _is_sequential_cell(lname):
                continue
            if _is_fixed_keep(lname):
                f_keep.append(line)
                continue
            if _first_bool_expr(line) is None:
                continue
            combinational.append(line)

    by_name: Dict[str, List[str]] = {}
    for line in combinational:
        by_name.setdefault(_gate_name_token(line), []).append(line)

    entity_names = {name for name, lines in by_name.items() if len(lines) > 1}
    used = set()

    gate_groups: Dict[str, List[List[str]]] = {}

    for name in sorted(entity_names):
        lines = sorted(by_name[name])
        key = f"entity:{name}"
        gate_groups[key] = [lines]
        for ln in lines:
            used.add(id(ln))

    semantic_bins: Dict[str, List[List[str]]] = {}
    for line in combinational:
        if id(line) in used:
            continue
        gk = _semantic_group_key(line)
        semantic_bins.setdefault(gk, []).append([line])

    for gk, variants in semantic_bins.items():
        variants.sort(key=lambda v: v[0])
        gate_groups[gk] = variants

    return f_keep, gate_groups


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class GateSelectionEnvCB(gym.Env):
    """
    Combinational cells: (1) same truth table => one semantic group; (2) duplicate GATE name
    (multi-output macro) => one entity bundle. Per group choose one variant or None.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, genlib_origin, lib_path, design, max_delay, max_area):
        super().__init__()
        self.genlib_origin = genlib_origin
        self.lib_path = lib_path
        self.design = design
        self.max_delay = max_delay
        self.max_area = max_area

        self.f_keep, self.gate_groups = build_constraint_groups(genlib_origin)
        # Stable order: entity bundles first, then semantic TT keys
        entity_keys = sorted(k for k in self.gate_groups if k.startswith("entity:"))
        rest_keys = sorted(k for k in self.gate_groups if not k.startswith("entity:"))
        self._group_keys = entity_keys + rest_keys
        self.group_variants = [self.gate_groups[k] for k in self._group_keys]
        self.branch_sizes = [len(variants) + 1 for variants in self.group_variants]  # +1 = None
        self.num_groups = len(self._group_keys)
        if self.num_groups == 0:
            raise ValueError("No combinational gate groups after parsing; check .genlib format.")

        self.max_branches = int(max(self.branch_sizes))
        self.action_space = spaces.Discrete(self.max_branches)
        # Per-group choice progress (normalized) + fractional episode progress
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_groups + 1,), dtype=np.float32
        )

        self.selection_count = 0
        self.choices: List[int] = [0] * self.num_groups
        self.state = self._build_observation()

    @property
    def group_keys(self) -> List[str]:
        return self._group_keys

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(self.num_groups + 1, dtype=np.float32)
        for i in range(self.selection_count):
            n_br = self.branch_sizes[i]
            obs[i] = (self.choices[i] + 1.0) / float(n_br)
        obs[-1] = float(self.selection_count) / float(max(1, self.num_groups))
        return obs

    def _current_valid_n(self) -> int:
        return self.branch_sizes[self.selection_count]

    def step(self, action: int):
        valid_n = self._current_valid_n()
        if action < 0 or action >= valid_n:
            action = random.randrange(0, valid_n)

        self.choices[self.selection_count] = action
        self.selection_count += 1
        done = self.selection_count >= self.num_groups
        reward = 0.0
        delay = self.max_delay
        area = self.max_area
        out_genlib = ""
        out_blif = ""
        selected_lines: List[str] = []

        choice_snapshot: List[Tuple[str, int]] = []
        if done:
            choice_snapshot = list(zip(self._group_keys, list(self.choices)))
            selected_lines = self._lines_from_choices()
            delay, area, out_genlib, out_blif = self.technology_mapper(selected_lines)
            reward = self.calculate_reward(delay, area)
            next_state = self.reset()
        else:
            next_state = self._build_observation()

        return (
            next_state,
            reward,
            done,
            delay,
            area,
            out_genlib,
            out_blif,
            selected_lines,
            choice_snapshot,
        )

    def _lines_from_choices(self) -> List[str]:
        lines: List[str] = []
        for i, a in enumerate(self.choices):
            variants = self.group_variants[i]
            if a < len(variants):
                lines.extend(variants[a])
        return lines

    def technology_mapper(self, partial_cell_lines: List[str]):
        lines_partial = list(partial_cell_lines) + self.f_keep
        design_basename = os.path.basename(self.design)
        design_name = os.path.splitext(design_basename)[0]
        output_genlib_file = os.path.join(
            self.lib_path,
            f"{design_basename}_{len(lines_partial)}_dqn_cb_samplelib.genlib",
        )
        lib_origin = self.genlib_origin[:-7] + ".lib"
        temp_blif = os.path.join("temp_blifs", f"{design_name}_dqn_cb_temp.blif")
        with open(output_genlib_file, "w") as out_gen:
            for line in lines_partial:
                out_gen.write(line + "\n")

        abc_cmd = (
            f"abc -c 'read {output_genlib_file}; read {self.design}; map; write {temp_blif}; "
            f"read {lib_origin}; read -m {temp_blif}; ps; topo; stime;'"
        )
        try:
            res = subprocess.check_output(abc_cmd, shell=True, text=True)
            match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", res)
            match_a = re.search(r"Area\s*=\s*([\d.]+)", res)
            delay = float(match_d.group(1)) if match_d else float("inf")
            area = float(match_a.group(1)) if match_a else float("inf")
        except subprocess.CalledProcessError as e:
            print("Failed to execute ABC:", e)
            delay, area = float("inf"), float("inf")

        return delay, area, output_genlib_file, temp_blif

    def calculate_reward(self, delay, area):
        if delay == float("inf") or area == float("inf"):
            return float("-inf")
        normalized_delay = delay / self.max_delay
        normalized_area = area / self.max_area
        return -np.sqrt(normalized_delay * normalized_area)

    def reset(self):
        self.selection_count = 0
        self.choices = [0] * self.num_groups
        self.state = self._build_observation()
        return self.state

    def render(self, mode="human"):
        print(f"Choices per group: {self.choices}")

    def close(self):
        pass


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
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

    def select_action(self, state, valid_n: int, epsilon: float = 0.2):
        if np.random.rand() < epsilon:
            return random.randrange(0, valid_n)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.model(s).squeeze(0)
            q = q.clone()
            q[valid_n:] = float("-inf")
            return int(q.argmax().item())

    def update_batch(self, batch):
        states, actions, rewards, next_states, dones, valid_ns, valid_next_ns = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        valid_next_ns = torch.LongTensor(np.array(valid_next_ns)).to(self.device)

        current_qs = self.model(states).gather(1, actions).squeeze(1)
        next_q_all = self.model(next_states)
        mask = torch.arange(next_q_all.size(1), device=self.device).unsqueeze(0) < valid_next_ns.unsqueeze(
            1
        )
        masked = next_q_all.masked_fill(~mask, float("-inf"))
        next_qs = masked.max(1)[0]
        expected_qs = rewards + self.gamma * (1.0 - dones) * next_qs

        loss = F.mse_loss(current_qs, expected_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_agent(num_episodes, agent, env: GateSelectionEnvCB, batch_size, buffer_size, paths, hyperparams):
    replay_buffer = ReplayBuffer(buffer_size)
    highest_reward = float("-inf")

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            valid_n = env._current_valid_n()
            action = agent.select_action(state, valid_n, epsilon=hyperparams["epsilon"])
            next_state, reward, done, delay, area, out_genlib, out_blif, selected_lines, choice_snapshot = (
                env.step(action)
            )
            valid_next = env._current_valid_n() if not done else env.max_branches
            replay_buffer.push((state, action, reward, next_state, float(done), valid_n, valid_next))
            state = next_state

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.update_batch(batch)

            if done and reward > highest_reward:
                highest_reward = reward
                best_result = (delay, area)
                print("Current Best Result: ", best_result)
                choice_summary = choice_snapshot

                if os.path.exists(out_genlib):
                    shutil.copy(out_genlib, paths["genlib_path"])
                if os.path.exists(out_blif):
                    shutil.copy(out_blif, paths["blif_path"])

                with open(paths["txt_path"], "w") as f:
                    f.write(f"Per-group choices (normalized_key, branch_index): {choice_summary}\n")
                    f.write(f"Selected line count (excl. fixed keep): {len(selected_lines)}\n")
                    for ln in selected_lines:
                        f.write(ln + "\n")

                with open(paths["csv_path"], "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Metric", "Value"])
                    writer.writerow(["Best_Delay", delay])
                    writer.writerow(["Best_Area", area])
                    writer.writerow(["Best_Reward", reward])
                    writer.writerow(["Baseline_Delay", hyperparams["baseline_delay"]])
                    writer.writerow(["Baseline_Area", hyperparams["baseline_area"]])
                    writer.writerow(["---", "---"])
                    writer.writerow(["Hyperparameter", "Value"])
                    for k, v in hyperparams.items():
                        if k not in ["baseline_delay", "baseline_area"]:
                            writer.writerow([k, v])

        print(f"Episode {episode + 1}, Highest Reward = {highest_reward}")


def main():
    genlib_origin = sys.argv[-1]
    design = sys.argv[-2]
    # Third CLI arg kept for parity with batched_DQN_nf.py (sample_gate); unused here.
    _sample_gate_unused = sys.argv[-3] if len(sys.argv) >= 4 else None

    lib_origin = genlib_origin[:-7] + ".lib"
    design_basename = os.path.basename(design)
    design_name = os.path.splitext(design_basename)[0]
    temp_blif = os.path.join("temp_blifs", f"{design_name}_dqn_cb_temp.blif")
    lib_path = "gen_newlibs/"
    rlmethodname = "dqn_cb"
    lib_name = os.path.basename(genlib_origin).replace(".genlib", "")

    os.makedirs("temp_blifs", exist_ok=True)
    os.makedirs("gen_newlibs", exist_ok=True)
    best_output_dir = os.path.join("output", "best", lib_name, design_name)
    os.makedirs(best_output_dir, exist_ok=True)

    paths = {
        "genlib_path": os.path.join(best_output_dir, f"best_{lib_name}_{rlmethodname}.genlib"),
        "blif_path": os.path.join(best_output_dir, f"best_{design_name}_{rlmethodname}.blif"),
        "csv_path": os.path.join(best_output_dir, f"metrics_{rlmethodname}.csv"),
        "txt_path": os.path.join(best_output_dir, f"best_gates_{rlmethodname}.txt"),
    }

    abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; stime; " % (
        genlib_origin,
        design,
        temp_blif,
        lib_origin,
        temp_blif,
    )
    res = subprocess.check_output(("abc", "-c", abc_cmd))
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    max_delay = float(match_d.group(1))
    max_area = float(match_a.group(1))
    print("Baseline Delay: ", max_delay)
    print("Baseline Area: ", max_area)

    num_episodes = 5000
    batch_size = 10
    buffer_size = 10000
    epsilon = 0.2
    learning_rate = 0.001

    hyperparams = {
        "baseline_delay": max_delay,
        "baseline_area": max_area,
        "num_episodes": num_episodes,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "epsilon": epsilon,
        "learning_rate": learning_rate,
        "mode": "constraint_groups_sequential_dqn",
    }

    env = GateSelectionEnvCB(genlib_origin, lib_path, design, max_delay, max_area)
    state_size = env.observation_space.shape[0]
    action_size = env.max_branches
    agent = DQNAgent(state_size, action_size, learning_rate=learning_rate)

    start = time.time()
    train_agent(num_episodes, agent, env, batch_size, buffer_size, paths, hyperparams)
    end = time.time()
    print("Total time: ", end - start)


if __name__ == "__main__":
    main()
