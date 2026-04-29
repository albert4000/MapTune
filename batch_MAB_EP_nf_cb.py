"""
Constraint-based batched epsilon-greedy MAB over logic-function groups with a per-group None arm.

CLI:
  python3 batch_MAB_EP_nf_cb.py <design.blif> <library.genlib>

Groups combinational gates by Boolean equivalence (full truth table under alphabetically
sorted variable names). Parsing failures fall back to whitespace-stripped text keys.
"""
import csv
import os
import random
import re
import shutil
import subprocess
import sys
import time

import numpy as np

if len(sys.argv) < 3:
    print(
        "Usage: python3 batch_MAB_EP_nf_cb.py <design.blif> <library.genlib>",
        file=sys.stderr,
    )
    sys.exit(1)

# --- argv / paths ---
genlib_origin = sys.argv[-1]
lib_origin = genlib_origin[:-7] + ".lib"
design = sys.argv[-2]
design_basename = os.path.basename(design)
circuit_name = os.path.splitext(design_basename)[0]
temp_blif = os.path.join("temp_blifs", f"{circuit_name}_bs_ep_cb_temp.blif")
lib_path = "gen_newlibs/"
rlmethodname = "bs_ep_cb"
lib_name = os.path.basename(genlib_origin).replace(".genlib", "")

os.makedirs("temp_blifs", exist_ok=True)
os.makedirs("gen_newlibs", exist_ok=True)
best_output_dir = os.path.join("output", "best", lib_name, circuit_name)
os.makedirs(best_output_dir, exist_ok=True)

paths = {
    "genlib_path": os.path.join(best_output_dir, f"best_{lib_name}_{rlmethodname}.genlib"),
    "blif_path": os.path.join(best_output_dir, f"best_{circuit_name}_{rlmethodname}.blif"),
    "csv_path": os.path.join(best_output_dir, f"metrics_{rlmethodname}.csv"),
    "txt_path": os.path.join(best_output_dir, f"best_gates_{rlmethodname}.txt"),
}

# `| tee` etc.: stdout is not a tty → block buffering; tiny prints sit until ~8KiB or exit.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (OSError, ValueError):
        pass


def _normalize_bool_signature(bool_fragment: str) -> str:
    """Syntactic fallback key when Boolean parse fails."""
    s = bool_fragment.strip()
    s = re.sub(r"\s+", "", s)
    return s


def _split_output_and_rhs(bool_fragment: str) -> tuple[str, str]:
    """Split 'Y=(A*B);' -> ('Y', '(A*B)'); supports CON=, SN=, z=."""
    s = bool_fragment.strip()
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(";")
    m = re.match(r"^([A-Za-z]\w*)\s*=\s*(.+)$", s)
    if m:
        return m.group(1), m.group(2)
    return "Y", s


# --- genlib Boolean -> truth table (AND/OR/NOT, variables as identifiers) ---


def _tokenize_bool(s: str) -> list[str]:
    toks: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in "+*!()":
            toks.append(c)
            i += 1
        elif c.isalpha() or c == "_":
            j = i + 1
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            toks.append(s[i:j])
            i = j
        else:
            raise ValueError(f"unexpected character {c!r} in {s!r}")
    return toks


class _BoolParser:
    def __init__(self, toks: list[str]):
        self.toks = toks
        self.i = 0

    def _peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _eat(self, expected: str | None = None) -> str:
        t = self._peek()
        if t is None:
            raise ValueError("unexpected end of expression")
        if expected is not None and t != expected:
            raise ValueError(f"expected {expected!r}, got {t!r}")
        self.i += 1
        return t

    def parse(self):
        e = self._parse_or()
        if self._peek() is not None:
            raise ValueError(f"trailing tokens: {self.toks[self.i :]}")
        return e

    def _parse_or(self):
        lhs = self._parse_and()
        while self._peek() == "+":
            self._eat("+")
            rhs = self._parse_and()
            lhs = ("or", lhs, rhs)
        return lhs

    def _parse_and(self):
        lhs = self._parse_unary()
        while self._peek() == "*":
            self._eat("*")
            rhs = self._parse_unary()
            lhs = ("and", lhs, rhs)
        return lhs

    def _parse_unary(self):
        if self._peek() == "!":
            self._eat("!")
            return ("not", self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self):
        t = self._peek()
        if t == "(":
            self._eat("(")
            inner = self._parse_or()
            self._eat(")")
            return inner
        if t is None or t in "+*)":
            raise ValueError(f"bad primary at {t!r}")
        self._eat()
        if t in ("CONST0", "const0"):
            return ("const", 0)
        if t in ("CONST1", "const1"):
            return ("const", 1)
        return ("var", t)


def _collect_vars(node) -> set[str]:
    if node[0] == "var":
        return {node[1]}
    if node[0] == "const":
        return set()
    if node[0] == "not":
        return _collect_vars(node[1])
    if node[0] in ("and", "or"):
        return _collect_vars(node[1]) | _collect_vars(node[2])
    raise ValueError(node)


def _eval_expr(node, env: dict[str, int]) -> int:
    if node[0] == "const":
        return int(node[1])
    if node[0] == "var":
        return int(env[node[1]])
    if node[0] == "not":
        return 1 - _eval_expr(node[1], env)
    if node[0] == "and":
        return _eval_expr(node[1], env) & _eval_expr(node[2], env)
    if node[0] == "or":
        return min(1, _eval_expr(node[1], env) | _eval_expr(node[2], env))
    raise ValueError(node)


def _truth_table_key(output_pin: str, rhs: str) -> str:
    """
    Canonical key: same combinational function -> same key (up to variable rename
    via sorted-name bit ordering). Different output pins (Y vs CON) stay distinct.
    """
    rhs_clean = re.sub(r"\s+", "", rhs)
    tree = _BoolParser(_tokenize_bool(rhs_clean)).parse()
    vars_sorted = sorted(_collect_vars(tree))
    n = len(vars_sorted)
    if n > 16:
        raise ValueError("too many inputs for truth-table key")
    bits: list[int] = []
    for mask in range(1 << n):
        env = {vars_sorted[k]: (mask >> k) & 1 for k in range(n)}
        bits.append(_eval_expr(tree, env))
    tt_int = 0
    for b in bits:
        tt_int = (tt_int << 1) | b
    return f"tt:{output_pin}:n{n}:{tt_int}"


def bool_equivalence_key(bool_fragment: str) -> str:
    out_pin, rhs = _split_output_and_rhs(bool_fragment)
    try:
        return _truth_table_key(out_pin, rhs)
    except (ValueError, IndexError):
        return "raw:" + _normalize_bool_signature(bool_fragment)


def _is_sequential_gate(gate_name: str) -> bool:
    u = gate_name.upper()
    return "DFF" in u or "DHL" in u


def scan_multi_output_macros(genlib_path: str) -> list[tuple[str, list[str], int]]:
    """
    In .genlib, one combinational output is one GATE line. Macros with multiple outputs
    repeat the same cell name with different output assignments (e.g. CON= / SN=).
    Returns sorted list of (gate_name, sorted_output_pins, line_count).
    """
    from collections import defaultdict

    acc: dict[str, dict] = defaultdict(lambda: {"ports": set(), "n": 0})
    with open(genlib_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("GATE"):
                continue
            pin_idx = line.find(" PIN ")
            if pin_idx < 0:
                continue
            head = line[:pin_idx].strip()
            parts = head.split(None, 3)
            if len(parts) < 4:
                continue
            gate_name, bool_part = parts[1], parts[3]
            out_pin, _rhs = _split_output_and_rhs(bool_part)
            acc[gate_name]["ports"].add(out_pin)
            acc[gate_name]["n"] += 1
    out: list[tuple[str, list[str], int]] = []
    for gate_name in sorted(acc.keys()):
        ports = acc[gate_name]["ports"]
        nlines = acc[gate_name]["n"]
        if len(ports) > 1 or nlines > 1:
            out.append((gate_name, sorted(ports), nlines))
    return out


def _is_fixed_keep_line(line: str) -> bool:
    """INV/BUF drivers, sky130/gf180 naming, and CONST cells — always in genlib, not in MAB groups."""
    if line.startswith("GATE BUF") or line.startswith("GATE INV"):
        return True
    if line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv"):
        return True
    if line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith(
        "GATE gf180mcu_fd_sc_mcu7t5v0__inv"
    ):
        return True
    if "CONST0" in line or "CONST1" in line:
        return True
    if re.match(r"^GATE\s+_const[01]_", line):
        return True
    return False


def parse_genlib_grouped(genlib_path: str):
    """
    Build gate_groups: normalized boolean signature -> list of full GATE lines (combinational, not fixed-keep).
    f_keep: BUF/INV/CONST lines. Excludes sequential (DFF/DHL) entirely.
    """
    gate_groups: dict[str, list[str]] = {}
    f_keep: list[str] = []

    with open(genlib_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("GATE"):
                continue
            pin_idx = line.find(" PIN ")
            if pin_idx < 0:
                continue
            head = line[:pin_idx].strip()
            parts = head.split(None, 3)
            if len(parts) < 4:
                continue
            gate_name, _area, bool_part = parts[1], parts[2], parts[3]

            if _is_sequential_gate(gate_name):
                continue
            if _is_fixed_keep_line(line):
                f_keep.append(line)
                continue

            key = bool_equivalence_key(bool_part)
            gate_groups.setdefault(key, []).append(line)

    # Deterministic group order for MultiDiscrete axis alignment
    group_order = sorted(gate_groups.keys())
    return gate_groups, f_keep, group_order


def actions_to_selected_lines(
    gate_groups: dict[str, list[str]],
    group_order: list[str],
    actions: list[int],
) -> list[str]:
    """Map per-group arm index to lines; index == len(group) means None for that function."""
    selected: list[str] = []
    for g, sig in enumerate(group_order):
        variants = gate_groups[sig]
        arm = actions[g]
        if arm < len(variants):
            selected.append(variants[arm])
    return selected


def technology_mapper_cb(f_keep: list[str], selected_lines: list[str]):
    lines_partial = list(selected_lines) + list(f_keep)
    output_genlib_file = os.path.join(
        lib_path, f"{design_basename}_{len(lines_partial)}_{rlmethodname}_samplelib.genlib"
    )
    with open(output_genlib_file, "w") as out_gen:
        for ln in lines_partial:
            out_gen.write(ln + "\n")

    abc_cmd = (
        "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; stime; "
        % (output_genlib_file, design, temp_blif, lib_origin, temp_blif)
    )
    res = subprocess.check_output(("abc", "-c", abc_cmd))
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    if match_d and match_a:
        delay = float(match_d.group(1))
        area = float(match_a.group(1))
    else:
        delay, area = float("nan"), float("nan")
    return delay, area, output_genlib_file, temp_blif


def calculate_reward(max_delay, max_area, delay, area):
    normalized_delay = delay / max_delay
    normalized_area = area / max_area
    return -np.sqrt(normalized_delay * normalized_area)


class EpsilonGreedyGroupMAB:
    """One epsilon-greedy MAB per logic group; shared reward updates all chosen arms."""

    def __init__(self, arm_counts: list[int], epsilon: float, batch_size: int):
        self.arm_counts = arm_counts
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.q_values = [[0.0] * c for c in arm_counts]
        self.counts = [[0] * c for c in arm_counts]

    def select_batch_actions(self):
        batches = []
        for _ in range(self.batch_size):
            actions = []
            for g, n_arms in enumerate(self.arm_counts):
                if random.random() > self.epsilon:
                    actions.append(int(np.argmax(self.q_values[g])))
                else:
                    actions.append(random.randint(0, n_arms - 1))
            batches.append(actions)
        return batches

    def update_batch(self, batch_actions, rewards):
        for actions, reward in zip(batch_actions, rewards):
            for g, arm in enumerate(actions):
                self.counts[g][arm] += 1
                n = self.counts[g][arm]
                self.q_values[g][arm] = (self.q_values[g][arm] * (n - 1) + reward) / n


# --- baseline ---
start = time.time()
abc_cmd = (
    "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; stime; "
    % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
)
res = subprocess.check_output(("abc", "-c", abc_cmd))
match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
max_delay = float(match_d.group(1))
max_area = float(match_a.group(1))
print("Baseline Delay:", max_delay)
print("Baseline Area:", max_area)

gate_groups, f_keep, group_order = parse_genlib_grouped(genlib_origin)
arm_counts = [len(gate_groups[sig]) + 1 for sig in group_order]
print(
    f"CB groups: {len(group_order)}; arms per group (incl. None): min={min(arm_counts)} max={max(arm_counts)}"
)
multi = scan_multi_output_macros(genlib_origin)
print("Multi-output macros (same cell name, multiple GATE lines / pins):")
if not multi:
    print("  (none detected)")
else:
    for gname, ports, nlines in multi:
        print(f"  {gname}: {nlines} lines, pins {ports}")

batch_size = 10
epsilon = 0.2
num_iterations = 100

mab = EpsilonGreedyGroupMAB(arm_counts, epsilon, batch_size)
best_actions = None
best_result = (float("inf"), float("inf"))
best_reward = -float("inf")

for i in range(num_iterations):
    print(f"Batch iteration: {i}")
    batch_actions = mab.select_batch_actions()
    batch_rewards = []
    for actions in batch_actions:
        selected = actions_to_selected_lines(gate_groups, group_order, actions)
        delay, area, out_genlib, out_blif = technology_mapper_cb(f_keep, selected)
        if np.isnan(delay) or np.isnan(area):
            reward = -float("inf")
        else:
            reward = calculate_reward(max_delay, max_area, delay, area)
        if reward > best_reward:
            best_reward = reward
            best_result = (delay, area)
            best_actions = list(actions)
            if os.path.exists(out_genlib):
                shutil.copy(out_genlib, paths["genlib_path"])
            if os.path.exists(out_blif):
                shutil.copy(out_blif, paths["blif_path"])
            with open(paths["txt_path"], "w") as f:
                f.write(f"Per-group arm indices (incl. None as last arm): {best_actions}\n")
                f.write(f"Num groups: {len(group_order)}\n")
                sel_lines = actions_to_selected_lines(gate_groups, group_order, best_actions)
                names = [ln.split(None, 2)[1] for ln in sel_lines]
                f.write(f"Selected gate names: {names}\n")
            with open(paths["csv_path"], "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Best_Delay", delay])
                writer.writerow(["Best_Area", area])
                writer.writerow(["Best_Reward", reward])
                writer.writerow(["Baseline_Delay", max_delay])
                writer.writerow(["Baseline_Area", max_area])
                writer.writerow(["---", "---"])
                writer.writerow(["Hyperparameter", "Value"])
                writer.writerow(["num_groups", len(group_order)])
                writer.writerow(["num_iterations", num_iterations])
                writer.writerow(["batch_size", batch_size])
                writer.writerow(["epsilon", mab.epsilon])
        batch_rewards.append(reward)
    print("Current best reward: ", best_reward)
    print("Current best result: ", best_result)
    mab.update_batch(batch_actions, batch_rewards)

end = time.time()
print("Best Delay:", best_result[0])
print("Best Area:", best_result[1])
print("Best Reward:", best_reward)
print("Total time:", end - start)
