import re
import sys
import subprocess

design_path = {
    "bar": "benchmarks/bar.blif",
    "multiplier": "benchmarks/multiplier.blif",
    "priority": "benchmarks/priority.blif",
    "sin": "benchmarks/sin.blif",
    "sqrt": "benchmarks/sqrt.blif",
    "voter": "benchmarks/voter.blif",
}

library_path = {
    "7nm": "7nm.lib",
    # "nan45": "nan45.lib",
    # "sky130": "sky130.lib",
}

for library, lib_path in library_path.items():
    for design, design_path in design_path.items():
        temp_blif = f"temp_blifs/{design}.blif"
        abc_cmd = f"read {lib_path}; read {design_path}; map; write {temp_blif}; read {lib_path}; read -m {temp_blif}; topo; upsize; dnsize; stime;"
        res = subprocess.check_output(('abc', '-c', abc_cmd))
        match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
        match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
        max_delay = float(match_d.group(1))
        max_area = float(match_a.group(1))
        print(f"{library} {design} Delay: {max_delay}, Area: {max_area}")