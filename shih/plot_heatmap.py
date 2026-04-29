import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python plot_heatmap.py <path_to_action_distribution.npy>")
    sys.exit(1)

npy_path = sys.argv[1]
if not os.path.exists(npy_path):
    print(f"Error: File {npy_path} not found.")
    sys.exit(1)

action_dist = np.load(npy_path)

action_dist_t = action_dist.T

plt.figure(figsize=(30, 20))

vmax_value = 5

plt.imshow(action_dist_t, aspect='auto', cmap='magma', origin='lower', vmax=vmax_value)

cbar = plt.colorbar(label='Selection Frequency')

plt.title('Action Selection Distribution over Episodes (Enhanced Scale)')
plt.xlabel('Episode')
plt.ylabel('Gate Index')

output_filename = npy_path.replace('.npy', '_enhanced_heatmap.png')

plt.tight_layout()
plt.savefig(output_filename)
plt.close()

print(f"Enhanced heatmap saved to {output_filename}")