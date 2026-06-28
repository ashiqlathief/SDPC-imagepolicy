
import os
import pickle
import numpy as np

from isaac.dataset.sim_path import sim_framework_path

data_dir = sim_framework_path("isaac", "dataset", "avoiding_crazyflie", "data")

# list only pkl files
data_list = sorted([f for f in os.listdir(data_dir) if f.endswith(".pkl")])

episode_counts = []
all_episode_lengths = []

files_with_2plus = []

for file in data_list:
    path = os.path.join(data_dir, file)

    with open(path, "rb") as fp:
        d = pickle.load(fp)

    # number of episodes in this file
    n_eps = len(d["states"])
    episode_counts.append(n_eps)

    if n_eps >= 2:
        files_with_2plus.append((file, n_eps))

    # collect lengths of each episode in this file
    for ep in d["states"]:
        # ep shape: (T, state_dim)
        all_episode_lengths.append(ep.shape[0])

episode_counts = np.array(episode_counts, dtype=int)
all_episode_lengths = np.array(all_episode_lengths, dtype=int)

print(f"Total .pkl files found: {len(data_list)}")
print(f"Total episodes found: {len(all_episode_lengths)}")

print("\nFiles with 2+ episodes:")
if len(files_with_2plus) == 0:
    print("  (none)")
else:
    for fname, n_eps in files_with_2plus:
        print(f"  {fname}: {n_eps} episodes")

print("\nEpisode length stats (timesteps):")
print("data points (sum of lengths):", int(np.sum(all_episode_lengths)))
print("mean:", float(np.mean(all_episode_lengths)))
print("std :", float(np.std(all_episode_lengths)))
print("max :", int(np.max(all_episode_lengths)))
print("min :", int(np.min(all_episode_lengths)))

print("\nPer-file episode-count stats:")
print("mean episodes/file:", float(np.mean(episode_counts)))
print("max  episodes/file:", int(np.max(episode_counts)))
print("min  episodes/file:", int(np.min(episode_counts)))
