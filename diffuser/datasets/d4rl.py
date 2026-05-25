import os
import collections
import numpy as np
import minari
import pickle
import pdb

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

from warp import pos

from diffuser.utils.path import project_path

MINARI_DATASETS = ['pointmaze-open-dense-v2', 'pointmaze-umaze-dense-v2', 'pointmaze-medium-dense-v2', 'pointmaze-large-dense-v2', 'antmaze-umaze-v1', 'antmaze-medium-diverse-v1', 'antmaze-large-diverse-v1']

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
 
#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def convert_minari_to_d4rl(dataset_minari):
    episodes_generator = dataset_minari.iterate_episodes()

    dataset = {}
    keys = ['observations', 'actions', 'rewards', 'terminals', 'timeouts']

    dataset = {key: [] for key in keys}

    for episode in episodes_generator:
        # dataset['observations'] = np.concatenate([episode.observations[key] for key in episode.observations], axis=1)
        dataset['observations'] = np.concatenate((episode.observations['observation'], episode.observations['desired_goal']), axis=1)
        dataset['actions'].append(episode.actions)
        dataset['rewards'].append(episode.rewards)
        dataset['terminals'].append(episode.terminations)
        # dataset['timeouts'].append(episode['timeouts'])

    return dataset


def get_dataset(env):
    if type(env) == str:
        dataset_minari = minari.load_dataset(env, download=True)

        dataset = dataset_minari.iterate_episodes()
        # dataset = convert_minari_to_d4rl(dataset_minari)
        # with open('data/' + env + '_dataset.pkl', 'rb') as f:
        #     dataset = pickle.load(f)
    else:
        dataset = env.get_dataset()

    return dataset

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """

    if env in MINARI_DATASETS:
        dataset = minari.load_dataset(env, download=True)
        episodes_generator = dataset.iterate_episodes()

        for episode in episodes_generator:
            if type(episode.observations) == dict:      # Minari dataset
                if 'antmaze' in env:
                    observations = np.concatenate((episode.observations['achieved_goal'], 
                                                   episode.observations['observation'], 
                                                   episode.observations['desired_goal']), 
                                                   axis=1)
                else:
                    observations = np.concatenate((episode.observations['observation'],
                                                   episode.observations['desired_goal']),
                                                   axis=1)
                goal_dim = episode.observations['desired_goal'].shape[1]
                observations[0, -goal_dim:] = observations[1, -goal_dim:]      # Ensure that the goal is already set in the first timestep (from previous episode)
            else:
                observations = episode.observations

            if observations.shape[0] == episode.actions.shape[0] + 1:
                observations = observations[:-1]
            
            episode_data = {
                'observations': observations,
                'actions': episode.actions,
                'rewards': episode.rewards,
                'terminals': episode.terminations
            }

            if 'antmaze' in env:
                first_index = np.where(np.linalg.norm(episode.observations['achieved_goal'] - episode.observations['desired_goal'], axis=1) <= 0.5)[0]
                if first_index.size == 0:
                    continue        # No successful episode
                else:
                    first_index = first_index[0]
                    episode_data['terminals'][first_index] = 1
                    for data_key in episode_data:
                        episode_data[data_key] = episode_data[data_key][:first_index+1]

            yield episode_data

    elif env == 'avoiding-d3il' or env == 'd3il-avoiding':
        print(f"DEBUG: sequence_dataset called with env='{env}'")
        # Load mobile robot data from .npy files
        import glob
        data_dir = project_path("isaac", "dataset", "avoiding", "data")
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        
        print(f"Loading {len(npy_files)} mobile robot trajectories from {data_dir}")
        
        for npy_file in npy_files:
            tmp = np.load(npy_file)
            subsample_rate = 1
            robot_pos = tmp[::subsample_rate, 1:3] # [x, y]
            
            # Compute world-frame velocities from position differences
            vel_state = robot_pos[1:] - robot_pos[:-1]
            valid_len = len(vel_state)
            
            # State: [x, y, x, y] - duplicate position to match 4D observation structure
            input_state = np.concatenate((robot_pos[:-1], robot_pos[:-1]), axis=-1)     #seed 5

            # curr = robot_pos[:-1]   # (T-1, 2)
            # des  = robot_pos[1:]    # (T-1, 2)  one-step-ahead desired
            # vel_state = des - curr    # (T-1, 2)  delta position per step
            # input_state = np.concatenate([des, curr], axis=-1)  # [x_des,y_des,x,y]
            # valid_len = len(vel_state)

            episode_data = {
                'observations': input_state,
                'actions': vel_state,
                'rewards': np.zeros(valid_len),
                'terminals': np.concatenate((np.zeros(valid_len-1), np.array([1])))
            }
            
            yield episode_data
        

    elif env == 'avoiding-d3il-v1' or env == 'd3il-avoiding-v1':
        data_dir = project_path("isaac", "dataset", "avoidingoriginal", "data")
        print("Resolved data path:", data_dir)
        state_files = os.listdir(data_dir)
        #here i think i have to change the action and observation for drone

        for file in state_files:
            with open(os.path.join(data_dir, file), 'rb') as f:
                path = os.path.join(data_dir, file)
                env_state = pickle.load(f)

                robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
                robot_c_pos = env_state['robot']['c_pos'][:, :2]

                input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)

                vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
                valid_len = len(vel_state)

            episode_data = {
                'observations': input_state[:-1],
                'actions': vel_state,
                'rewards': np.zeros(valid_len),
                'terminals': np.concatenate((np.zeros(valid_len-1), np.array([1])))
            }

            yield episode_data
    elif env == 'avoiding-crazyflie' or env == 'crazyflie-avoiding':
        
        data_dir = project_path("isaac", "dataset", "avoiding_crazyflie", "data")
        print("Resolved data path:", data_dir)
        # state_files = os.listdir(data_dir)
        state_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pkl")]) #prevents loading .py, .png
        
        for file in state_files:
            path = os.path.join(data_dir, file)
            # print("[DEBUG] loading env_state from:", path, flush=True)
            with open(path, 'rb') as f:
                env_state = pickle.load(f)
            
            states  = env_state["states"][0]               # (T, 13)
            # actions = env_state["actions_motor_forces"][0] # (T, 4)
            # targets = env_state["targets"][0]              # (T, 3)
            
            # ---- choice for observation ----
            pos    = states[:, 0:2]
            
            # Compute world-frame velocities from position differences
            vel_state = pos[1:] - pos[:-1]
            valid_len = len(vel_state)
            
            # State: [x, y, x, y] - duplicate position to match 4D observation structure
            input_state = np.concatenate((pos[:-1], pos[:-1]), axis=-1)     #seed 5

            episode_data = {
                'observations': input_state,
                'actions': vel_state,
                'rewards': np.zeros(valid_len),
                'terminals': np.concatenate((np.zeros(valid_len-1), np.array([1])))
            }

            yield episode_data
    else:
        raise NotImplementedError


