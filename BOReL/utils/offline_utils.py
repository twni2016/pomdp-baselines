import os
import numpy as np
import gym
import torch
from utils import helpers as utl
import matplotlib.pyplot as plt
from torchkit import pytorch_utils as ptu
from environments.make_env import make_env


def get_model_path(folder, model_name="encoder", iteration=None):
    """

    :param folder:
    :param model_name: from available file names in folder
    :param iteration:
    :return:
    """

    files = os.listdir(folder)
    files = [file if file.startswith(model_name) else None for file in files]
    files = list(filter(None, files))
    # check if any files left
    assert len(files) > 0, "model_name is invalid/irrelevant!"

    # get iteration indices
    idx = [file[len(model_name) : -3] for file in files]
    if iteration is not None:
        if type(iteration) != str:
            iteration = str(iteration)
        # check that iteration file exist
        assert iteration in idx, "iteration number is invalid! no such file."
        path = os.path.join(folder, files[idx.index(iteration)])
    else:  # get final iteration
        last_idx = idx.index(str(np.array(idx, dtype=int).max()))
        path = os.path.join(folder, files[last_idx])

    return path


def extract_goal_from_path(path):
    """
    Assumes path name structure of seed_{}_goal_{}_{}_..._{} (seed, goal description)
    """
    goal_desc = path.split("goal_")[-1]
    goal = [float(goal_part) for goal_part in goal_desc.split("_")]
    return goal


def create_goal_path_ext_from_goal(goal):
    """
    Assumes goal is array or list and returns goal_{}_{}_..._{} (goal description)
    """
    # print(type(goal))
    # print('yes' if type(goal) is np.ndarray else 'no')
    # print(goal.ndim)
    if not isinstance(goal, list) and goal.ndim == 0:
        goal = [goal]
    goal_desc = "goal"
    for goal_part in goal:
        goal_desc += "_" + str(goal_part.round(3))
    return goal_desc


def vis_train_tasks(env, goals):
    """
    for 2D goal tasks
    """
    env.plot_env()
    for goal in goals:
        circle = plt.Circle((goal[0], goal[1]), radius=env.goal_radius, alpha=0.3)
        plt.gca().add_artist(circle)


def expand_args(args, include_act_space=False):
    # create env to get parameters
    env = make_env(args.env_name, args.max_rollouts_per_task, seed=args.seed, n_tasks=1)

    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        args.action_dim = 1
    else:
        args.action_dim = env.action_space.shape[0]
    args.obs_dim = env.observation_space.shape[0]

    args.trajectory_len = env.unwrapped._max_episode_steps * args.max_rollouts_per_task
    args.num_states = (
        env.unwrapped.num_states if hasattr(env.unwrapped, "num_states") else None
    )
    if include_act_space:
        args.act_space = env.action_space
    return args, env


def load_transitions(path, device=ptu.device):
    """
        return arrays of obs, action ,rewards, next_obs, terminals
    :param path: path to directory in which there are numpy files
    :return:
    """

    obs = ptu.FloatTensor(np.load(os.path.join(path, "obs.npy"))).to(device)
    actions = ptu.FloatTensor(np.load(os.path.join(path, "actions.npy"))).to(device)
    rewards = ptu.FloatTensor(np.load(os.path.join(path, "rewards.npy"))).to(device)
    next_obs = ptu.FloatTensor(np.load(os.path.join(path, "next_obs.npy"))).to(device)
    terminals = ptu.FloatTensor(np.load(os.path.join(path, "terminals.npy"))).to(device)
    return obs, actions, rewards, next_obs, terminals


def create_rewards_arr(env, path):
    """
    creates rewards array from observations and actions
    mainly for mujoco, where solving single sparse task is not easy.
    """
    obs = np.load(os.path.join(path, "obs.npy"))
    actions = np.load(os.path.join(path, "actions.npy"))
    rewards = np.zeros((obs.shape[0], 1))
    for i in range(len(rewards)):
        rewards[i] = env.unwrapped.reward(obs[i], actions[i])

    np.save(os.path.join(path, "rewards.npy"), rewards)


def save_transitions(path, obs, actions, rewards, next_obs, terminals):
    """
        similar to load_transitions
    :param path: path to directory in which there are numpy files
    :return:
    """
    os.makedirs(path, exist_ok=True)
    # np.save(os.path.join(path, 'obs'), ptu.get_numpy(obs))
    # np.save(os.path.join(path, 'actions'), ptu.get_numpy(actions))
    # np.save(os.path.join(path, 'rewards'), ptu.get_numpy(rewards))
    # np.save(os.path.join(path, 'next_obs'), ptu.get_numpy(next_obs))
    # np.save(os.path.join(path, 'terminals'), ptu.get_numpy(terminals))
    np.save(os.path.join(path, "obs"), obs)
    np.save(os.path.join(path, "actions"), actions)
    np.save(os.path.join(path, "rewards"), rewards)
    np.save(os.path.join(path, "next_obs"), next_obs)
    np.save(os.path.join(path, "terminals"), terminals)


def load_trained_vae(vae, path):
    paths = {
        "encoder_path": get_model_path(path, model_name="encoder"),
        "reward_decoder_path": get_model_path(path, model_name="reward_decoder"),
    }

    vae.load_model(**paths)


def load_dataset(
    data_dir, args, num_tasks=None, allow_dense_data_loading=True, arr_type="tensor"
):
    dataset = []
    env_dir = (
        args.env_name.replace("Sparse", "")
        if "dense_train_sparse_test" in args
        and args.dense_train_sparse_test is True
        and allow_dense_data_loading
        else args.env_name
    )
    exps_dir = os.path.join(args.main_data_dir, env_dir, data_dir)
    goals = []
    all_dirs = os.listdir(exps_dir)
    if num_tasks is None:
        tasks = np.random.permutation(len(all_dirs))
    else:
        tasks = np.random.choice(len(all_dirs), num_tasks)
    for i, task in enumerate(tasks):
        exp_dir = os.path.join(exps_dir, all_dirs[task])
        goals.append(extract_goal_from_path(all_dirs[task]))
        if "rewards.npy" not in os.listdir(exp_dir):
            print("rewards.npy file doesn't exist. Creating it..")
            env = make_env(args.env_name, args.max_rollouts_per_task, n_tasks=1)
            create_rewards_arr(env, path=exp_dir)
            print("Created rewards.npy file.")
        obs, actions, rewards, next_obs, terminals = load_transitions(exp_dir)

        if obs.dim() < 3:
            obs = obs.reshape(-1, args.trajectory_len, obs.shape[-1]).transpose(0, 1)
            actions = actions.reshape(
                -1, args.trajectory_len, actions.shape[-1]
            ).transpose(0, 1)
            rewards = rewards.reshape(
                -1, args.trajectory_len, rewards.shape[-1]
            ).transpose(0, 1)
            next_obs = next_obs.reshape(
                -1, args.trajectory_len, next_obs.shape[-1]
            ).transpose(0, 1)
            terminals = terminals.reshape(
                -1, args.trajectory_len, terminals.shape[-1]
            ).transpose(0, 1)
            if args.num_trajs_per_task is not None:
                obs = obs[:, : args.num_trajs_per_task, :]
                actions = actions[:, : args.num_trajs_per_task, :]
                rewards = rewards[:, : args.num_trajs_per_task, :]
                next_obs = next_obs[:, : args.num_trajs_per_task, :]
                terminals = terminals[:, : args.num_trajs_per_task, :]
        else:
            if args.num_trajs_per_task is not None:
                obs = obs[:, : args.num_trajs_per_task, :]
                actions = actions[:, : args.num_trajs_per_task, :]
                rewards = rewards[:, : args.num_trajs_per_task, :]
                next_obs = next_obs[:, : args.num_trajs_per_task, :]
                terminals = terminals[:, : args.num_trajs_per_task, :]
            obs = obs.transpose(0, 1).reshape(-1, obs.shape[-1])
            actions = actions.transpose(0, 1).reshape(-1, actions.shape[-1])
            rewards = rewards.transpose(0, 1).reshape(-1, rewards.shape[-1])
            next_obs = next_obs.transpose(0, 1).reshape(-1, next_obs.shape[-1])
            terminals = terminals.transpose(0, 1).reshape(-1, terminals.shape[-1])

        if arr_type == "numpy":
            obs = ptu.get_numpy(obs)
            actions = ptu.get_numpy(actions)
            rewards = ptu.get_numpy(rewards)
            next_obs = ptu.get_numpy(next_obs)
            terminals = ptu.get_numpy(terminals)

        dataset.append([obs, actions, rewards, next_obs, terminals])
        # print(exp_dir)
        # print('Obs shape: ' + str(np.shape(dataset[-1][0])) +
        #       '. Act shape: ' + str(np.shape(dataset[-1][1])) +
        #       '. Reward shape: ' + str(np.shape(dataset[-1][2])) +
        #       '. Next obs shape: ' + str(np.shape(dataset[-1][3])))
    print("{} experiments loaded.".format(i + 1))
    goals = np.vstack(goals)

    return dataset, goals


def save_dataset(path, dataset, goals):
    for goal, set in zip(goals, dataset):
        save_path = os.path.join(path, create_goal_path_ext_from_goal(goal))
        # save_path = os.path.join(path, 'goal_{}_{}'.format(goal[0], goal[1]))
        save_transitions(
            save_path,
            obs=set[0],
            actions=set[1],
            rewards=set[2],
            next_obs=set[3],
            terminals=set[4],
        )


def batch_to_trajectories(dataset, args):
    traj_dataset = []
    for set in dataset:
        obs, actions, rewards, next_obs, terminals = (
            set[0],
            set[1],
            set[2],
            set[3],
            set[4],
        )
        obs = obs.reshape(-1, args.trajectory_len, obs.shape[-1]).transpose(0, 1)
        actions = actions.reshape(-1, args.trajectory_len, actions.shape[-1]).transpose(
            0, 1
        )
        rewards = rewards.reshape(-1, args.trajectory_len, rewards.shape[-1]).transpose(
            0, 1
        )
        next_obs = next_obs.reshape(
            -1, args.trajectory_len, next_obs.shape[-1]
        ).transpose(0, 1)
        terminals = terminals.reshape(
            -1, args.trajectory_len, terminals.shape[-1]
        ).transpose(0, 1)
        traj_dataset.append([obs, actions, rewards, next_obs, terminals])
    return traj_dataset


def trajectories_to_batch(dataset):
    traj_dataset = []
    for set in dataset:
        obs, actions, rewards, next_obs, terminals = (
            set[0],
            set[1],
            set[2],
            set[3],
            set[4],
        )
        obs = ptu.get_numpy(obs.transpose(0, 1).reshape(-1, obs.shape[-1]))
        actions = ptu.get_numpy(actions.transpose(0, 1).reshape(-1, actions.shape[-1]))
        rewards = ptu.get_numpy(rewards.transpose(0, 1).reshape(-1, rewards.shape[-1]))
        next_obs = ptu.get_numpy(
            next_obs.transpose(0, 1).reshape(-1, next_obs.shape[-1])
        )
        terminals = ptu.get_numpy(
            terminals.transpose(0, 1).reshape(-1, terminals.shape[-1])
        )
        traj_dataset.append([obs, actions, rewards, next_obs, terminals])
    return traj_dataset


def transform_mdp_to_bamdp_rollouts(
    vae, args, obs, actions, rewards, next_obs, terminals
):
    """

    :param vae:
    :param args:
    :param obs: shape (trajectory_len, n_rollouts, dim)
    :param actions:
    :param rewards:
    :param next_obs:
    :param terminals:
    :return:
    """

    # augmented_obs = ptu.zeros((obs.shape[0], obs.shape[1] + 2 * args.task_embedding_size))
    augmented_obs = ptu.zeros(
        (obs.shape[0], obs.shape[1], obs.shape[2] + 2 * args.task_embedding_size)
    )
    # augmented_next_obs = ptu.zeros((obs.shape[0], obs.shape[1] + 2 * args.task_embedding_size))
    augmented_next_obs = ptu.zeros(
        (obs.shape[0], obs.shape[1], obs.shape[2] + 2 * args.task_embedding_size)
    )
    if args.belief_rewards:
        belief_rewards = ptu.zeros_like(rewards)
    else:
        belief_rewards = None

    with torch.no_grad():
        # _, mean, logvar, hidden_state = vae.encoder.prior(batch_size=1)
        _, mean, logvar, hidden_state = vae.encoder.prior(batch_size=obs.shape[1])
        augmented_obs[0, :, :] = torch.cat((obs[0], mean[0], logvar[0]), dim=-1)
    for step in range(args.trajectory_len):
        # update encoding
        _, mean, logvar, hidden_state = utl.update_encoding(
            encoder=vae.encoder,
            obs=next_obs[step].unsqueeze(0),
            action=actions[step].unsqueeze(0),
            reward=rewards[step].unsqueeze(0),
            done=terminals[step].unsqueeze(0),
            hidden_state=hidden_state,
        )

        # augment data
        augmented_next_obs[step, :, :] = torch.cat(
            (next_obs[step], mean, logvar), dim=-1
        )
        if args.belief_rewards:
            with torch.no_grad():
                belief_rewards[step, :, :] = vae.compute_belief_reward(
                    mean.unsqueeze(dim=0),
                    logvar.unsqueeze(dim=0),
                    obs[step].unsqueeze(dim=0),
                    next_obs[step].unsqueeze(dim=0),
                    actions[step].unsqueeze(dim=0),
                )

    augmented_obs[1:, :, :] = augmented_next_obs[:-1, :, :].clone()

    return augmented_obs, belief_rewards, augmented_next_obs


def transform_mdps_ds_to_bamdp_ds(dataset, vae, args):
    """

    :param dataset: list of lists of lists. each list is list of arrays
    (s,a,r,s',done) arrays of size (traj_len, n_trajs, dim)
    :param vae: trained vae model
    :return:
    """

    bamdp_dataset = []

    for i, set in enumerate(dataset):
        obs, actions, rewards, next_obs, terminals = set
        (
            augmented_obs,
            belief_rewards,
            augmented_next_obs,
        ) = transform_mdp_to_bamdp_rollouts(
            vae,
            args,
            ptu.FloatTensor(obs),
            ptu.FloatTensor(actions),
            ptu.FloatTensor(rewards),
            ptu.FloatTensor(next_obs),
            ptu.FloatTensor(terminals),
        )
        rewards = (
            belief_rewards if belief_rewards is not None else ptu.FloatTensor(rewards)
        )

        bamdp_dataset.append(
            [
                ptu.get_numpy(augmented_obs),
                actions,
                ptu.get_numpy(rewards),
                ptu.get_numpy(augmented_next_obs),
                terminals,
            ]
        )
        print("{} datasets were processed.".format(i + 1))
    return bamdp_dataset


def mix_task_rollouts(dataset, env, goals, args, fraction=1.0):
    if args.max_rollouts_per_task == 2:
        mix_until_time = int(args.trajectory_len / args.max_rollouts_per_task)
        num_rollouts = dataset[0][0].shape[1]
        num_rollots_to_mix = int(num_rollouts * fraction)
        num_tasks = len(dataset)
        for i, set in enumerate(dataset):
            rollouts_to_mix = np.random.choice(num_rollouts, num_rollots_to_mix)
            tasks_to_mix_from = np.random.choice(
                list(range(i)) + list(range(i + 1, num_tasks)), num_rollots_to_mix
            )
            for rollout_idx, task in zip(rollouts_to_mix, tasks_to_mix_from):
                mixed_from_rollout = np.random.choice(
                    num_rollouts
                )  # which rollout to switch with
                set[0][:mix_until_time, rollout_idx, :] = dataset[task][0][
                    :mix_until_time, mixed_from_rollout, :
                ]
                set[1][:mix_until_time, rollout_idx, :] = dataset[task][1][
                    :mix_until_time, mixed_from_rollout, :
                ]
                set[2][:mix_until_time, rollout_idx, :] = relabel_rollout(
                    env,
                    goals[i],
                    dataset[task][3][:mix_until_time, mixed_from_rollout, :],
                    # dataset[task][0][:mix_until_time, mixed_from_rollout, :],
                    dataset[task][1][:mix_until_time, mixed_from_rollout, :],
                )
                set[3][:mix_until_time, rollout_idx, :] = dataset[task][3][
                    :mix_until_time, mixed_from_rollout, :
                ]
                set[4][:mix_until_time, rollout_idx, :] = dataset[task][4][
                    :mix_until_time, mixed_from_rollout, :
                ]

            print("Mixed {} datasets.".format(i + 1))
    else:
        single_traj_len = int(args.trajectory_len / args.max_rollouts_per_task)
        mix_times = np.arange(
            0, (args.trajectory_len + 1e-6) / 2, single_traj_len
        ).astype(int)
        num_rollouts = dataset[0][0].shape[1]
        num_rollots_to_mix = int(num_rollouts * fraction)
        num_tasks = len(dataset)
        for i, set in enumerate(dataset):
            rollouts_to_mix = np.random.choice(num_rollouts, num_rollots_to_mix)
            for (mix_start, mix_end) in zip(mix_times[:-1], mix_times[1:]):
                tasks_to_mix_from = np.random.choice(
                    list(range(i)) + list(range(i + 1, num_tasks)), num_rollots_to_mix
                )
                for rollout_idx, task in zip(rollouts_to_mix, tasks_to_mix_from):
                    mixed_from_rollout = np.random.choice(
                        num_rollouts
                    )  # which rollout timestep to switch with
                    set[0][mix_start:mix_end, rollout_idx, :] = dataset[task][0][
                        mix_start:mix_end, mixed_from_rollout, :
                    ]
                    set[1][mix_start:mix_end, rollout_idx, :] = dataset[task][1][
                        mix_start:mix_end, mixed_from_rollout, :
                    ]
                    set[2][mix_start:mix_end, rollout_idx, :] = relabel_rollout(
                        env,
                        goals[i],
                        dataset[task][3][mix_start:mix_end, mixed_from_rollout, :],
                        dataset[task][1][mix_start:mix_end, mixed_from_rollout, :],
                    )

                    set[3][mix_start:mix_end, rollout_idx, :] = dataset[task][3][
                        mix_start:mix_end, mixed_from_rollout, :
                    ]
                    set[4][mix_start:mix_end, rollout_idx, :] = dataset[task][4][
                        mix_start:mix_end, mixed_from_rollout, :
                    ]

            print("Mixed {} datasets.".format(i + 1))
    return dataset, goals


def relabel_rollout(env, goal, observations, actions):
    env.set_goal(goal)
    rewards = [
        env.reward(obs, action)
        for (obs, action) in zip(
            ptu.get_numpy(observations)
            if type(observations) is not np.ndarray
            else observations,
            ptu.get_numpy(actions) if type(actions) is not np.ndarray else actions,
        )
    ]
    if type(observations) is np.ndarray:
        return np.vstack(rewards)
    else:
        return ptu.FloatTensor(np.vstack(rewards))
