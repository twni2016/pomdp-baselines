import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Rectangle
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import gym
import numpy as np
from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import pylab as pl


sns.set(style="darkgrid")
cols_deep = sns.color_palette("deep", 10)
cols_dark = sns.color_palette("dark", 10)


def set_default_mpl():
    from matplotlib import cycler

    colors = cycler(
        "color", ["#EE6666", "#3388BB", "#9988DD", "#EECC55", "#88BB44", "#FFBBBB"]
    )
    plt.rc(
        "axes",
        facecolor="#E6E6E6",
        edgecolor="none",
        axisbelow=True,
        grid=True,
        prop_cycle=colors,
    )
    plt.rc("grid", color="w", linestyle="solid")
    # plt.rc('xtick', direction='out', color='gray')
    plt.rc("xtick", direction="out", color="k")
    # plt.rc('ytick', direction='out', color='gray')
    plt.rc("ytick", direction="out", color="k")
    plt.rc("patch", edgecolor="#E6E6E6")

    plt.rcParams.update({"font.size": 20})
    plt.rcParams.update({"xtick.labelsize": 15})
    plt.rcParams.update({"ytick.labelsize": 15})
    plt.rcParams.update({"axes.titlesize": 24})
    plt.rcParams.update({"axes.labelsize": 20})
    plt.rcParams.update({"lines.linewidth": 2})


def evaluate_vae(encoder, decoder, actions, rewards, states):
    """

    :param encoder: RNN encoder network
    :param decoder: reward decoder
    :param actions: array of actions of shape: (T, batch, action_dim)
    :param rewards: array of rewards of shape: (T, batch, 1)
    :param states: array of states of shape: (T, batch, state_dim)
    :return:
    """

    if actions.dim() != 3:
        actions = actions.unsqueeze(dim=0)
        states = states.unsqueeze(dim=0)
        rewards = rewards.unsqueeze(dim=0)

    T, batch_size, _ = actions.size()

    means, logvars, hidden_states, reward_preds = [], [], [], []
    with torch.no_grad():
        task_sample, task_mean, task_logvar, hidden_state = encoder.prior(batch_size)
    means.append(task_mean)
    logvars.append(task_logvar)
    hidden_states.append(hidden_state)
    reward_preds.append(ptu.get_numpy(decoder(task_sample, None)))

    for action, reward, state in zip(actions, rewards, states):
        action = action.unsqueeze(dim=0)
        state = state.unsqueeze(dim=0)
        reward = reward.unsqueeze(dim=0)
        with torch.no_grad():
            task_sample, task_mean, task_logvar, hidden_state = encoder(
                actions=action.float(),
                states=state,
                rewards=reward,
                hidden_state=hidden_state,
                return_prior=False,
            )
        means.append(task_mean.unsqueeze(dim=0))
        logvars.append(task_logvar.unsqueeze(dim=0))
        hidden_states.append(hidden_state)
        reward_preds.append(ptu.get_numpy(decoder(task_sample.unsqueeze(dim=0), None)))

    means = torch.cat(means, dim=0)
    logvars = torch.cat(logvars, dim=0)
    hidden_states = torch.cat(hidden_states, dim=0)
    reward_preds = np.vstack(reward_preds)
    return means, logvars, hidden_states, reward_preds


def rollout_policy(env, learner):
    is_vae_exist = "vae" in dir(learner)

    observations = []
    actions = []
    rewards = []
    values = []
    if is_vae_exist:
        latent_samples = []
        latent_means = []
        latent_logvars = []

    obs = ptu.from_numpy(env.reset())
    obs = obs.reshape(-1, obs.shape[-1])
    observations.append(obs)
    done_rollout = False
    if is_vae_exist:
        # get prior parameters
        with torch.no_grad():
            (
                task_sample,
                task_mean,
                task_logvar,
                hidden_state,
            ) = learner.vae.encoder.prior(batch_size=1)
        # store
        latent_samples.append(ptu.get_numpy(task_sample[0, 0]))
        latent_means.append(ptu.get_numpy(task_mean[0, 0]))
        latent_logvars.append(ptu.get_numpy(task_logvar[0, 0]))

    while not done_rollout:
        if is_vae_exist:
            # add distribution parameters to observation - policy is conditioned on posterior
            augmented_obs = learner.get_augmented_obs(
                obs=obs, task_mu=task_mean, task_std=task_logvar
            )
            with torch.no_grad():
                action, value = learner.agent.act(obs=augmented_obs, deterministic=True)
        else:
            action, _, _, _ = learner.agent.act(obs=obs)

        # observe reward and next obs
        next_obs, reward, done, info = utl.env_step(env, action.squeeze(dim=0))
        # store
        observations.append(next_obs)
        actions.append(action)
        values.append(value)
        rewards.append(reward.item())
        done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

        if is_vae_exist:
            # update encoding
            task_sample, task_mean, task_logvar, hidden_state = learner.vae.encoder(
                action,
                next_obs,
                reward.reshape((1, 1)),
                hidden_state,
                return_prior=False,
            )

            # values.append(value.item())
            latent_samples.append(ptu.get_numpy(task_sample[0]))
            latent_means.append(ptu.get_numpy(task_mean[0]))
            latent_logvars.append(ptu.get_numpy(task_logvar[0]))
        # set: obs <- next_obs
        obs = next_obs.clone()
    if is_vae_exist:
        return (
            observations,
            actions,
            rewards,
            values,
            latent_samples,
            latent_means,
            latent_logvars,
        )
    else:
        return observations, actions, rewards, values


def get_test_rollout(args, env, policy, encoder=None):
    num_episodes = args.max_rollouts_per_task

    # --- initialise things we want to keep track of ---

    episode_prev_obs = [[] for _ in range(num_episodes)]
    episode_next_obs = [[] for _ in range(num_episodes)]
    episode_actions = [[] for _ in range(num_episodes)]
    episode_rewards = [[] for _ in range(num_episodes)]

    episode_returns = []
    episode_lengths = []

    if encoder is not None:
        episode_latent_samples = [[] for _ in range(num_episodes)]
        episode_latent_means = [[] for _ in range(num_episodes)]
        episode_latent_logvars = [[] for _ in range(num_episodes)]
    else:
        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
        episode_latent_means = episode_latent_logvars = None
    # --- roll out policy ---

    # (re)set environment
    [obs_raw, obs_normalised] = env.reset()
    obs_raw = obs_raw.reshape((1, -1)).to(ptu.device)
    obs_normalised = obs_normalised.reshape((1, -1)).to(ptu.device)

    for episode_idx in range(num_episodes):

        curr_rollout_rew = []

        if encoder is not None:
            if episode_idx == 0 and encoder:
                # reset to prior
                (
                    curr_latent_sample,
                    curr_latent_mean,
                    curr_latent_logvar,
                    hidden_state,
                ) = encoder.prior(1)
                curr_latent_sample = curr_latent_sample[0].to(ptu.device)
                curr_latent_mean = curr_latent_mean[0].to(ptu.device)
                curr_latent_logvar = curr_latent_logvar[0].to(ptu.device)

            episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
            episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
            episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

        for step_idx in range(1, env._max_episode_steps + 1):

            episode_prev_obs[episode_idx].append(obs_raw.clone())

            _, action, _ = utl.select_action(
                args=args,
                policy=policy,
                obs=obs_normalised if args.norm_obs_for_policy else obs_raw,
                deterministic=True,
                task_sample=curr_latent_sample,
                task_mean=curr_latent_mean,
                task_logvar=curr_latent_logvar,
            )

            # observe reward and next obs
            (
                (obs_raw, obs_normalised),
                (rew_raw, rew_normalised),
                done,
                infos,
            ) = utl.env_step(env, action)
            obs_raw = obs_raw.reshape((1, -1)).to(ptu.device)
            obs_normalised = obs_normalised.reshape((1, -1)).to(ptu.device)

            if encoder is not None:
                # update task embedding
                (
                    curr_latent_sample,
                    curr_latent_mean,
                    curr_latent_logvar,
                    hidden_state,
                ) = encoder(
                    action.float().to(ptu.device),
                    obs_raw,
                    rew_raw.reshape((1, 1)).float().to(ptu.device),
                    hidden_state,
                    return_prior=False,
                )

                episode_latent_samples[episode_idx].append(
                    curr_latent_sample[0].clone()
                )
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(
                    curr_latent_logvar[0].clone()
                )

            episode_next_obs[episode_idx].append(obs_raw.clone())
            episode_rewards[episode_idx].append(rew_raw.clone())
            episode_actions[episode_idx].append(action.clone())

            if infos[0]["done_mdp"]:
                break

        episode_returns.append(sum(curr_rollout_rew))
        episode_lengths.append(step_idx)

    # clean up
    if encoder is not None:
        episode_latent_means = [torch.stack(e) for e in episode_latent_means]
        episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

    episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
    episode_next_obs = [torch.cat(e) for e in episode_next_obs]
    episode_actions = [torch.cat(e) for e in episode_actions]
    episode_rewards = [torch.cat(r) for r in episode_rewards]

    return (
        episode_latent_means,
        episode_latent_logvars,
        episode_prev_obs,
        episode_next_obs,
        episode_actions,
        episode_rewards,
        episode_returns,
    )


def plot_latents(
    latent_means, latent_logvars, rewards_preds, num_episodes, num_steps_per_episode
):
    """
    Plot mean/variance/pred_rewards over time
    """
    # set_default_mpl()

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 2)
    plt.plot(range(latent_means.shape[0]), latent_means, ".-", alpha=0.5)
    plt.plot(range(latent_means.shape[0]), latent_means.mean(axis=1), "k.-")
    for tj in np.cumsum([0, *[num_steps_per_episode for _ in range(num_episodes)]]):
        span = latent_means.max() - latent_means.min()
        plt.plot(
            [tj + 0.5, tj + 0.5],
            [latent_means.min() - span * 0.05, latent_means.max() + span * 0.05],
            "k--",
            alpha=0.5,
        )
    plt.xlabel("env steps", fontsize=15)
    plt.ylabel("latent mean", fontsize=15)

    plt.subplot(2, 2, 4)
    latent_vars = np.exp(latent_logvars)
    plt.plot(range(latent_vars.shape[0]), latent_vars, ".-", alpha=0.5)
    plt.plot(range(latent_vars.shape[0]), latent_vars.mean(axis=1), "k.-")
    for tj in np.cumsum([0, *[num_steps_per_episode for _ in range(num_episodes)]]):
        span = latent_vars.max() - latent_vars.min()
        plt.plot(
            [tj + 0.5, tj + 0.5],
            [latent_vars.min() - span * 0.05, latent_vars.max() + span * 0.05],
            "k--",
            alpha=0.5,
        )
    plt.xlabel("env steps", fontsize=15)
    plt.ylabel("latent variance", fontsize=15)

    plt.subplot(1, 2, 1)
    plt.plot(range(rewards_preds.shape[0]), rewards_preds, ".-", alpha=0.5)
    plt.plot(range(rewards_preds.shape[0]), rewards_preds.mean(axis=1), "k.-")
    for tj in np.cumsum([0, *[num_steps_per_episode for _ in range(num_episodes)]]):
        span = rewards_preds.max() - rewards_preds.min()
        plt.plot(
            [tj + 0.5, tj + 0.5],
            [rewards_preds.min() - span * 0.05, rewards_preds.max() + span * 0.05],
            "k--",
            alpha=0.5,
        )
    plt.xlabel("env steps", fontsize=15)
    plt.ylabel(r"$R^{+}=\mathbb{E}[P(R=1)]$ for each cell", fontsize=15)

    plt.tight_layout()
    plt.show()


def vis_rew_pred(args, rew_pred_arr, goal, **kwargs):
    env = gym.make(args.env_name)
    if args.env_name.startswith("GridNavi"):
        fig = plt.figure(figsize=(6, 6))
    else:  # 'TwoRooms'
        fig = plt.figure(figsize=(12, 6))

    ax = plt.gca()
    cmap = plt.cm.viridis
    for state in env.states:
        cell = Rectangle(
            (state[0], state[1]),
            width=1,
            height=1,
            fc=cmap(
                rew_pred_arr[ptu.get_numpy(env.task_to_id(ptu.FloatTensor(state)))[0]]
            ),
        )
        ax.add_patch(cell)
        ax.text(
            state[0] + 0.5,
            state[1] + 0.5,
            rew_pred_arr[ptu.get_numpy(env.task_to_id(ptu.FloatTensor(state)))[0]],
            ha="center",
            va="center",
            color="w",
        )

    plt.xlim(
        env.observation_space.low[0] - 0.1, env.observation_space.high[0] + 1 + 0.1
    )
    plt.ylim(
        env.observation_space.low[1] - 0.1, env.observation_space.high[1] + 1 + 0.1
    )

    # add goal's position on grid
    line = Line2D(
        [goal[0] + 0.3, goal[0] + 0.7],
        [goal[1] + 0.3, goal[1] + 0.7],
        lw=5,
        color="black",
        axes=ax,
    )
    ax.add_line(line)
    line = Line2D(
        [goal[0] + 0.3, goal[0] + 0.7],
        [goal[1] + 0.7, goal[1] + 0.3],
        lw=5,
        color="black",
        axes=ax,
    )
    ax.add_line(line)
    if "title" in kwargs:
        plt.title(kwargs["title"])

    if args.env_name.startswith("GridNavi"):
        ax.axis("equal")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both", length=0)

    fig.tight_layout()
    return fig


def plot_discretized_belief_halfcircle(
    belief_rewards, center_points, env, observations
):

    fig = plt.figure()
    env.plot_behavior(observations, plot_env=True, color=cols_deep[3], linewidth=5)
    res = center_points[1, 0] - center_points[0, 0]
    normal = pl.Normalize(0.0, 1.0)
    colors = pl.cm.gray(normal(belief_rewards))

    for (x, y), c in zip(center_points, colors):
        rec = Rectangle((x, y), res, res, facecolor=c, alpha=0.85, edgecolor="none")
        plt.gca().add_patch(rec)

    cax, _ = cbar.make_axes(plt.gca())
    cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.gray, norm=normal)
    return fig


def plot_rew_pred_vs_rew(rewards, reward_preds):
    fig = plt.figure()
    # plt.plot(range(len(rewards)), rewards, 'o--', color=cols_dark[3], label='rew')
    plt.scatter(range(len(rewards)), rewards, color=cols_dark[0], label="rew")
    # plt.plot(range(len(reward_preds)), reward_preds, 'o--', color=cols_dark[2], label='rew pred.')
    plt.scatter(
        range(len(reward_preds)), reward_preds, color=cols_dark[1], label="rew pred."
    )
    plt.legend()
    return fig


def plot_rollouts(observations, env):
    """
        very similar to visualize behaviour but targeted to TensorBoard vis.
    :param observations:
    :param env:
    :return:
    """
    episode_len = env.unwrapped._max_episode_steps
    assert (
        (len(observations) - 1) / episode_len
    ).is_integer(), "Error in observations length - env mismatch"

    if isinstance(observations, list):
        observations = torch.cat(observations)
    if (
        observations.shape[-1] > 2
    ):  # when 2 first dimensions are 2D position (PointRobot and AntSemiCircle)
        observations = observations[:, :2]

    num_episodes = int((len(observations) - 1) / episode_len)
    plot_env = True

    fig = plt.figure(figsize=(12, 10))

    for episode in range(num_episodes):
        env.plot_behavior(
            observations[episode * episode_len + 1 : (episode + 1) * episode_len + 1],
            plot_env=plot_env,
            color=cols_dark[episode],
            label="Episode {}".format(episode + 1),
        )
        plot_env = False  # after first time, do not plot env again, only rollouts
    plt.legend()

    return fig


def plot_visited_states(observations, env):
    # Targeted for 2D position tasks (PointRobot and AntSemiCircle)
    fig = plt.figure(figsize=(12, 10))
    env.plot_env()
    plt.scatter(observations[:, 0], observations[:, 1], color=cols_dark[3], marker=".")
    # sns.kdeplot(observations[:, 0], observations[:, 1], cmap="Reds", shade=True, shade_lowest=False)
    return fig


def predict_rewards(learner, means, logvars):
    reward_preds = ptu.zeros([means.shape[0], learner.env.num_states])
    for t in range(reward_preds.shape[0]):
        task_samples = learner.vae.encoder._sample_gaussian(
            ptu.FloatTensor(means[t]), ptu.FloatTensor(logvars[t]), num=50
        )
        reward_preds[t, :] = (
            learner.vae.reward_decoder(ptu.FloatTensor(task_samples), None)
            .mean(dim=0)
            .detach()
        )

    return ptu.get_numpy(reward_preds)


def visualize_bahavior(observations, env):
    """

    :param observations:
    :param env:
    :param num_episodes:
    :return:
    """

    episode_len = env.unwrapped._max_episode_steps
    assert (
        (len(observations) - 1) / episode_len
    ).is_integer(), "Error in observations length - env mismatch"

    if isinstance(observations, list):
        observations = torch.cat(observations)
    if (
        observations.shape[-1] > 2
    ):  # when 2 first dimensions are 2D position (PointRobot and AntSemiCircle)
        observations = observations[:, :2]

    num_episodes = int((len(observations) - 1) / episode_len)
    timesteps = np.linspace(1, episode_len, 4, dtype=int)

    fig = plt.figure(figsize=(10, 10))

    for episode in range(num_episodes):
        for t_i, timestep in enumerate(timesteps):
            plt.subplot(
                num_episodes, len(timesteps), t_i + 1 + episode * len(timesteps)
            )
            env.plot_behavior(
                torch.cat(
                    (
                        observations[:1, :],
                        observations[
                            episode * episode_len
                            + 1 : episode * episode_len
                            + 1
                            + timestep
                        ],
                    )
                )
            )
            if t_i == 0:
                plt.ylabel("Episode {}".format(episode + 1))
            if episode == 0:
                plt.title("t={}".format(timestep))

    # plt.show()    # commented for TB vis
    return fig


def sample_random_normal(dim, n_samples):
    return np.random.normal(size=(n_samples, dim))


def visualize_latent_space(latent_dim, n_samples, decoder):
    latents = ptu.FloatTensor(sample_random_normal(latent_dim, n_samples))

    pred_rewards = ptu.get_numpy(decoder(latents, None))
    goal_locations = np.argmax(pred_rewards, axis=-1)

    # embed to lower dim space - if dim > 2
    if latent_dim > 2:
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(latents)

    # create DataFrame
    data = tsne_results if latent_dim > 2 else latents

    df = pd.DataFrame(data, columns=["x1", "x2"])
    df["y"] = goal_locations

    fig = plt.figure(figsize=(6, 6))
    sns.scatterplot(
        x="x1",
        y="x2",
        hue="y",
        s=30,
        palette=sns.color_palette("hls", len(np.unique(df["y"]))),
        data=df,
        legend="full",
        ax=plt.gca(),
    )
    fig.show()

    return data, goal_locations
