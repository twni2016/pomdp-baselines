import numpy as np
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
import torch
import matplotlib.pyplot as plt
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HalfCheetahEnv(HalfCheetahEnv_):
    def _get_obs(self):
        return (
            np.concatenate(
                [
                    self.sim.data.qpos.flat[1:],
                    self.sim.data.qvel.flat,
                    self.get_body_com("torso").flat,
                ]
            )
            .astype(np.float32)
            .flatten()
        )

    def viewer_setup(self):
        camera_id = self.model.camera_name2id("track")
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        self.viewer._hide_overlay = True

    def render(self, mode="human"):
        if mode == "rgb_array":
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            return data
        elif mode == "human":
            self._get_viewer().render()

    @staticmethod
    def visualise_behaviour(
        env, args, policy, iter_idx, encoder=None, image_folder=None, **kwargs
    ):

        # TODO: are we going to use the decoders for anything? Some visualisations?

        num_episodes = args.max_rollouts_per_task
        unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

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
            sample_embeddings = args.sample_embeddings
        else:
            episode_latent_samples = (
                episode_latent_means
            ) = episode_latent_logvars = None
            sample_embeddings = False

        # --- roll out policy ---

        # (re)set environment
        env.reset_task()
        (obs_raw, obs_normalised) = env.reset()
        obs_raw = obs_raw.float().reshape((1, -1)).to(device)
        obs_normalised = obs_normalised.float().reshape((1, -1)).to(device)
        start_obs_raw = obs_raw.clone()

        # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
        if hasattr(args, "hidden_size"):
            hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        else:
            hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        task = env.get_task()
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        pos[0] = [unwrapped_env.get_body_com("torso")[0]]

        for episode_idx in range(num_episodes):

            curr_rollout_rew = []

            if episode_idx == 0:
                if encoder is not None:
                    # reset to prior
                    (
                        curr_latent_sample,
                        curr_latent_mean,
                        curr_latent_logvar,
                        hidden_state,
                    ) = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0].to(device)
                    curr_latent_mean = curr_latent_mean[0].to(device)
                    curr_latent_logvar = curr_latent_logvar[0].to(device)
                else:
                    curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

            if encoder is not None:
                episode_latent_samples[episode_idx].append(
                    curr_latent_sample[0].clone()
                )
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(
                    curr_latent_logvar[0].clone()
                )

            # keep track of position
            pos[episode_idx].append(unwrapped_env.get_body_com("torso")[0].copy())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs_raw.clone())
                else:
                    episode_prev_obs[episode_idx].append(obs_raw.clone())
                # act
                o_aug = utl.get_augmented_obs(
                    args,
                    obs_normalised if args.norm_obs_for_policy else obs_raw,
                    curr_latent_sample,
                    curr_latent_mean,
                    curr_latent_logvar,
                )
                _, action, _ = policy.act(o_aug, deterministic=True)

                (
                    (obs_raw, obs_normalised),
                    (rew_raw, rew_normalised),
                    done,
                    info,
                ) = env.step(action.cpu().detach())
                obs_raw = obs_raw.float().reshape((1, -1)).to(device)
                obs_normalised = obs_normalised.float().reshape((1, -1)).to(device)

                # keep track of position
                pos[episode_idx].append(unwrapped_env.get_body_com("torso")[0].copy())

                if encoder is not None:
                    # update task embedding
                    (
                        curr_latent_sample,
                        curr_latent_mean,
                        curr_latent_logvar,
                        hidden_state,
                    ) = encoder(
                        action.float().to(device),
                        obs_raw,
                        torch.tensor(rew_raw).reshape((1, 1)).float().to(device),
                        hidden_state,
                        return_prior=False,
                    )

                    episode_latent_samples[episode_idx].append(
                        curr_latent_sample[0].clone()
                    )
                    episode_latent_means[episode_idx].append(
                        curr_latent_mean[0].clone()
                    )
                    episode_latent_logvars[episode_idx].append(
                        curr_latent_logvar[0].clone()
                    )

                episode_next_obs[episode_idx].append(obs_raw.clone())
                episode_rewards[episode_idx].append(rew_raw.clone())
                episode_actions[episode_idx].append(action.clone())

                if info[0]["done_mdp"] and not done:
                    start_obs_raw = info[0]["start_state"]
                    start_obs_raw = (
                        torch.from_numpy(start_obs_raw)
                        .float()
                        .reshape((1, -1))
                        .to(device)
                    )
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
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        # plot the movement of the half-cheetah
        plt.figure(figsize=(7, 4 * num_episodes))
        min_x = min([min(p) for p in pos])
        max_x = max([max(p) for p in pos])
        span = max_x - min_x
        for i in range(num_episodes):
            plt.subplot(num_episodes, 1, i + 1)
            plt.plot(pos[i], range(len(pos[i])), "k")
            plt.title("task: ".format(task), fontsize=15)
            plt.ylabel("steps (ep {})".format(i), fontsize=15)
            if i == num_episodes - 1:
                plt.xlabel("position", fontsize=15)
            else:
                plt.xticks([])
            plt.xlim(min_x - 0.05 * span, max_x + 0.05 * span)
        plt.tight_layout()
        if image_folder is not None:
            plt.savefig("{}/{}_behaviour".format(image_folder, iter_idx))
            plt.close()
        else:
            plt.show()

        return (
            episode_latent_means,
            episode_latent_logvars,
            episode_prev_obs,
            episode_next_obs,
            episode_actions,
            episode_rewards,
            episode_returns,
        )
