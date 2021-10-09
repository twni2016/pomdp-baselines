"""(Further) adapted from a2c_baselines/a2c_episodes.py

Ideally, we should only have to modify the trajectory segment generation
"""

import os
import os.path as osp
import numpy as np
import time
import tensorflow as tf
import joblib
import logging
import pickle

# from baselines.a2c.utils import find_trainable_variables, Scheduler, make_path, discount_with_dones
# from baselines.a2c.utils import mse

from ..baselines import logger
from ..baselines.common import explained_variance
from ..baselines.a2c.utils import (
    find_trainable_variables,
    Scheduler,
    make_path,
    discount_with_dones,
)
from tensorflow import losses

# from ..baselines.a2c.utils import mse

from ..ppo2_baselines.ppo2_episodes import constfn  # only needed for adaptive epsilon
from ..a2c_baselines.a2c_episodes import Runner as BaseRunner


class EPOptModel(object):
    """Modification of the Model class in a2c_episdoes.py which supports variable batch sizes for EPOpt"""

    def __init__(
        self,
        policy,
        ob_space,
        ac_space,
        nenvs,
        nsteps,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lr=7e-4,
        alpha=0.99,
        epsilon=1e-5,
        total_timesteps=int(20e6),
        lrschedule="linear",
    ):

        """
        sess = tf.get_default_session()
        nbatch = nenvs*nsteps

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)

        A = train_model.pdtype.sample_placeholder([nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])
        """

        # begin diff
        sess = tf.get_default_session()

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, None, reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        # end diff

        neglogpac = train_model.pd.neglogp(A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        # vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        entropy = tf.reduce_mean(train_model.pd.entropy())
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(
            learning_rate=LR, decay=alpha, epsilon=epsilon
        )
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: rewards, LR: cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train], td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class EPOptRunner(BaseRunner):
    """Modified the trajectory generator in A2C to follow EPOpt-e"""

    def run(
        self,
        *,
        # EPOpt specific - could go in __init__ but epsilon is callable
        paths,
        epsilon
    ):
        """Instead of doing a trajectory of nsteps (ie, "horizon"), do a
        sample N "paths" and then return the bottom epsilon-percentile
        """
        # Currently only works with single-threading
        assert self.env.num_envs == 1

        # Store all N trajectories sampled then return data of bottom-epsilon
        # lists -> lists of lists
        n_mb_obs, n_mb_rewards, n_mb_actions, n_mb_values, n_mb_dones = (
            [[] for _ in range(paths)],
            [[] for _ in range(paths)],
            [[] for _ in range(paths)],
            [[] for _ in range(paths)],
            [[] for _ in range(paths)],
        )
        num_episodes = 0
        mb_states = self.states
        for N in range(paths):

            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = (
                n_mb_obs[N],
                n_mb_rewards[N],
                n_mb_actions[N],
                n_mb_values[N],
                n_mb_dones[N],
            )
            for _ in range(self.env.venv.envs[0].spec.max_episode_steps):
                actions, values, states, _ = self.model.step(
                    self.obs, self.states, self.dones
                )
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions)
                mb_values.append(values)
                mb_dones.append(self.dones)
                obs, rewards, dones, _ = self.env.step(actions)
                self.states = states
                self.dones = dones
                for i, done in enumerate(dones):
                    if done:
                        self.obs[i] = self.obs[i] * 0
                self.obs = obs
                mb_rewards.append(rewards)
                # We only want to do one episode
                if self.dones:
                    break
            mb_dones.append(self.dones)

        # Compute the worst epsilon paths and concatenate them
        episode_returns = [sum(r) for r in n_mb_rewards]
        cutoff = np.percentile(episode_returns, 100 * epsilon)

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        for N in range(paths):
            # if n_mb_rewards[N] <= cutoff:
            if episode_returns[N] <= cutoff:
                # only count the episodes that are returned
                num_episodes += 1
                # "cache" values to keep track of final ones
                next_obs = n_mb_obs[N]
                next_rewards = n_mb_rewards[N]
                next_actions = n_mb_actions[N]
                next_values = n_mb_values[N]
                next_dones = n_mb_dones[N]
                # concatenate
                mb_obs.extend(next_obs)
                mb_rewards.extend(next_rewards)
                mb_actions.extend(next_actions)
                mb_values.extend(next_values)
                # when constructing mb_dones, only append
                # next_dones[:-1] except for the last episode
                mb_dones.extend(next_dones[:-1])
        mb_dones.append(next_dones[-1])

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).squeeze()
        # mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1,0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        if self.discrete:
            mb_actions = np.asarray(mb_actions, dtype=np.int).swapaxes(1, 0)
        else:
            mb_actions = np.asarray(mb_actions, dtype=np.float32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        # We can't just use self.obs etc, because the last of the N paths
        # may not be included in the update
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        # last_values = self.model.value(next_obs[-1], n_last_states[-1], next_dones[-1]).tolist()

        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(
            zip(mb_rewards, mb_dones, last_values)
        ):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(
                    rewards + [value], dones + [0], self.gamma
                )[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        if self.discrete:
            mb_actions = mb_actions.reshape(mb_rewards.shape)
        else:
            mb_actions = mb_actions.reshape(
                (mb_rewards.shape[0], self.ac_space.shape[0])
            )
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return (
            mb_obs,
            mb_states,
            mb_rewards,
            mb_masks,
            mb_actions,
            mb_values,
            num_episodes,
        )


def learn(
    policy,
    env,
    nsteps=5,
    total_episodes=int(10e3),
    max_timesteps=int(20e5),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule="linear",
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    save_interval=100,
    log_interval=100,
    keep_all_ckpt=False,
    paths=100,
    epopt_epsilon=1.0,  # EPOpt specific
):

    # In the original paper, epsilon is fixed to 1.0 for the first 100
    # "iterations" before updating to desired value
    if isinstance(epopt_epsilon, float):
        epopt_epsilon = constfn(epopt_epsilon)
    else:
        assert callable(epopt_epsilon)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda: EPOptModel(
        policy=policy,
        ob_space=ob_space,
        ac_space=ac_space,
        nenvs=nenvs,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        lr=lr,
        alpha=alpha,
        epsilon=epsilon,
        total_timesteps=max_timesteps,
        lrschedule=lrschedule,
    )
    if save_interval and logger.get_dir():
        import cloudpickle

        with open(osp.join(logger.get_dir(), "make_model.pkl"), "wb") as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = EPOptRunner(env, model, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs * nsteps
    tfirststart = time.time()
    update = 0
    episodes_so_far = 0
    old_savepath = None
    while True:
        update += 1
        if episodes_so_far >= total_episodes:
            break

        epsilonnow = epopt_epsilon(update)
        obs, states, rewards, masks, actions, values, num_episodes = runner.run(
            paths=paths, epsilon=epsilonnow
        )
        assert num_episodes == np.sum(masks), (num_episodes, np.sum(masks))
        episodes_so_far += num_episodes

        policy_loss, value_loss, policy_entropy = model.train(
            obs, states, rewards, masks, actions, values
        )
        nseconds = time.time() - tfirststart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("epsilon", epsilonnow)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("total_episodes", episodes_so_far)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("time_elapsed", nseconds)
            logger.dump_tabular()

        if (
            save_interval
            and logger.get_dir()
            and (update % save_interval == 0 or update == 1)
        ):
            checkdir = osp.join(logger.get_dir(), "checkpoints")
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, "%.5i" % update)
            print("Saving to", savepath)
            obs_norms = {}
            obs_norms["clipob"] = env.clipob
            obs_norms["mean"] = env.ob_rms.mean
            obs_norms["var"] = env.ob_rms.var + env.epsilon
            with open(osp.join(checkdir, "normalize"), "wb") as f:
                pickle.dump(obs_norms, f, pickle.HIGHEST_PROTOCOL)
            model.save(savepath)

            if not keep_all_ckpt and old_savepath:
                print("Removing previous checkpoint", old_savepath)
                os.remove(old_savepath)
            old_savepath = savepath

    env.close()
