import os
import os.path as osp
import numpy as np
import time
import tensorflow as tf
import joblib
import logging
from baselines import logger
import pickle

from baselines.common import explained_variance
from baselines.a2c.utils import (
    find_trainable_variables,
    Scheduler,
    make_path,
    discount_with_dones,
)
from baselines.a2c.utils import mse


class Model(object):
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

        sess = tf.get_default_session()
        nbatch = nenvs * nsteps

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(
            sess, ob_space, ac_space, nenvs * nsteps, nsteps, reuse=True
        )

        A = train_model.pdtype.sample_placeholder([nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
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

        def train(
            obs,
            states,
            discounted_rewards,
            rewards,
            masks,
            prev_actions,
            actions,
            values,
            dones,
        ):
            advs = discounted_rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            # reshape actions, rewards, and dones to have first dimension of size nenvs*nsteps, existing second dimension
            # this is already done for obs
            rews = np.reshape(rewards, (nbatch, 1))
            ds = np.reshape(np.asarray(dones, dtype=np.float32), (nbatch, 1))
            if len(ac_space.shape) == 0:
                prev_actions = np.reshape(prev_actions, (nbatch,))
                one_hot = np.eye(ac_space.n)[prev_actions]
                for i in range(nbatch):
                    if prev_actions[i] == -1:
                        one_hot[i, :] = np.zeros((ac_space.n,), dtype=np.int)
                x = np.concatenate((obs, one_hot, rews, ds), axis=1)
                actions = np.reshape(actions, (nbatch,))
            else:
                prev_actions = np.reshape(prev_actions, (nbatch, ac_space.shape[0]))
                x = np.concatenate((obs, prev_actions, rews, ds), axis=1)
            td_map = {
                train_model.X: x,
                A: actions,
                ADV: advs,
                R: discounted_rewards,
                LR: cur_lr,
            }
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


class Runner(object):
    def __init__(self, env, model, nsteps=5, gamma=0.99, episodes_per_trial=5):
        self.env = env
        self.model = model
        self.nenv = env.num_envs
        ob_shape = env.observation_space.shape
        self.batch_ob_shape = (self.nenv * nsteps,) + ob_shape
        self.obs = np.zeros((self.nenv,) + ob_shape, dtype=np.float32)
        obs = env.reset([True for _ in range(self.nenv)])
        self.obs[:] = obs
        self.gamma = gamma
        self.nsteps = nsteps
        self.episodes_per_trial = episodes_per_trial
        self.states = model.initial_state
        self.dones = [False for _ in range(self.nenv)]
        self.masks = [False for _ in range(self.nenv)]
        self.rewards = [0.0 for _ in range(self.nenv)]
        self.episode_in_trial = [0 for _ in range(self.nenv)]
        ac_space = env.action_space
        self.ac_space = ac_space
        if len(ac_space.shape) == 0:
            self.discrete = True
            self.batch_ac_shape = (self.nenv * nsteps, 1)
            self.actions = [-1 for _ in range(self.nenv)]
        else:
            self.discrete = False
            self.batch_ac_shape = (self.nenv * nsteps, ac_space.shape[0])
            self.actions = np.zeros((self.nenv, ac_space.shape[0]), dtype=np.float32)

    def run(self):
        (
            mb_obs,
            prev_rewards,
            mb_rewards,
            prev_actions,
            mb_actions,
            mb_values,
            mb_dones,
            mb_masks,
        ) = ([], [], [], [], [], [], [], [])
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(
                self.obs,
                self.states,
                self.actions,
                self.rewards,
                self.dones,
                self.masks,
            )
            mb_obs.append(np.copy(self.obs))
            prev_actions.append(self.actions)
            mb_actions.append(actions)
            mb_values.append(values)
            prev_rewards.append(self.rewards)
            mb_masks.append(self.masks)
            mb_dones.append(self.dones)
            # if end_of_trial, if episode gets done in the next step, we need to reset environment parameters
            end_of_trial = [
                self.episode_in_trial[i] == (self.episodes_per_trial - 1)
                for i in range(self.nenv)
            ]
            obs, rewards, dones, _ = self.env.step(actions, end_of_trial)
            self.actions = actions
            self.states = states
            self.dones = dones
            self.masks = [False for _ in range(self.nenv)]
            self.rewards = rewards
            self.obs[:] = obs
            mb_rewards.append(rewards)
            for i, done in enumerate(self.dones):
                if done:
                    self.episode_in_trial[
                        i
                    ] += 1  # episode finished in the current step
                    self.episode_in_trial[i] %= self.episodes_per_trial
                    if self.episode_in_trial[i] == 0:
                        self.masks[i] = True
                        self.rewards[i] = 0.0
                        self.dones[i] = False
                        if self.discrete:
                            self.actions[i] = -1
                        else:
                            self.actions[i] = np.zeros(
                                (self.ac_space.shape[0]), dtype=np.float32
                            )
        mb_masks.append(self.masks)  # nsteps+1 records
        # batch of steps to batch of rollouts
        mb_obs = (
            np.asarray(mb_obs, dtype=np.float32)
            .swapaxes(1, 0)
            .reshape(self.batch_ob_shape)
        )
        prev_rewards = np.asarray(prev_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        if self.discrete:
            prev_actions = np.asarray(prev_actions, dtype=np.int).swapaxes(1, 0)
            mb_actions = np.asarray(mb_actions, dtype=np.int).swapaxes(1, 0)
        else:
            prev_actions = np.asarray(prev_actions, dtype=np.float32).swapaxes(1, 0)
            mb_actions = np.asarray(mb_actions, dtype=np.float32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = np.asarray(mb_masks, dtype=np.bool).swapaxes(1, 0)
        last_values = self.model.value(
            self.obs, self.states, self.actions, self.rewards, self.dones, self.masks
        ).tolist()
        num_trials = np.sum(mb_masks[:, 1:])
        # discount/bootstrap off value fn
        discounted_rewards = []
        for n, (rewards, masks, value) in enumerate(
            zip(mb_rewards, mb_masks[:, 1:], last_values)
        ):
            rewards = rewards.tolist()
            masks = masks.tolist()
            if masks[-1] == 0:
                rewards = discount_with_dones(
                    rewards + [value], masks + [0], self.gamma
                )[:-1]
            else:
                rewards = discount_with_dones(rewards, masks, self.gamma)
            discounted_rewards.append(rewards)
        discounted_rewards = np.asarray(discounted_rewards, dtype=np.float32)
        prev_rewards = prev_rewards.flatten()
        discounted_rewards = discounted_rewards.flatten()
        prev_actions = prev_actions.reshape(self.batch_ac_shape)
        mb_actions = mb_actions.reshape(self.batch_ac_shape)
        mb_values = mb_values.flatten()
        mb_masks = mb_masks[:, :-1]
        mb_masks = mb_masks.flatten()
        mb_dones = mb_dones.flatten()
        return (
            mb_obs,
            mb_states,
            discounted_rewards,
            prev_rewards,
            mb_masks,
            prev_actions,
            mb_actions,
            mb_values,
            mb_dones,
            num_trials,
        )


def learn(
    policy,
    env,
    nsteps=5,
    total_trials=1000,
    episodes_per_trial=5,
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
):

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda: Model(
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
    runner = Runner(
        env, model, nsteps=nsteps, gamma=gamma, episodes_per_trial=episodes_per_trial
    )
    # Episodes per trial is needed to compute the masks, which is done in the Runner class

    nbatch = nenvs * nsteps
    tfirststart = time.time()
    update = 0
    trials_so_far = 0
    old_savepath = None
    while True:
        update += 1
        if trials_so_far >= total_trials:
            break

        (
            obs,
            states,
            discounted,
            rewards,
            masks,
            prev_actions,
            actions,
            values,
            dones,
            num_trials,
        ) = runner.run()
        trials_so_far += num_trials

        policy_loss, value_loss, policy_entropy = model.train(
            obs,
            states,
            discounted,
            rewards,
            masks,
            prev_actions,
            actions,
            values,
            dones,
        )
        nseconds = time.time() - tfirststart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("total_trials", trials_so_far)
            logger.record_tabular("total_episodes", trials_so_far * episodes_per_trial)
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
