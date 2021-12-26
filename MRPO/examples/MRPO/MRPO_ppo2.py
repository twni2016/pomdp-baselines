# -*- coding: future_fstrings -*-
import os
import time
import joblib
import numpy as np
import os.path as osp
from utils import logger
import pickle
import tensorflow as tf
from collections import deque
from MRPO.examples.baselines.common import explained_variance

from MRPO.examples.ppo2_baselines.ppo2_episodes import constfn, sf01
from MRPO.examples.ppo2_baselines.ppo2_episodes import Runner as BaseRunner
from MRPO.examples.MRPO import base
from MRPO.examples.baselines import bench
from MRPO.examples.baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from MRPO.examples.baselines.common.vec_env.vec_normalize import VecNormalize


class MRPOModel(object):
    def __init__(
        self,
        *,
        policy,
        ob_space,
        ac_space,
        nbatch_act,
        nbatch_train,
        nsteps,
        ent_coef,
        vf_coef,
        max_grad_norm
    ):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        # begin diff
        train_model = policy(sess, ob_space, ac_space, nbatch_act, None, reuse=True)
        # end diff

        A = train_model.pdtype.sample_placeholder([None])  # action
        ADV = tf.placeholder(tf.float32, [None])  # adavantage
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])
        neglogpac = train_model.pd.neglogp(A)  # old policy

        # entropy bonus
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # value loss
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(
            train_model.vf - OLDVPRED, -CLIPRANGE, CLIPRANGE
        )
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # clipped surrogate objective
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        approxkl = 0.5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(
            tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE))
        )
        loss = (
            pg_loss - entropy * ent_coef + vf_loss * vf_coef
        )  # L^{CLIP}, entropy bonus, L^{VF}

        # train the model, separate gradients computation and update enables operations to gradients, like clip
        with tf.variable_scope("model"):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(
            lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None
        ):
            advs = returns - values  # returns is the estimator for Q
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {
                train_model.X: obs,
                A: actions,
                ADV: advs,
                R: returns,
                LR: lr,
                CLIPRANGE: cliprange,
                OLDNEGLOGPAC: neglogpacs,
                OLDVPRED: values,
            }
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train], td_map
            )[:-1]

        self.loss_names = [
            "policy_loss",
            "value_loss",
            "policy_entropy",
            "approxkl",
            "clipfrac",
        ]

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101


class MRPORunner(BaseRunner):
    """Modified the trajectory generator in PPO2 to follow EPOpt-e"""

    def __init__(self, env, model, nsteps, gamma, lam):
        super(MRPORunner, self).__init__(
            env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam
        )
        self.gamma_lam = gamma / (1 - gamma) ** 2
        # self.masses = [0.75, 0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 1.0, 1.25]
        # self.lengths = [0.75, 0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 1.0, 1.25]
        # self.masses = [0.75, 0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 1.0, 1.25]
        # self.lengths = [0.75, 0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 1.0, 1.25]
        # self.densities = [750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1250]
        # self.frictions = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]
        # self.powers = [0.6, 0.9]
        # self.env_num = len(self.densities) * len(self.frictions)

    def run(
        self,
        *,
        # EPOpt specific - could go in __init__ but epsilon is callable
        paths,
        wst_env,
        eps
    ):
        """Instead of doing a trajectory of nsteps (ie, "horizon"), do a
        sample N "paths" and then return the bottom epsilon-percentile
        """
        # multienvs = self.env.num_envs > 1

        # Store all N trajectories sampled then return data of bottom-epsilon
        # lists -> lists of lists
        # paths = self.env_num
        (
            n_mb_obs,
            n_mb_rewards,
            n_mb_actions,
            n_mb_values,
            n_mb_dones,
            n_mb_neglogpacs,
            n_mb_envparam,
            n_mb_bnobn,
        ) = (
            [[] for _ in range(paths)],
            [[] for _ in range(paths)],
            [[] for _ in range(paths)],
            [[] for _ in range(paths)],
            [[] for _ in range(paths)],
            [[] for _ in range(paths)],
            [[] for _ in range(paths)],
            [[] for _ in range(paths)],
        )
        n_epinfos = [[] for _ in range(paths)]
        mb_states = self.states
        num_episodes = 0
        self.dones = [True]

        for N in range(paths):
            (
                mb_obs,
                mb_rewards,
                mb_actions,
                mb_values,
                mb_dones,
                mb_neglogpacs,
                epinfos,
                mb_enparam,
            ) = (
                n_mb_obs[N],
                n_mb_rewards[N],
                n_mb_actions[N],
                n_mb_values[N],
                n_mb_dones[N],
                n_mb_neglogpacs[N],
                n_epinfos[N],
                n_mb_envparam[N],
            )
            mb_enparam.append(self.env.envs[0].env.env.env.density)
            mb_enparam.append(self.env.envs[0].env.env.env.friction)

            for _ in range(
                self.env.venv.envs[0].spec.max_episode_steps
            ):  # rollout till done

                actions, values, self.states, neglogpacs = self.model.step(
                    self.obs, self.states, self.dones
                )
                mb_obs.append(self.obs.copy())
                mb_actions.append(actions)
                mb_values.append(values)
                mb_neglogpacs.append(neglogpacs)
                mb_dones.append(self.dones)
                self.obs[:], rewards, self.dones, infos, self.obsbn[:] = self.env.step(
                    actions
                )

                # self.obs[:], rewards, self.dones, infos = self.env.step(actions)
                for info in infos:
                    maybeepinfo = info.get("episode")
                    if maybeepinfo:
                        epinfos.append(maybeepinfo)
                mb_rewards.append(rewards)
                # Stop once single thread has finished an episode
                if self.dones:  # ie [True]
                    break

        # Compute the worst epsilon paths and concatenate them
        episode_returns = np.array([sum(r) for r in n_mb_rewards]).squeeze()

        print("----------episode return-------")
        print(episode_returns)
        # cutoff = np.percentile(episode_returns, 0)   # return the worst
        cutoff = np.min(episode_returns)
        minidx_ = np.argwhere(episode_returns == cutoff).squeeze()
        print("worst env:{}".format(minidx_))
        if isinstance(minidx_, list):
            print("list")
            minidx = minidx_[0]
        else:
            minidx = minidx_
        wst_envparam = n_mb_envparam[minidx]  # get worst env parameter

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        epinfos = []
        envparam_normalizatio = np.zeros(shape=np.array(wst_envparam).shape)
        # require the bounds
        envparam_normalizatio[0] = (
            self.env.envs[0].env.env.env.RANDOM_UPPER_DENSITY
            - self.env.envs[0].env.env.env.RANDOM_LOWER_DENSITY
        ) * 2
        envparam_normalizatio[1] = (
            self.env.envs[0].env.env.env.RANDOM_UPPER_FRICTION
            - self.env.envs[0].env.env.env.RANDOM_LOWER_FRICTION
        )  # *2can be tuned

        # compute obj list
        return_cutoff = np.percentile(episode_returns, 10)  # worst 10%
        epinfos_percen10 = []
        epinfos_all = []
        episode_returns_percen10 = []

        wst_percen_params_list = []
        wst_percen_idx_list = []

        # 1. first add worst 10% envs
        for N in range(paths):
            # record average performance
            epinfos_all.extend(n_epinfos[N])

            # record 10% worst performance
            if episode_returns[N] <= return_cutoff:
                epinfos_percen10.extend(n_epinfos[N])
                episode_returns_percen10.extend([episode_returns[N]])
                wst_percen_params_list.append(n_mb_envparam[N])
                wst_percen_idx_list.extend([N])

                num_episodes += 1
                # "cache" values to keep track of final ones
                next_obs = n_mb_obs[N]
                next_rewards = n_mb_rewards[N]
                next_actions = n_mb_actions[N]
                next_values = n_mb_values[N]
                next_dones = n_mb_dones[N]
                next_neglogpacs = n_mb_neglogpacs[N]
                next_epinfos = n_epinfos[N]
                # concatenate
                mb_obs.extend(next_obs)
                mb_rewards.extend(next_rewards)
                mb_actions.extend(next_actions)
                mb_values.extend(next_values)
                mb_dones.extend(next_dones)
                mb_neglogpacs.extend(next_neglogpacs)
                epinfos.extend(next_epinfos)
                print("N:{}   return:{}".format(N, next_epinfos[0]["r"]))

        # 2. then add other envs
        wst_percen_params_arr = np.array(wst_percen_params_list)
        for N in range(paths):

            next_rewards = n_mb_rewards[N]

            envparam_comp_arr = (
                wst_percen_params_arr - np.array(n_mb_envparam[N])
            ) / np.array(envparam_normalizatio)
            # print(envparam_comp_arr)
            envparam_comp = np.sum(
                envparam_comp_arr, axis=1
            ).__abs__()  # l1 norm distance
            # print(envparam_comp)
            objs = episode_returns[N] - eps * envparam_comp
            # NOTE: R[i] - eps * max (l1 distances) > R_worst
            if all(objs > episode_returns_percen10) and N not in wst_percen_idx_list:

                num_episodes += 1
                # "cache" values to keep track of final ones
                next_obs = n_mb_obs[N]
                next_rewards = n_mb_rewards[N]
                next_actions = n_mb_actions[N]
                next_values = n_mb_values[N]
                next_dones = n_mb_dones[N]
                next_neglogpacs = n_mb_neglogpacs[N]
                next_epinfos = n_epinfos[N]
                # concatenate
                mb_obs.extend(next_obs)
                mb_rewards.extend(next_rewards)
                mb_actions.extend(next_actions)
                mb_values.extend(next_values)
                mb_dones.extend(next_dones)
                mb_neglogpacs.extend(next_neglogpacs)
                epinfos.extend(next_epinfos)
                # print(next_epinfos)
                # print('N:{}   return:{}'.format(N, next_epinfos[0]['r']))

        total_steps = len(mb_rewards)
        epremean_percentile10 = safemean([epinfo["r"] for epinfo in epinfos_percen10])
        eprewmean_all = safemean([epinfo["r"] for epinfo in epinfos_all])

        # total and avg env steps
        eplen_all = sum([epinfo["l"] for epinfo in epinfos_all])
        eplen = sum([epinfo["l"] for epinfo in epinfos])
        eplenmean_all = safemean([epinfo["l"] for epinfo in epinfos_all])
        eplenmean_percentile10 = safemean([epinfo["l"] for epinfo in epinfos_percen10])

        print("{} envs choosed".format(num_episodes))
        print("eprewmean_all:{}".format(eprewmean_all))
        print("epremean_percentile10:{}".format(epremean_percentile10))

        #  batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        # We can't just use self.obs etc, because the last of the N paths
        # may not be included in the update
        last_values = self.model.value(
            self.obs, self.states, self.dones
        )  # value function

        #  discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0

        # Instead using nsteps, use the total number of steps in all kept trajectories
        # for t in reversed(range(self.nsteps)):
        for t in reversed(range(total_steps)):
            # if t == self.nsteps - 1:
            if t == total_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = (
                mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            )  # eq.12
            mb_advs[t] = lastgaelam = (
                delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            )

        mb_returns = mb_advs + mb_values

        return (
            *map(
                sf01,
                (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs),
            ),
            mb_states,
            epinfos,
            num_episodes,
            eprewmean_all,
            epremean_percentile10,
            np.array(episode_returns).squeeze(),
            eplen_all,
            eplen,
            eplenmean_all,
            eplenmean_percentile10,
        )


def learn(
    *,
    policy,
    env,
    env_id,
    nsteps,
    total_episodes,
    ent_coef,
    lr,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gamma=0.99,
    lam=0.95,
    log_interval=10,
    nminibatches=4,
    noptepochs=4,
    cliprange=0.2,
    save_interval=0,
    keep_all_ckpt=False,
    paths=100,
    eps_start=1.0,
    eps_end=40,
    eps_raise=1.005  # EPOpt specific 1.005
):
    """Only difference here is that epsilon and N are specified and passed to
    runner.run()
    """
    # FIXME:
    # Callable lr and cliprange don't work (at the moment) with the
    # total_env_steps terminating condition
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        raise NotImplementedError
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        raise NotImplementedError

    nenvs = 1
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    max_episode_steps = env.venv.envs[0].spec.max_episode_steps
    total_env_steps = int(total_episodes * max_episode_steps)
    logger.log("train total env steps", total_env_steps)

    eps = eps_start

    make_model = lambda: MRPOModel(
        policy=policy,
        ob_space=ob_space,
        ac_space=ac_space,
        nbatch_act=nenvs,
        nbatch_train=nbatch_train,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
    )

    model = make_model()
    runner = MRPORunner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    tfirststart = time.time()

    update = 0
    env_steps_all = 0
    env_steps_used = 0
    num_rollouts = 0
    num_rollouts_used = 0

    def make_env():
        wst_env = base.make_env(env_id, outdir=logger.get_dir())
        wst_env.seed(6)  # worst env, but same as env
        return wst_env

    wst_env = DummyVecEnv([make_env])
    wst_env = VecNormalize(wst_env)

    while True:
        update += 1
        if env_steps_all >= total_env_steps:
            break

        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches

        tstart = time.time()
        frac = 1.0 - (env_steps_all / total_env_steps)  # [0, 1]
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        (
            obs,
            returns,
            masks,
            actions,
            values,
            neglogpacs,
            states,
            epinfos,
            num_episodes,
            eprewmean_all,
            epremean_percentile10,
            episodes_returns,
            eplen_all,
            eplen,
            eplenmean_all,
            eplenmean_percentile10,
        ) = runner.run(
            paths=paths, wst_env=wst_env, eps=eps
        )  # pylint: disable=E0632

        eps = eps_raise * eps
        # eps = min(eps_end, eps_raise * eps)

        assert num_episodes == np.sum(masks), (num_episodes, np.sum(masks))
        env_steps_all += eplen_all
        env_steps_used += eplen
        num_rollouts += paths
        num_rollouts_used += num_episodes

        mblossvals = []
        if states is None:  # nonrecurrent version
            for _ in range(noptepochs):
                mblossvals.append(
                    model.train(
                        lrnow,
                        cliprangenow,
                        *(obs, returns, masks, actions, values, neglogpacs),
                    )
                )
        else:  # recurrent version
            raise NotImplementedError("Use examples.epopt_lstm")

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))

        # log
        if update % log_interval == 0:
            ev = explained_variance(values, returns)
            logger.record_step(env_steps_all)
            logger.record_tabular("z/env_steps", env_steps_all)
            logger.record_tabular("z/env_steps_used", env_steps_used)
            logger.record_tabular("z/rollouts", num_rollouts)
            logger.record_tabular("z/rollouts_used", num_rollouts_used)
            logger.record_tabular(
                "z/iterations", update
            )  # used in original paper x axis
            logger.record_tabular("z/fps", fps)
            logger.record_tabular("z/time_cost", tnow - tfirststart)

            logger.record_tabular("MRPO/envs_choosed_ratio", num_episodes / paths)
            logger.record_tabular("MRPO/eps", eps)
            logger.record_tabular("MRPO/explained_variance", float(ev))

            logger.record_tabular("metrics/return_eval_avg", eprewmean_all)
            logger.record_tabular("metrics/return_eval_worst", epremean_percentile10)
            logger.record_tabular("metrics/total_steps_eval_avg", eplenmean_all)
            logger.record_tabular(
                "metrics/total_steps_eval_worst", eplenmean_percentile10
            )

            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.record_tabular("MRPO/" + lossname, lossval)
            logger.dump_tabular()

            # save model
            if save_interval and update % save_interval == 0:
                checkdir = osp.join(logger.get_dir(), "checkpoints")
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, "%.5i" % update)
                print("Saving to", savepath)
                obs_norms = {}
                obs_norms["clipob"] = env.clipob
                obs_norms["mean"] = env.ob_rms.mean
                obs_norms["var"] = env.ob_rms.var + env.epsilon
                with open(osp.join(checkdir, "%.5i" % update + "normalize"), "wb") as f:
                    pickle.dump(obs_norms, f, pickle.HIGHEST_PROTOCOL)
                model.save(savepath)

    env.close()


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
