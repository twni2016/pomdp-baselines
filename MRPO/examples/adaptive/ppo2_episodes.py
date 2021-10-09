import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
import pickle


class Model(object):
    def __init__(
        self,
        policy,
        ob_space,
        ac_space,
        nbatch_act,
        nbatch_train,
        nsteps,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        akl_coef=0,
    ):

        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        A = train_model.pdtype.sample_placeholder([nbatch_train])
        ADV = tf.placeholder(tf.float32, [nbatch_train])
        R = tf.placeholder(tf.float32, [nbatch_train])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [nbatch_train])
        OLDVPRED = tf.placeholder(tf.float32, [nbatch_train])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(
            train_model.vf - OLDVPRED, -CLIPRANGE, CLIPRANGE
        )
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = 0.5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(
            tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE))
        )
        if not akl_coef:
            # add approxkl as a penalty to try and stabilize training
            loss = (
                pg_loss - entropy * ent_coef + vf_loss * vf_coef + approxkl * akl_coef
            )
        else:
            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope("model"):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(
            lr,
            cliprange,
            obs,
            prev_rewards,
            returns,
            dones,
            masks,
            prev_actions,
            actions,
            values,
            neglogpacs,
            states,
        ):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            rews = np.reshape(prev_rewards, (nbatch_train, 1))
            ds = np.reshape(np.asarray(dones, dtype=np.float32), (nbatch_train, 1))
            if len(ac_space.shape) == 0:
                prev_actions = np.reshape(prev_actions, (nbatch_train,))
                one_hot = np.eye(ac_space.n)[prev_actions]
                for i in range(nbatch_train):
                    if prev_actions[i] == -1:
                        one_hot[i, :] = np.zeros((ac_space.n,), dtype=np.int)
                x = np.concatenate((obs, one_hot, rews, ds), axis=1)
                actions = np.reshape(actions, (nbatch_train,))
            else:
                prev_actions = np.reshape(
                    prev_actions, (nbatch_train, ac_space.shape[0])
                )
                x = np.concatenate((obs, prev_actions, rews, ds), axis=1)
            td_map = {
                train_model.X: x,
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

        self.train = train
        self.train_mdel = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class Runner(object):
    def __init__(self, env, model, nsteps, gamma, lam, episodes_per_trial=5):
        self.env = env
        self.model = model
        self.nenv = env.num_envs
        self.batch_ob_shape = (self.nenv * nsteps,) + env.observation_space.shape
        self.obs = np.zeros(
            (self.nenv,) + env.observation_space.shape, dtype=np.float32
        )
        self.obs[:] = env.reset([True for _ in range(self.nenv)])
        self.gamma = gamma
        self.lam = lam
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
            mb_neglogpacs,
        ) = ([], [], [], [], [], [], [], [], [])
        mb_states = self.states
        epinfos = []
        num_trials = 0
        for _ in range(self.nsteps):
            actions, values, states, neglogpacs = self.model.step(
                self.obs,
                self.states,
                self.actions,
                self.rewards,
                self.dones,
                self.masks,
            )
            mb_obs.append(self.obs.copy())
            prev_actions.append(self.actions)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            prev_rewards.append(self.rewards)
            mb_masks.append(self.masks)
            mb_dones.append(self.dones)
            # if end_of_trial, if this episode gets done in the next step, we need to reset environment parameters
            end_of_trial = [
                self.episode_in_trial[i] == (self.episodes_per_trial - 1)
                for i in range(self.nenv)
            ]
            obs, rewards, dones, infos = self.env.step(actions, end_of_trial)
            mb_rewards.append(rewards)
            self.actions = actions
            self.states = states
            self.obs[:] = obs
            self.dones = dones
            self.masks = [False for _ in range(self.nenv)]
            self.rewards = rewards
            for i, done in enumerate(self.dones):
                if done:
                    self.episode_in_trial[i] += 1
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
                        num_trials += 1
            for info in infos:
                maybeepinfo = info.get("episode")
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
        # format correctly
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        prev_rewards = np.asarray(prev_rewards, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        if self.discrete:
            prev_actions = np.asarray(prev_actions, dtype=np.int)
            mb_actions = np.asarray(mb_actions, dtype=np.int)
        else:
            prev_actions = np.asarray(prev_actions, dtype=np.float32)
            mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_masks = np.asarray(mb_masks, dtype=np.bool)
        last_values = self.model.value(
            self.obs, self.states, self.actions, self.rewards, self.dones, self.masks
        )
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                self.masks = np.asarray(self.masks, dtype=np.bool)
                nextnonterminal = 1.0 - self.masks
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_masks[t + 1]
                nextvalues = mb_values[t + 1]
            delta = (
                mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            )
            mb_advs[t] = lastgaelam = (
                delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            )
        mb_returns = mb_advs + mb_values
        mb_obs = np.swapaxes(mb_obs, 1, 0).reshape(self.batch_ob_shape)
        prev_rewards = np.swapaxes(prev_rewards, 1, 0).flatten()
        mb_returns = np.swapaxes(mb_returns, 1, 0).flatten()
        mb_dones = np.swapaxes(mb_dones, 1, 0).flatten()
        mb_masks = np.swapaxes(mb_masks, 1, 0).flatten()
        prev_actions = np.swapaxes(prev_actions, 1, 0).reshape(self.batch_ac_shape)
        mb_actions = np.swapaxes(mb_actions, 1, 0).reshape(self.batch_ac_shape)
        mb_values = np.swapaxes(mb_values, 1, 0).flatten()
        mb_neglogpacs = np.swapaxes(mb_neglogpacs, 1, 0).flatten()
        return (
            mb_obs,
            prev_rewards,
            mb_returns,
            mb_dones,
            mb_masks,
            prev_actions,
            mb_actions,
            mb_values,
            mb_neglogpacs,
            mb_states,
            epinfos,
            num_trials,
        )


def constfn(val):
    def f(_):
        return val

    return f


def learn(
    policy,
    env,
    nsteps,
    total_trials,
    episodes_per_trial,
    ent_coef,
    lr,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gamma=0.99,
    lam=0.95,
    log_interval=10,
    nminibatches=1,
    noptepochs=4,
    cliprange=0.2,
    save_interval=100,
    keep_all_ckpt=False,
    akl_coef=0,
):

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda: Model(
        policy=policy,
        ob_space=ob_space,
        ac_space=ac_space,
        nbatch_act=nenvs,
        nbatch_train=nbatch_train,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        akl_coef=akl_coef,
    )

    if save_interval and logger.get_dir():
        import cloudpickle

        with open(osp.join(logger.get_dir(), "make_model.pkl"), "wb") as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(
        env=env,
        model=model,
        nsteps=nsteps,
        gamma=gamma,
        lam=lam,
        episodes_per_trial=episodes_per_trial,
    )

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    update = 0
    trials_so_far = 0
    old_savepath = None
    while True:
        update += 1
        if trials_so_far >= total_trials:
            break

        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - float(trials_so_far) / total_trials
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        (
            obs,
            prev_rewards,
            returns,
            dones,
            masks,
            prev_actions,
            actions,
            values,
            neglogpacs,
            states,
            epinfos,
            num_trials,
        ) = runner.run()
        epinfobuf.extend(epinfos)
        trials_so_far += num_trials
        mblossvals = []

        assert nenvs % nminibatches == 0
        envsperbatch = nenvs // nminibatches
        envinds = np.arange(nenvs)
        flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
        envsperbatch = nbatch_train // nsteps
        for _ in range(noptepochs):
            np.random.shuffle(envinds)
            for start in range(0, nenvs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mbflatinds = flatinds[mbenvinds].ravel()
                slices = (
                    arr[mbflatinds]
                    for arr in (
                        obs,
                        prev_rewards,
                        returns,
                        dones,
                        masks,
                        prev_actions,
                        actions,
                        values,
                        neglogpacs,
                    )
                )
                mbstates = states[mbenvinds]
                mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update * nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nbatch)
            logger.logkv("total_trials", trials_so_far)
            logger.logkv("total_episodes", trials_so_far * episodes_per_trial)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv("eprewmean", safemean([epinfo["r"] for epinfo in epinfobuf]))
            logger.logkv("eplenmean", safemean([epinfo["l"] for epinfo in epinfobuf]))
            logger.logkv("time_elapsed", tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
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


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
