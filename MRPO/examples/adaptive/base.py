import os
import gym

import numpy as np
import tensorflow as tf
from baselines.common.distributions import make_pdtype
from baselines.a2c.utils import fc, batch_to_seq, seq_to_batch, lstm, ortho_init


def make_env(env_id, process_idx=0, outdir=None):
    import sunblaze_envs
    from .sunblaze_monitor import MonitorParameters

    env = sunblaze_envs.make(env_id)
    if outdir:
        env = MonitorParameters(
            env,
            output_filename=os.path.join(
                outdir, "env-parameters-{}.json".format(process_idx)
            ),
        )

    return env


def gru(xs, ms, s, scope, nh, init_scale=1.0, activ="tanh"):
    """Implements a gated recurrent unit"""
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    nsteps = len(xs)

    with tf.variable_scope(scope):
        wx1 = tf.get_variable("wx1", [nin, nh * 2], initializer=ortho_init(init_scale))
        wh1 = tf.get_variable("wh1", [nh, nh * 2], initializer=ortho_init(init_scale))
        b1 = tf.get_variable("b1", [nh * 2], initializer=tf.constant_initializer(0.0))
        wx2 = tf.get_variable("wx2", [nin, nh], initializer=ortho_init(init_scale))
        wh2 = tf.get_variable("wh2", [nh, nh], initializer=ortho_init(init_scale))
        b2 = tf.get_variable("b2", [nh], initializer=tf.constant_initializer(0.0))

    for idx, (x, m) in enumerate(zip(xs, ms)):
        s = s * (1 - m)  # resets hidden state of RNN
        y = tf.matmul(x, wx1) + tf.matmul(s, wh1) + b1
        z, r = tf.split(axis=1, num_or_size_splits=2, value=y)
        z = tf.nn.sigmoid(z)
        r = tf.nn.sigmoid(r)
        h = tf.matmul(x, wx2) + tf.matmul(s * r, wh2) + b2
        if activ == "tanh":
            h = tf.tanh(h)
        elif activ == "relu":
            h = tf.nn.relu(h)
        else:
            raise ValueError(activ)
        s = (1 - z) * h + z * s
        xs[idx] = s
    return xs, s


class lstm_policy(object):
    """Creates policy and value LSTM networks, with parameter sharing.
    In addition to the observation, the networks also take the action, reward, and done as inputs.
    There is one hidden layer with nlstm units, default 256.
    Environments with a discrete action space have a softmax policy, while environments with
    a continuous action space have Gaussian with diagonal covariance."""

    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        nbatch,
        nsteps,
        nlstm=256,
        reuse=False,
        feature_mlp=True,
    ):

        nenv = nbatch // nsteps
        # assume that inputs are vectors and reward is a scalar
        if len(ac_space.shape) == 0:
            # discrete set of actions, input as one-hot encodings
            nact = ac_space.n
            discrete = True
            input_length = ob_space.shape[0] + nact + 2
        else:
            actdim = ac_space.shape[0]
            discrete = False
            input_length = ob_space.shape[0] + actdim + 2
        input_shape = (nbatch, input_length)

        X = tf.placeholder(tf.float32, input_shape, name="Input")
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done with a trial at time t-1)
        S = tf.placeholder(
            tf.float32, [nenv, nlstm * 2]
        )  # states of the recurrent policy
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            if feature_mlp:
                print("Using feature network in front of LSTM")
                h1 = activ(fc(X, "fc1", nh=nlstm, init_scale=np.sqrt(2)))
                h2 = activ(fc(h1, "fc2", nh=nlstm, init_scale=np.sqrt(2)))
                xs = batch_to_seq(h2, nenv, nsteps)
            else:
                print("No feature network in front of LSTM")
                xs = batch_to_seq(X, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, "lstm1", nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, "vf", 1)
            if discrete:
                pi = fc(h5, "pi", nact, init_scale=0.01)
            else:
                pi = fc(h5, "pi", actdim, init_scale=0.01)
                logstd = tf.get_variable(
                    name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer()
                )

        self.pdtype = make_pdtype(ac_space)
        if discrete:
            self.pd = self.pdtype.pdfromflat(pi)
        else:
            pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
            self.pd = self.pdtype.pdfromflat(pdparam)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, ac, rew, done, mask):
            # if discrete action space, convert ac to one-hot encoding and done to int
            rew = np.reshape(np.asarray([rew]), (nbatch, 1))
            done = np.reshape(np.asarray([done], dtype=float), (nbatch, 1))
            if discrete:
                if ac[0] == -1:
                    ac = np.zeros((nbatch, nact), dtype=np.int)
                else:
                    ac = np.reshape(np.asarray([ac]), (nbatch,))
                    ac = np.eye(nact)[ac]
                x = np.concatenate((ob, ac, rew, done), axis=1)
            else:
                ac = np.reshape(np.asarray([ac]), (nbatch, actdim))
                x = np.concatenate((ob, ac, rew, done), axis=1)
            return sess.run([a0, v0, snew, neglogp0], {X: x, S: state, M: mask})

        def value(ob, state, ac, rew, done, mask):
            rew = np.reshape(np.asarray([rew]), (nbatch, 1))
            done = np.reshape(np.asarray([done], dtype=float), (nbatch, 1))
            if discrete:
                if ac[0] == -1:
                    ac = np.zeros((nbatch, nact), dtype=np.int)
                else:
                    ac = np.reshape(np.asarray([ac]), (nbatch,))
                    ac = np.eye(nact)[ac]
                x = np.concatenate((ob, ac, rew, np.array(done, dtype=float)), axis=1)
            else:
                ac = np.reshape(np.asarray([ac]), (nbatch, actdim))
                x = np.concatenate((ob, ac, rew, np.array(done, dtype=float)), axis=1)
            return sess.run(v0, {X: x, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class gru_policy(object):
    """Creates policy and value GRU networks, with parameter sharing.
    In addition to the observation, the networks also take action, reward, and done as inputs.
    There is one hidden layer with nlstm units, default 256.
    Environments with a discrete action space have a softmax policy, while environments with
    a continuous action space have Gaussian with diagonal covariance."""

    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        nbatch,
        nsteps,
        feat_activ="tanh",
        gru_activ="tanh",
        ngru=256,
        reuse=False,
    ):

        nenv = nbatch // nsteps
        if len(ac_space.shape) == 0:
            nact = ac_space.n
            discrete = True
            input_length = ob_space.shape[0] + nact + 2
        else:
            actdim = ac_space.shape[0]
            discrete = False
            input_length = ob_space.shape[0] + actdim + 2
        input_shape = (nbatch, input_length)

        X = tf.placeholder(tf.float32, input_shape, name="Input")
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done with a trial at time t-1)
        S = tf.placeholder(tf.float32, [nenv, ngru])  # states of the recurrent policy
        with tf.variable_scope("model", reuse=reuse):
            if feat_activ == "tanh":
                activ = tf.tanh
            elif feat_activ == "relu":
                activ = tf.nn.relu
            else:
                raise ValueError(feat_activ)
            h1 = activ(fc(X, "fc1", nh=ngru, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, "fc2", nh=ngru, init_scale=np.sqrt(2)))
            xs = batch_to_seq(h2, nenv, nsteps)

            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = gru(xs, ms, S, "gru1", nh=ngru, activ=gru_activ)
            h5 = seq_to_batch(h5)
            vf = fc(h5, "vf", 1)
            if discrete:
                pi = fc(h5, "pi", nact, init_scale=0.01)
            else:
                pi = fc(h5, "pi", actdim, init_scale=0.01)
                logstd = tf.get_variable(
                    name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer()
                )

        self.pdtype = make_pdtype(ac_space)
        if discrete:
            self.pd = self.pdtype.pdfromflat(pi)
        else:
            pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
            self.pd = self.pdtype.pdfromflat(pdparam)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, ngru), dtype=np.float32)

        def step(ob, state, ac, rew, done, mask):
            rew = np.reshape(np.asarray([rew]), (nbatch, 1))
            done = np.reshape(np.asarray([done], dtype=float), (nbatch, 1))
            if discrete:
                if ac[0] == -1:
                    ac = np.zeros((nbatch, nact), dtype=np.int)
                else:
                    ac = np.reshape(np.asarray([ac]), (nbatch,))
                    ac = np.eye(nact)[ac]
                x = np.concatenate((ob, ac, rew, done), axis=1)
            else:
                ac = np.reshape(np.asarray([ac]), (nbatch, actdim))
                x = np.concatenate((ob, ac, rew, done), axis=1)
            return sess.run([a0, v0, snew, neglogp0], {X: x, S: state, M: mask})

        def value(ob, state, ac, rew, done, mask):
            rew = np.reshape(np.asarray([rew]), (nbatch, 1))
            done = np.reshape(np.asarray([done], dtype=float), (nbatch, 1))
            if discrete:
                if ac[0] == -1:
                    ac = np.zeros((nbatch, nact), dtype=np.int)
                else:
                    ac = np.reshape(np.asarray([ac]), (nbatch,))
                    ac = np.eye(nact)[ac]
                x = np.concatenate((ob, ac, rew, done), axis=1)
            else:
                ac = np.reshape(np.asarray([ac]), (nbatch, actdim))
                x = np.concatenate((ob, ac, rew, done), axis=1)
            return sess.run(v0, {X: x, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class mlp_policy(object):
    """Creates policy and value MLP networks, with no parameter sharing.
    They are 2-hidden-layer with 64 hidden units in each layer and tanh activations.
    Both discrete and continuous action spaces are supported.
    Environments with a discrete action space have a softmax policy, while environments with
    a continuous action space have a Gaussian with diagonal covariance."""

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):

        ob_shape = (nbatch,) + ob_space.shape
        if len(ac_space.shape) == 0:
            # discrete set of actions
            nact = ac_space.n
            discrete = True
        else:
            # continuous box of actions
            actdim = ac_space.shape[0]
            discrete = False
        X = tf.placeholder(tf.float32, ob_shape, name="Ob")

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, "vf_fc1", nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, "vf_fc2", nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, "vf", 1)[:, 0]
            h1 = activ(fc(X, "pi_fc1", nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, "pi_fc2", nh=64, init_scale=np.sqrt(2)))
            if discrete:
                pi = fc(h2, "pi", nact, init_scale=0.01)
            else:
                pi = fc(h2, "pi", actdim, init_scale=0.01)
                logstd = tf.get_variable(
                    name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer()
                )

        self.pdtype = make_pdtype(ac_space)
        if discrete:
            self.pd = self.pdtype.pdfromflat(pi)
        else:
            pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
            self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
