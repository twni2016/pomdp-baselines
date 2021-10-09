"""Based on a2c_episodes/base.py, but modified to handle varying obs length for EPOpt"""

import os
import gym

import numpy as np
import tensorflow as tf
from ..baselines.common.distributions import make_pdtype
from ..baselines.a2c.utils import fc, seq_to_batch, ortho_init


def make_env(env_id, process_idx=0, outdir=None):
    import sunblaze_envs

    env = sunblaze_envs.make(env_id)
    if outdir:
        env = sunblaze_envs.MonitorParameters(
            env,
            output_filename=os.path.join(
                outdir, "env-parameters-{}.json".format(process_idx)
            ),
        )

    return env


class mlp_policy(object):
    """Creates policy and value MLP networks, with no parameter sharing.
    They are 2-hidden-layer with 64 hidden units in each layer and tanh activations.
    Both discrete and continuous action spaces are supported.
    Environments with a discrete action space have a softmax policy, while environments with
    a continuous action space have a Gaussian with diagonal covariance."""

    def __init__(self, sess, ob_space, ac_space, nenvs, nsteps, reuse=False):

        if nsteps is None:
            ob_shape = (None,) + ob_space.shape
        else:
            ob_shape = (nenvs * nsteps,) + ob_space.shape

        if len(ac_space.shape) == 0:
            nact = ac_space.n
            discrete = True
        else:
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
