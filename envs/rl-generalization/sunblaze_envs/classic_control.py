import math

from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.classic_control.mountain_car import MountainCarEnv
from gym.envs.classic_control.acrobot import AcrobotEnv
from gym.envs.classic_control.pendulum import PendulumEnv
import numpy as np

from .base import EnvBinarySuccessMixin

"""
# from: https://github.com/openai/gym/blob/0c91364cd4a7ea70f242a28b85c3aea2d74aa35a/gym/envs/classic_control/pendulum.py#L89
def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
"""


def uniform_exclude_inner(np_uniform, a, b, a_i, b_i):
    """Draw sample from uniform distribution, excluding an inner range"""
    if not (a < a_i and b_i < b):
        raise ValueError(
            "Bad range, inner: ({},{}), outer: ({},{})".format(a, b, a_i, b_i)
        )
    while True:
        # Resample until value is in-range
        result = np_uniform(a, b)
        if (a <= result and result < a_i) or (b_i <= result and result < b):
            return result


# Cart pole environment variants.


class ModifiableCartPoleEnv(CartPoleEnv, EnvBinarySuccessMixin):

    RANDOM_LOWER_FORCE_MAG = 5.0
    RANDOM_UPPER_FORCE_MAG = 15.0
    EXTREME_LOWER_FORCE_MAG = 1.0
    EXTREME_UPPER_FORCE_MAG = 20.0

    RANDOM_LOWER_LENGTH = 0.25
    RANDOM_UPPER_LENGTH = 0.75
    EXTREME_LOWER_LENGTH = 0.05
    EXTREME_UPPER_LENGTH = 1.0

    RANDOM_LOWER_MASSPOLE = 0.05
    RANDOM_UPPER_MASSPOLE = 0.5
    EXTREME_LOWER_MASSPOLE = 0.01
    EXTREME_UPPER_MASSPOLE = 1.0

    def _followup(self):
        """Cascade values of new (variable) parameters"""
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

    def reset(self, new=True):
        """new is a boolean variable telling whether to regenerate the environment parameters"""
        """Default is to just ignore it"""
        self.nsteps = 0
        return super(ModifiableCartPoleEnv, self).reset()

    @property
    def parameters(self):
        return {
            "id": self.spec.id,
        }

    def step(self, *args, **kwargs):
        """Wrapper to increment new variable nsteps"""
        self.nsteps += 1
        return super().step(*args, **kwargs)

    def is_success(self):
        """Returns True is current state indicates success, False otherwise
        Balance for at least 195 time steps ("definition" of success in Gym:
        https://github.com/openai/gym/wiki/CartPole-v0#solved-requirements)
        """
        target = 195
        if self.nsteps >= target:
            # print("[SUCCESS]: nsteps is {}, reached target {}".format(
            #      self.nsteps, target))
            return True
        else:
            # print("[NO SUCCESS]: nsteps is {}, target {}".format(
            #      self.nsteps, target))
            return False


class StrongPushCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(StrongPushCartPole, self).__init__()
        self.force_mag = self.EXTREME_UPPER_FORCE_MAG

    @property
    def parameters(self):
        parameters = super(StrongPushCartPole, self).parameters
        parameters.update(
            {
                "force": self.force_mag,
            }
        )
        return parameters


class WeakPushCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(WeakPushCartPole, self).__init__()
        self.force_mag = self.EXTREME_LOWER_FORCE_MAG

    @property
    def parameters(self):
        parameters = super(WeakPushCartPole, self).parameters
        parameters.update(
            {
                "force": self.force_mag,
            }
        )
        return parameters


class RandomStrongPushCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomStrongPushCartPole, self).__init__()
        self.force_mag = self.np_random.uniform(
            self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG
        )

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.force_mag = self.np_random.uniform(
                self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG
            )
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomStrongPushCartPole, self).parameters
        parameters.update(
            {
                "force": self.force_mag,
            }
        )
        return parameters


class RandomWeakPushCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomWeakPushCartPole, self).__init__()
        self.force_mag = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_FORCE_MAG,
            self.EXTREME_UPPER_FORCE_MAG,
            self.RANDOM_LOWER_FORCE_MAG,
            self.RANDOM_UPPER_FORCE_MAG,
        )

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.force_mag = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_FORCE_MAG,
                self.EXTREME_UPPER_FORCE_MAG,
                self.RANDOM_LOWER_FORCE_MAG,
                self.RANDOM_UPPER_FORCE_MAG,
            )
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomWeakPushCartPole, self).parameters
        parameters.update(
            {
                "force": self.force_mag,
            }
        )
        return parameters


class ShortPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(ShortPoleCartPole, self).__init__()
        self.length = self.EXTREME_LOWER_LENGTH
        self._followup()

    @property
    def parameters(self):
        parameters = super(ShortPoleCartPole, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class LongPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(LongPoleCartPole, self).__init__()
        self.length = self.EXTREME_UPPER_LENGTH
        self._followup()

    @property
    def parameters(self):
        parameters = super(LongPoleCartPole, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomLongPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomLongPoleCartPole, self).__init__()
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
        )
        self._followup()

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.length = self.np_random.uniform(
                self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
            )
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomLongPoleCartPole, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomShortPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomShortPoleCartPole, self).__init__()
        self.length = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_LENGTH,
            self.EXTREME_UPPER_LENGTH,
            self.RANDOM_LOWER_LENGTH,
            self.RANDOM_UPPER_LENGTH,
        )
        self._followup()

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.length = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_LENGTH,
                self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH,
                self.RANDOM_UPPER_LENGTH,
            )
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomShortPoleCartPole, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class LightPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(LightPoleCartPole, self).__init__()
        self.masspole = self.EXTREME_LOWER_MASSPOLE
        self._followup()

    @property
    def parameters(self):
        parameters = super(LightPoleCartPole, self).parameters
        parameters.update(
            {
                "mass": self.masspole,
            }
        )
        return parameters


class HeavyPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(HeavyPoleCartPole, self).__init__()
        self.masspole = self.EXTREME_UPPER_MASSPOLE
        self._followup()

    @property
    def parameters(self):
        parameters = super(HeavyPoleCartPole, self).parameters
        parameters.update(
            {
                "mass": self.masspole,
            }
        )
        return parameters


class RandomHeavyPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomHeavyPoleCartPole, self).__init__()
        self.masspole = self.np_random.uniform(
            self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE
        )
        self._followup()

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.masspole = self.np_random.uniform(
                self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE
            )
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomHeavyPoleCartPole, self).parameters
        parameters.update(
            {
                "mass": self.masspole,
            }
        )
        return parameters


class RandomLightPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomLightPoleCartPole, self).__init__()
        self.masspole = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_MASSPOLE,
            self.EXTREME_UPPER_MASSPOLE,
            self.RANDOM_LOWER_MASSPOLE,
            self.RANDOM_UPPER_MASSPOLE,
        )
        self._followup()

    def reset(self, new=True):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.masspole = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASSPOLE,
                self.EXTREME_UPPER_MASSPOLE,
                self.RANDOM_LOWER_MASSPOLE,
                self.RANDOM_UPPER_MASSPOLE,
            )
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomLightPoleCartPole, self).parameters
        parameters.update(
            {
                "mass": self.masspole,
            }
        )
        return parameters


class RandomNormalCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomNormalCartPole, self).__init__()
        self.force_mag = self.np_random.uniform(
            self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG
        )
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
        )
        self.masspole = self.np_random.uniform(
            self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE
        )
        self._followup()

    def reset(self, new=True):
        self.nsteps = 0  # for super.is_success()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.force_mag = self.np_random.uniform(
                self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG
            )
            self.length = self.np_random.uniform(
                self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
            )
            self.masspole = self.np_random.uniform(
                self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE
            )
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomNormalCartPole, self).parameters
        # parameters.update({'force_mag': self.force_mag, 'length': self.length, 'masspole': self.masspole, })
        parameters.update(
            {
                "force_mag": self.force_mag,
                "length": self.length,
                "masspole": self.masspole,
                "total_mass": self.total_mass,
                "polemass_length": self.polemass_length,
            }
        )
        return parameters


class RandomExtremeCartPole(ModifiableCartPoleEnv):
    def __init__(self):
        super(RandomExtremeCartPole, self).__init__()
        """
        self.force_mag = self.np_random.uniform(self.LOWER_FORCE_MAG, self.UPPER_FORCE_MAG)
        self.length = self.np_random.uniform(self.LOWER_LENGTH, self.UPPER_LENGTH)
        self.masspole = self.np_random.uniform(self.LOWER_MASSPOLE, self.UPPER_MASSPOLE)
        """
        self.force_mag = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_FORCE_MAG,
            self.EXTREME_UPPER_FORCE_MAG,
            self.RANDOM_LOWER_FORCE_MAG,
            self.RANDOM_UPPER_FORCE_MAG,
        )
        self.length = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_LENGTH,
            self.EXTREME_UPPER_LENGTH,
            self.RANDOM_LOWER_LENGTH,
            self.RANDOM_UPPER_LENGTH,
        )
        self.masspole = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_MASSPOLE,
            self.EXTREME_UPPER_MASSPOLE,
            self.RANDOM_LOWER_MASSPOLE,
            self.RANDOM_UPPER_MASSPOLE,
        )

        self._followup()
        # NOTE(cpacker): even though we're just changing the above params,
        # we still need to regen the other var dependencies
        # We need to scan through the other methods to make sure the same
        # mistake isn't being made

        # self.gravity = 9.8
        # self.masscart = 1.0
        # self.masspole = 0.1
        # self.total_mass = (self.masspole + self.masscart)
        # self.length = 0.5 # actually half the pole's length
        # self.polemass_length = (self.masspole * self.length)
        # self.force_mag = 10.0
        # self.tau = 0.02  # seconds between state updates

    def reset(self, new=True):
        self.nsteps = 0  # for super.is_success()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        """
        self.force_mag = self.np_random.uniform(self.LOWER_FORCE_MAG, self.UPPER_FORCE_MAG)
        self.length = self.np_random.uniform(self.LOWER_LENGTH, self.UPPER_LENGTH)
        self.masspole = self.np_random.uniform(self.LOWER_MASSPOLE, self.UPPER_MASSPOLE)
        """
        if new:
            self.force_mag = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_FORCE_MAG,
                self.EXTREME_UPPER_FORCE_MAG,
                self.RANDOM_LOWER_FORCE_MAG,
                self.RANDOM_UPPER_FORCE_MAG,
            )
            self.length = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_LENGTH,
                self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH,
                self.RANDOM_UPPER_LENGTH,
            )
            self.masspole = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASSPOLE,
                self.EXTREME_UPPER_MASSPOLE,
                self.RANDOM_LOWER_MASSPOLE,
                self.RANDOM_UPPER_MASSPOLE,
            )
            self._followup()
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomExtremeCartPole, self).parameters
        parameters.update(
            {
                "force_mag": self.force_mag,
                "length": self.length,
                "masspole": self.masspole,
                "total_mass": self.total_mass,
                "polemass_length": self.polemass_length,
            }
        )
        return parameters


# Mountain car environment variants.


class ModifiableMountainCarEnv(MountainCarEnv):
    """A variant of mountain car without hardcoded force/mass."""

    RANDOM_LOWER_FORCE = 0.0005
    RANDOM_UPPER_FORCE = 0.005
    EXTREME_LOWER_FORCE = 0.0001
    EXTREME_UPPER_FORCE = 0.01

    RANDOM_LOWER_MASS = 0.001
    RANDOM_UPPER_MASS = 0.005
    EXTREME_LOWER_MASS = 0.0005
    EXTREME_UPPER_MASS = 0.01

    def __init__(self):
        super(ModifiableMountainCarEnv, self).__init__()

        self.force = 0.001
        self.mass = 0.0025

    def step(self, action):
        """Rewritten to remove hard-coding of values in original code"""
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.mass)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        done = bool(position >= self.goal_position)
        reward = -1.0

        # New additions to support is_success()
        self.nsteps += 1
        target = 110
        if self.nsteps <= target and done:
            # print("[SUCCESS]: nsteps is {}, done before target {}".format(
            #      self.nsteps, target))
            self.success = True
        else:
            # print("[NO SUCCESS]: nsteps is {}, not done before target {}".format(
            #      self.nsteps, target))
            self.success = False
        ###

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self, new=True):
        self.nsteps = 0
        return super(ModifiableMountainCarEnv, self).reset()

    @property
    def parameters(self):
        return {
            "id": self.spec.id,
        }

    def is_success(self):
        """Returns True is current state indicates success, False otherwise
        get to the top of the hill within 110 time steps (definition of success in Gym)

        MountainCar sets done=True once the car reaches the "top of the hill",
        so we can just check if done=True and nsteps<=110. See:
        https://github.com/openai/gym/blob/0ccb08dfa1535624b45645e141af9398e2eba416/gym/envs/classic_control/mountain_car.py#L49
        """
        # NOTE: Moved logic to step()
        return self.success


class WeakForceMountainCar(ModifiableMountainCarEnv):
    def __init__(self):
        super(WeakForceMountainCar, self).__init__()
        self.force = self.EXTREME_LOWER_FORCE

    @property
    def parameters(self):
        parameters = super(WeakForceMountainCar, self).parameters
        parameters.update(
            {
                "force": self.force,
            }
        )
        return parameters


class StrongForceMountainCar(ModifiableMountainCarEnv):
    def __init__(self):
        super(StrongForceMountainCar, self).__init__()
        self.force = self.EXTREME_UPPER_FORCE

    @property
    def parameters(self):
        parameters = super(StrongForceMountainCar, self).parameters
        parameters.update(
            {
                "force": self.force,
            }
        )
        return parameters


class RandomStrongForceMountainCar(ModifiableMountainCarEnv):
    def reset(self, new=True):
        if new:
            self.force = self.np_random.uniform(
                self.RANDOM_LOWER_FORCE, self.RANDOM_UPPER_FORCE
            )
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomStrongForceMountainCar, self).parameters
        parameters.update(
            {
                "force": self.force,
            }
        )
        return parameters


class RandomWeakForceMountainCar(ModifiableMountainCarEnv):
    def reset(self, new=True):
        if new:
            self.force = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_FORCE,
                self.EXTREME_UPPER_FORCE,
                self.RANDOM_LOWER_FORCE,
                self.RANDOM_UPPER_FORCE,
            )
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomWeakForceMountainCar, self).parameters
        parameters.update(
            {
                "force": self.force,
            }
        )
        return parameters


class LightCarMountainCar(ModifiableMountainCarEnv):
    def __init__(self):
        super(LightCarMountainCar, self).__init__()
        self.mass = self.EXTREME_LOWER_MASS

    @property
    def parameters(self):
        parameters = super(LightCarMountainCar, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class HeavyCarMountainCar(ModifiableMountainCarEnv):
    def __init__(self):
        super(HeavyCarMountainCar, self).__init__()
        self.mass = self.EXTREME_UPPER_MASS

    @property
    def parameters(self):
        parameters = super(HeavyCarMountainCar, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomHeavyCarMountainCar(ModifiableMountainCarEnv):
    def reset(self, new=True):
        if new:
            self.mass = self.np_random.uniform(
                self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
            )
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomHeavyCarMountainCar, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomLightCarMountainCar(ModifiableMountainCarEnv):
    def reset(self, new=True):
        if new:
            self.mass = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASS,
                self.EXTREME_UPPER_MASS,
                self.RANDOM_LOWER_MASS,
                self.RANDOM_UPPER_MASS,
            )
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomLightCarMountainCar, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomNormalMountainCar(ModifiableMountainCarEnv):
    def reset(self, new=True):
        self.nsteps = 0  # for is_success()
        if new:
            self.force = self.np_random.uniform(
                self.RANDOM_LOWER_FORCE, self.RANDOM_UPPER_FORCE
            )
            self.mass = self.np_random.uniform(
                self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
            )
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomNormalMountainCar, self).parameters
        parameters.update(
            {
                "force": self.force,
                "mass": self.mass,
            }
        )
        return parameters


class RandomExtremeMountainCar(ModifiableMountainCarEnv):

    # TODO(cpacker): Is there any reason to not have an __init__?
    def reset(self, new=True):
        self.nsteps = 0  # for is_success()
        if new:
            self.force = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_FORCE,
                self.EXTREME_UPPER_FORCE,
                self.RANDOM_LOWER_FORCE,
                self.RANDOM_UPPER_FORCE,
            )
            self.mass = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASS,
                self.EXTREME_UPPER_MASS,
                self.RANDOM_LOWER_MASS,
                self.RANDOM_UPPER_MASS,
            )

        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    @property
    def parameters(self):
        parameters = super(RandomExtremeMountainCar, self).parameters
        parameters.update(
            {
                "force": self.force,
                "mass": self.mass,
            }
        )
        return parameters


# Pendulum environment variants.


class ModifiablePendulumEnv(PendulumEnv):
    """The pendulum environment without length and mass of object hard-coded."""

    RANDOM_LOWER_MASS = 0.75
    RANDOM_UPPER_MASS = 1.25
    EXTREME_LOWER_MASS = 0.5
    EXTREME_UPPER_MASS = 1.5

    RANDOM_LOWER_LENGTH = 0.75
    RANDOM_UPPER_LENGTH = 1.25
    EXTREME_LOWER_LENGTH = 0.5
    EXTREME_UPPER_LENGTH = 1.5

    def __init__(self):
        super(ModifiablePendulumEnv, self).__init__()

        self.mass = 1.0
        self.length = 1.0

    def step(self, u):
        th, thdot = self.state
        g = 10.0
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        angle_normalize = ((th + np.pi) % (2 * np.pi)) - np.pi
        costs = angle_normalize ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = (
            thdot
            + (
                -3 * g / (2 * self.length) * np.sin(th + np.pi)
                + 3.0 / (self.mass * self.length ** 2) * u
            )
            * dt
        )
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        normalized = ((newth + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([newth, newthdot])

        # Extra calculations for is_success()
        # TODO(cpacker): be consistent in increment before or after func body
        self.nsteps += 1
        # Track how long angle has been < pi/3
        if -np.pi / 3 <= normalized and normalized <= np.pi / 3:
            self.nsteps_vertical += 1
        else:
            self.nsteps_vertical = 0
        # Success if if angle has been kept at vertical for 100 steps
        target = 100
        if self.nsteps_vertical >= target:
            # print("[SUCCESS]: nsteps is {}, nsteps_vertical is {}, reached target {}".format(
            #      self.nsteps, self.nsteps_vertical, target))
            self.success = True
        else:
            # print("[NO SUCCESS]: nsteps is {}, nsteps_vertical is {}, target {}".format(
            #      self.nsteps, self.nsteps_vertical, target))
            self.success = False

        return self._get_obs(), -costs, False, {}

    def reset(self, new=True):
        # Extra state for is_success()
        self.nsteps = 0
        self.nsteps_vertical = 0
        return super(ModifiablePendulumEnv, self).reset()

    @property
    def parameters(self):
        return {
            "id": self.spec.id,
        }

    def is_success(self):
        """Returns True if current state indicates success, False otherwise

        Success: keep the angle of the pendulum at most pi/3 radians from
        vertical for the last 100 time steps of a trajectory with length 200
        (max_length is set to 200 in sunblaze_envs/__init__.py)
        """
        return self.success


class LightPendulum(ModifiablePendulumEnv):
    def __init__(self):
        super(LightPendulum, self).__init__()
        self.mass = self.EXTREME_LOWER_MASS

    @property
    def parameters(self):
        parameters = super(LightPendulum, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class HeavyPendulum(ModifiablePendulumEnv):
    def __init__(self):
        super(HeavyPendulum, self).__init__()
        self.mass = self.EXTREME_UPPER_MASS

    @property
    def parameters(self):
        parameters = super(HeavyPendulum, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomHeavyPendulum(ModifiablePendulumEnv):
    def __init__(self):
        super(RandomHeavyPendulum, self).__init__()
        self.mass = self.np_random.uniform(
            self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
        )

    def reset(self, new=True):
        if new:
            self.mass = self.np_random.uniform(
                self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
            )
        return super(RandomHeavyPendulum, self).reset(new)

    @property
    def parameters(self):
        parameters = super(RandomHeavyPendulum, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomLightPendulum(ModifiablePendulumEnv):
    def __init__(self):
        super(RandomLightPendulum, self).__init__()
        self.mass = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_MASS,
            self.EXTREME_UPPER_MASS,
            self.RANDOM_LOWER_MASS,
            self.RANDOM_UPPER_MASS,
        )

    def reset(self, new=True):
        if new:
            self.mass = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASS,
                self.EXTREME_UPPER_MASS,
                self.RANDOM_LOWER_MASS,
                self.RANDOM_UPPER_MASS,
            )
        return super(RandomLightPendulum, self).reset(new)

    @property
    def parameters(self):
        parameters = super(RandomLightPendulum, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class ShortPendulum(ModifiablePendulumEnv):
    def __init__(self):
        super(ShortPendulum, self).__init__()
        self.length = self.EXTREME_LOWER_LENGTH

    @property
    def parameters(self):
        parameters = super(ShortPendulum, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class LongPendulum(ModifiablePendulumEnv):
    def __init__(self):
        super(LongPendulum, self).__init__()
        self.length = self.EXTREME_UPPER_LENGTH

    @property
    def parameters(self):
        parameters = super(LongPendulum, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomLongPendulum(ModifiablePendulumEnv):
    def __init__(self):
        super(RandomLongPendulum, self).__init__()
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
        )

    def reset(self, new=True):
        if new:
            self.length = self.np_random.uniform(
                self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
            )
        return super(RandomLongPendulum, self).reset(new)

    @property
    def parameters(self):
        parameters = super(RandomLongPendulum, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomShortPendulum(ModifiablePendulumEnv):
    def __init__(self):
        super(RandomShortPendulum, self).__init__()
        self.length = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_LENGTH,
            self.EXTREME_UPPER_LENGTH,
            self.RANDOM_LOWER_LENGTH,
            self.RANDOM_UPPER_LENGTH,
        )

    def reset(self, new=True):
        if new:
            self.length = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_LENGTH,
                self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH,
                self.RANDOM_UPPER_LENGTH,
            )
        return super(RandomShortPendulum, self).reset(new)

    @property
    def parameters(self):
        parameters = super(RandomShortPendulum, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomNormalPendulum(ModifiablePendulumEnv):
    def __init__(self):
        super(RandomNormalPendulum, self).__init__()
        self.mass = self.np_random.uniform(
            self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
        )
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
        )

    def reset(self, new=True):
        if new:
            self.mass = self.np_random.uniform(
                self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
            )
            self.length = self.np_random.uniform(
                self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
            )
        return super(RandomNormalPendulum, self).reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalPendulum, self).parameters
        parameters.update(
            {
                "mass": self.mass,
                "length": self.length,
            }
        )
        return parameters


class RandomExtremePendulum(ModifiablePendulumEnv):
    def __init__(self):
        super(RandomExtremePendulum, self).__init__()
        self.mass = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_MASS,
            self.EXTREME_UPPER_MASS,
            self.RANDOM_LOWER_MASS,
            self.RANDOM_UPPER_MASS,
        )
        self.length = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_LENGTH,
            self.EXTREME_UPPER_LENGTH,
            self.RANDOM_LOWER_LENGTH,
            self.RANDOM_UPPER_LENGTH,
        )

    def reset(self, new=True):
        if new:
            self.mass = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASS,
                self.EXTREME_UPPER_MASS,
                self.RANDOM_LOWER_MASS,
                self.RANDOM_UPPER_MASS,
            )
            self.length = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_LENGTH,
                self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH,
                self.RANDOM_UPPER_LENGTH,
            )
        return super(RandomExtremePendulum, self).reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremePendulum, self).parameters
        parameters.update(
            {
                "mass": self.mass,
                "length": self.length,
            }
        )
        return parameters


# Acrobot variants.


class ModifiableAcrobotEnv(AcrobotEnv):

    RANDOM_LOWER_MASS = 0.75
    RANDOM_UPPER_MASS = 1.25
    EXTREME_LOWER_MASS = 0.5
    EXTREME_UPPER_MASS = 1.5

    RANDOM_LOWER_LENGTH = 0.75
    RANDOM_UPPER_LENGTH = 1.25
    EXTREME_LOWER_LENGTH = 0.5
    EXTREME_UPPER_LENGTH = 1.5

    RANDOM_LOWER_INERTIA = 0.75
    RANDOM_UPPER_INERTIA = 1.25
    EXTREME_LOWER_INERTIA = 0.5
    EXTREME_UPPER_INERTIA = 1.5

    def reset(self, new=True):
        self.nsteps = 0
        return super(ModifiableAcrobotEnv, self).reset()

    @property
    def parameters(self):
        return {
            "id": self.spec.id,
        }

    def step(self, *args, **kwargs):
        """Wrapper to increment new variable nsteps"""

        self.nsteps += 1
        ret = super().step(*args, **kwargs)

        # Moved logic to step wrapper because success triggers done which
        # triggers reset() in a higher level step wrapper
        # With logic in is_success(),
        # we need to cache the 'done' flag ourselves to use in is_success(),
        # since the wrapper around this wrapper will call reset immediately

        target = 90
        if self.nsteps <= target and self._terminal():
            # print("[SUCCESS]: nsteps is {}, reached done in target {}".format(
            #      self.nsteps, target))
            self.success = True
        else:
            # print("[NO SUCCESS]: nsteps is {}, step limit {}".format(
            #      self.nsteps, target))
            self.success = False

        return ret

    def is_success(self):
        """Returns True if current state indicates success, False otherwise

        Success: swing the end of the second link to the desired height within
        90 time steps
        """
        return self.success


class LightAcrobot(ModifiableAcrobotEnv):
    def __init__(self):
        super(LightAcrobot, self).__init__()
        self.mass = self.EXTREME_LOWER_MASS

    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def parameters(self):
        parameters = super(LightAcrobot, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class HeavyAcrobot(ModifiableAcrobotEnv):
    def __init__(self):
        super(HeavyAcrobot, self).__init__()
        self.mass = self.EXTREME_UPPER_MASS

    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def parameters(self):
        parameters = super(HeavyAcrobot, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomHeavyAcrobot(ModifiableAcrobotEnv):
    def __init__(self):
        super(RandomHeavyAcrobot, self).__init__()
        self.mass = self.np_random.uniform(
            self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
        )

    def reset(self, new=True):
        if new:
            self.mass = self.np_random.uniform(
                self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
            )
        return super(RandomHeavyAcrobot, self).reset(new)

    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def parameters(self):
        parameters = super(RandomHeavyAcrobot, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomLightAcrobot(ModifiableAcrobotEnv):
    def __init__(self):
        super(RandomLightAcrobot, self).__init__()
        self.mass = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_MASS,
            self.EXTREME_UPPER_MASS,
            self.RANDOM_LOWER_MASS,
            self.RANDOM_UPPER_MASS,
        )

    def reset(self, new=True):
        if new:
            self.mass = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASS,
                self.EXTREME_UPPER_MASS,
                self.RANDOM_LOWER_MASS,
                self.RANDOM_UPPER_MASS,
            )
        return super(RandomLightAcrobot, self).reset(new)

    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def parameters(self):
        parameters = super(RandomLightAcrobot, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class ShortAcrobot(ModifiableAcrobotEnv):
    def __init__(self):
        super(ShortAcrobot, self).__init__()
        self.length = self.EXTREME_LOWER_LENGTH

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def parameters(self):
        parameters = super(ShortAcrobot, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class LongAcrobot(ModifiableAcrobotEnv):
    def __init__(self):
        super(LongAcrobot, self).__init__()
        self.length = self.EXTREME_UPPER_LENGTH

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def parameters(self):
        parameters = super(LongAcrobot, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomLongAcrobot(ModifiableAcrobotEnv):
    def __init__(self):
        super(RandomLongAcrobot, self).__init__()
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
        )

    def reset(self, new=True):
        if new:
            self.length = self.np_random.uniform(
                self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
            )
        return super(RandomLongAcrobot, self).reset(new)

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def parameters(self):
        parameters = super(RandomLongAcrobot, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomShortAcrobot(ModifiableAcrobotEnv):
    def __init__(self):
        super(RandomShortAcrobot, self).__init__()
        self.length = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_LENGTH,
            self.EXTREME_UPPER_LENGTH,
            self.RANDOM_LOWER_LENGTH,
            self.RANDOM_UPPER_LENGTH,
        )

    def reset(self, new=True):
        if new:
            self.length = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_LENGTH,
                self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH,
                self.RANDOM_UPPER_LENGTH,
            )
        return super(RandomShortAcrobot, self).reset(new)

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def parameters(self):
        parameters = super(RandomShortAcrobot, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class LowInertiaAcrobot(ModifiableAcrobotEnv):
    def __init__(self):
        super(LowInertiaAcrobot, self).__init__()
        self.inertia = self.EXTREME_LOWER_INERTIA

    @property
    def LINK_MOI(self):
        return self.inertia

    @property
    def parameters(self):
        parameters = super(LowInertiaAcrobot, self).parameters
        parameters.update(
            {
                "inertia": self.inertia,
            }
        )
        return parameters


class HighInertiaAcrobot(ModifiableAcrobotEnv):
    def __init__(self):
        super(HighInertiaAcrobot, self).__init__()
        self.inertia = self.EXTREME_UPPER_INERTIA

    @property
    def LINK_MOI(self):
        return self.inertia

    @property
    def parameters(self):
        parameters = super(HighInertiaAcrobot, self).parameters
        parameters.update(
            {
                "inertia": self.inertia,
            }
        )
        return parameters


class RandomHighInertiaAcrobot(ModifiableAcrobotEnv):
    def __init__(self):
        super(RandomHighInertiaAcrobot, self).__init__()
        self.inertia = self.np_random.uniform(
            self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA
        )

    def reset(self, new=True):
        if new:
            self.inertia = self.np_random.uniform(
                self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA
            )
        return super(RandomHighInertiaAcrobot, self).reset(new)

    @property
    def LINK_MOI(self):
        return self.inertia

    @property
    def parameters(self):
        parameters = super(RandomHighInertiaAcrobot, self).parameters
        parameters.update(
            {
                "inertia": self.inertia,
            }
        )
        return parameters


class RandomLowInertiaAcrobot(ModifiableAcrobotEnv):
    def __init__(self):
        super(RandomLowInertiaAcrobot, self).__init__()
        self.inertia = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_INERTIA,
            self.EXTREME_UPPER_INERTIA,
            self.RANDOM_LOWER_INERTIA,
            self.RANDOM_UPPER_INERTIA,
        )

    def reset(self, new=True):
        if new:
            self.inertia = self.np_random.uniform(
                self.np_random.uniform,
                self.EXTREME_LOWER_INERTIA,
                self.EXTREME_UPPER_INERTIA,
                self.RANDOM_LOWER_INERTIA,
                self.RANDOM_UPPER_INERTIA,
            )
        return super(RandomLowInertiaAcrobot, self).reset(new)

    @property
    def LINK_MOI(self):
        return self.inertia

    @property
    def parameters(self):
        parameters = super(RandomLowInertiaAcrobot, self).parameters
        parameters.update(
            {
                "inertia": self.inertia,
            }
        )
        return parameters


class RandomNormalAcrobot(ModifiableAcrobotEnv):
    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def LINK_MOI(self):
        return self.inertia

    def __init__(self):
        super(RandomNormalAcrobot, self).__init__()
        self.mass = self.np_random.uniform(
            self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
        )
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
        )
        self.inertia = self.np_random.uniform(
            self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA
        )

    def reset(self, new=True):
        if new:
            self.mass = self.np_random.uniform(
                self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
            )
            self.length = self.np_random.uniform(
                self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
            )
            self.inertia = self.np_random.uniform(
                self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA
            )
        # reset just resets .state
        return super(RandomNormalAcrobot, self).reset()

    @property
    def parameters(self):
        parameters = super(RandomNormalAcrobot, self).parameters
        parameters.update(
            {
                "mass": self.mass,
                "length": self.length,
                "inertia": self.inertia,
            }
        )
        return parameters


class RandomExtremeAcrobot(ModifiableAcrobotEnv):
    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def LINK_MOI(self):
        return self.inertia

    def __init__(self):
        super(RandomExtremeAcrobot, self).__init__()
        self.mass = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_MASS,
            self.EXTREME_UPPER_MASS,
            self.RANDOM_LOWER_MASS,
            self.RANDOM_UPPER_MASS,
        )
        self.length = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_LENGTH,
            self.EXTREME_UPPER_LENGTH,
            self.RANDOM_LOWER_LENGTH,
            self.RANDOM_UPPER_LENGTH,
        )
        self.inertia = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_INERTIA,
            self.EXTREME_UPPER_INERTIA,
            self.RANDOM_LOWER_INERTIA,
            self.RANDOM_UPPER_INERTIA,
        )

    def reset(self, new=True):
        if new:
            self.mass = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASS,
                self.EXTREME_UPPER_MASS,
                self.RANDOM_LOWER_MASS,
                self.RANDOM_UPPER_MASS,
            )
            self.length = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_LENGTH,
                self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH,
                self.RANDOM_UPPER_LENGTH,
            )
            self.inertia = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_INERTIA,
                self.EXTREME_UPPER_INERTIA,
                self.RANDOM_LOWER_INERTIA,
                self.RANDOM_UPPER_INERTIA,
            )
        # reset just resets .state
        return super(RandomExtremeAcrobot, self).reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremeAcrobot, self).parameters
        parameters.update(
            {
                "mass": self.mass,
                "length": self.length,
                "inertia": self.inertia,
            }
        )
        return parameters
