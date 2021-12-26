import contextlib
import os
import tempfile

import numpy as np
import xml.etree.ElementTree as ET
from gym import spaces

import roboschool
from roboschool.gym_mujoco_walkers import (
    RoboschoolForwardWalkerMujocoXML,
    RoboschoolHalfCheetah,
    RoboschoolHopper,
    RoboschoolWalker2d,
)

from .base import EnvBinarySuccessMixin
from .classic_control import uniform_exclude_inner

# Determine Roboschool asset location based on its module path.
ROBOSCHOOL_ASSETS = os.path.join(roboschool.__path__[0], "mujoco_assets")


class RoboschoolTrackDistSuccessMixin(EnvBinarySuccessMixin):
    """Treat reaching certain distance on track as a success."""

    def is_success(self):
        """Returns True is current state indicates success, False otherwise

        x=100 correlates to the end of the track on Roboschool,
        but with the default 1000 max episode length most (all?) agents
        won't reach it (DD PPO2 Hopper reaches ~40), so we use something lower
        """
        target_dist = 20
        if self.robot_body.pose().xyz()[0] >= target_dist:
            return True
        else:
            return False


class RoboschoolXMLModifierMixin:
    """Mixin with XML modification methods."""

    @contextlib.contextmanager
    def modify_xml(self, asset):
        """Context manager allowing XML asset modifcation."""

        # tree = ET.ElementTree(ET.Element(os.path.join(ROBOSCHOOL_ASSETS, asset)))
        tree = ET.parse(os.path.join(ROBOSCHOOL_ASSETS, asset))
        yield tree

        # Create a new temporary .xml file
        # mkstemp returns (int(file_descriptor), str(full_path))
        fd, path = tempfile.mkstemp(suffix=".xml")
        # Close the file to prevent a file descriptor leak
        # See: https://www.logilab.org/blogentry/17873
        # We can also wrap tree.write in 'with os.fdopen(fd, 'w')' instead
        os.close(fd)
        tree.write(path)

        # Delete previous file before overwriting self.model_xml
        if os.path.isfile(self.model_xml):
            os.remove(self.model_xml)
        self.model_xml = path

        # Original fix using mktemp:
        # mktemp (depreciated) returns str(full_path)
        #   modified_asset = tempfile.mktemp(suffix='.xml')
        #   tree.write(modified_asset)
        #   self.model_xml = modified_asset

    def __del__(self):
        """Deletes last remaining xml files after use"""
        # (Note: this won't ensure the final tmp file is deleted on a crash/SIGBREAK/etc.)
        if os.path.isfile(self.model_xml):
            os.remove(self.model_xml)


# Half Cheetah (Packer 2018)


class ModifiableRoboschoolHalfCheetah(
    RoboschoolHalfCheetah, RoboschoolTrackDistSuccessMixin
):

    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 1.4

    RANDOM_LOWER_POWER = 0.7
    RANDOM_UPPER_POWER = 1.1
    EXTREME_LOWER_POWER = 0.5
    EXTREME_UPPER_POWER = 1.3

    def _reset(self, new=True):
        return super(ModifiableRoboschoolHalfCheetah, self)._reset()

    @property
    def parameters(self):
        return {
            "id": self.spec.id,
        }


class RandomNormalHalfCheetah(
    RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah
):
    def __init__(self, oracle: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.oracle = oracle
        if oracle == True:
            print("WARNING! YOU ARE USING MDP, NOT POMDP!\n")
            self._reset()
            tmp_hidden_states = self.get_hidden_states()
            self.observation_space = spaces.Box(
                low=np.array(
                    [*self.observation_space.low, *([0] * len(tmp_hidden_states))]
                ),  # shape will be deduced from this
                high=np.array(
                    [*self.observation_space.high, *([1] * len(tmp_hidden_states))]
                ),
                dtype=np.float32,
            )

    def randomize_env(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY
        )
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION
        )
        self.power = self.np_random.uniform(
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER
        )

        with self.modify_xml("half_cheetah.xml") as tree:
            for elem in tree.iterfind("worldbody/body/geom"):
                elem.set("density", str(self.density))
            for elem in tree.iterfind("default/geom"):
                elem.set("friction", str(self.friction) + " .1 .1")

    def get_obs(self, state):
        if self.oracle:
            hidden_states = self.get_hidden_states()
            state = np.concatenate([state, hidden_states])
            return state
        else:
            return state

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        state = super(RandomNormalHalfCheetah, self)._reset(new)
        return self.get_obs(state)

    def _step(self, a):
        state, reward, done, info = super(RandomNormalHalfCheetah, self)._step(a)
        return self.get_obs(state), reward, done, info

    @property
    def parameters(self):
        parameters = super(RandomNormalHalfCheetah, self).parameters
        parameters.update(
            {
                "power": self.power,
                "density": self.density,
                "friction": self.friction,
            }
        )
        return parameters

    def get_hidden_states(self):
        hidden_states = np.array(
            [
                (self.density - self.RANDOM_LOWER_DENSITY)
                / (self.RANDOM_UPPER_DENSITY - self.RANDOM_LOWER_DENSITY),
                (self.friction - self.RANDOM_LOWER_FRICTION)
                / (self.RANDOM_UPPER_FRICTION - self.RANDOM_LOWER_FRICTION),
                (self.power - self.RANDOM_LOWER_POWER)
                / (self.RANDOM_UPPER_POWER - self.RANDOM_LOWER_POWER),
            ]
        )
        return hidden_states.copy()


class RandomExtremeHalfCheetah(
    RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah
):
    def __init__(self, oracle: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.oracle = oracle
        if oracle == True:
            print("WARNING! YOU ARE USING MDP, NOT POMDP!\n")
            self._reset()
            tmp_hidden_states = self.get_hidden_states()
            self.observation_space = spaces.Box(
                low=np.array(
                    [*self.observation_space.low, *([0] * len(tmp_hidden_states))]
                ),  # shape will be deduced from this
                high=np.array(
                    [*self.observation_space.high, *([1] * len(tmp_hidden_states))]
                ),
                dtype=np.float32,
            )

    def randomize_env(self):
        """
        # self.armature = self.np_random.uniform(0.2, 0.5)
        self.density = self.np_random.uniform(self.LOWER_DENSITY, self.UPPER_DENSITY)
        self.friction = self.np_random.uniform(self.LOWER_FRICTION, self.UPPER_FRICTION)
        self.power = self.np_random.uniform(self.LOWER_POWER, self.UPPER_POWER)
        """

        self.density = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_DENSITY,
            self.EXTREME_UPPER_DENSITY,
            self.RANDOM_LOWER_DENSITY,
            self.RANDOM_UPPER_DENSITY,
        )
        self.friction = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_FRICTION,
            self.EXTREME_UPPER_FRICTION,
            self.RANDOM_LOWER_FRICTION,
            self.RANDOM_UPPER_FRICTION,
        )
        self.power = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_POWER,
            self.EXTREME_UPPER_POWER,
            self.RANDOM_LOWER_POWER,
            self.RANDOM_UPPER_POWER,
        )

        with self.modify_xml("half_cheetah.xml") as tree:
            for elem in tree.iterfind("worldbody/body/geom"):
                elem.set("density", str(self.density))
            for elem in tree.iterfind("default/geom"):
                elem.set("friction", str(self.friction) + " .1 .1")

    def get_obs(self, state):
        if self.oracle:
            hidden_states = self.get_hidden_states()
            state = np.concatenate([state, hidden_states])
            return state
        else:
            return state

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        state = super(RandomExtremeHalfCheetah, self)._reset(new)
        return self.get_obs(state)

    def _step(self, a):
        state, reward, done, info = super(RandomExtremeHalfCheetah, self)._step(a)
        return self.get_obs(state), reward, done, info

    @property
    def parameters(self):
        parameters = super(RandomExtremeHalfCheetah, self).parameters
        parameters.update(
            {
                "power": self.power,
                "density": self.density,
                "friction": self.friction,
            }
        )
        return parameters

    def get_hidden_states(self):
        hidden_states = np.array(
            [
                (self.density - self.EXTREME_LOWER_DENSITY)
                / (self.EXTREME_UPPER_DENSITY - self.EXTREME_LOWER_DENSITY),
                (self.friction - self.EXTREME_LOWER_FRICTION)
                / (self.EXTREME_UPPER_FRICTION - self.EXTREME_LOWER_FRICTION),
                (self.power - self.EXTREME_LOWER_POWER)
                / (self.EXTREME_UPPER_POWER - self.EXTREME_LOWER_POWER),
            ]
        )
        return hidden_states.copy()


# Hopper (Packer 2018)


class ModifiableRoboschoolHopper(RoboschoolHopper, RoboschoolTrackDistSuccessMixin):

    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 1.4

    RANDOM_LOWER_POWER = 0.6
    RANDOM_UPPER_POWER = 0.9
    EXTREME_LOWER_POWER = 0.4
    EXTREME_UPPER_POWER = 1.1

    def _reset(self, new=True):
        return super(ModifiableRoboschoolHopper, self)._reset()

    @property
    def parameters(self):
        return {
            "id": self.spec.id,
        }


class RandomNormalHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def __init__(self, oracle: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.oracle = oracle
        if oracle == True:
            print("WARNING! YOU ARE USING MDP, NOT POMDP!\n")
            self._reset()
            tmp_hidden_states = self.get_hidden_states()
            self.observation_space = spaces.Box(
                low=np.array(
                    [*self.observation_space.low, *([0] * len(tmp_hidden_states))]
                ),  # shape will be deduced from this
                high=np.array(
                    [*self.observation_space.high, *([1] * len(tmp_hidden_states))]
                ),
                dtype=np.float32,
            )

    def randomize_env(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY
        )
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION
        )
        self.power = self.np_random.uniform(
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER
        )
        with self.modify_xml("hopper.xml") as tree:
            for elem in tree.iterfind("worldbody/body/geom"):
                elem.set("density", str(self.density))
            for elem in tree.iterfind("default/geom"):
                elem.set("friction", str(self.friction) + " .1 .1")

    def get_obs(self, state):
        if self.oracle:
            hidden_states = self.get_hidden_states()
            state = np.concatenate([state, hidden_states])
            return state
        else:
            return state

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        state = super(RandomNormalHopper, self)._reset(new)
        return self.get_obs(state)

    def _step(self, a):
        state, reward, done, info = super(RandomNormalHopper, self)._step(a)
        return self.get_obs(state), reward, done, info

    @property
    def parameters(self):
        parameters = super(RandomNormalHopper, self).parameters
        parameters.update(
            {
                "power": self.power,
                "density": self.density,
                "friction": self.friction,
            }
        )
        return parameters

    def get_hidden_states(self):
        hidden_states = np.array(
            [
                (self.density - self.RANDOM_LOWER_DENSITY)
                / (self.RANDOM_UPPER_DENSITY - self.RANDOM_LOWER_DENSITY),
                (self.friction - self.RANDOM_LOWER_FRICTION)
                / (self.RANDOM_UPPER_FRICTION - self.RANDOM_LOWER_FRICTION),
                (self.power - self.RANDOM_LOWER_POWER)
                / (self.RANDOM_UPPER_POWER - self.RANDOM_LOWER_POWER),
            ]
        )
        return hidden_states.copy()


class RandomExtremeHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def __init__(self, oracle: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.oracle = oracle
        if oracle == True:
            print("WARNING! YOU ARE USING MDP, NOT POMDP!\n")
            self._reset()
            tmp_hidden_states = self.get_hidden_states()
            self.observation_space = spaces.Box(
                low=np.array(
                    [*self.observation_space.low, *([0] * len(tmp_hidden_states))]
                ),  # shape will be deduced from this
                high=np.array(
                    [*self.observation_space.high, *([1] * len(tmp_hidden_states))]
                ),
                dtype=np.float32,
            )

    def randomize_env(self):
        """
        self.density = self.np_random.uniform(self.LOWER_DENSITY, self.UPPER_DENSITY)
        self.friction = self.np_random.uniform(self.LOWER_FRICTION, self.UPPER_FRICTION)
        self.power = self.np_random.uniform(self.LOWER_POWER, self.UPPER_POWER)
        """

        self.density = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_DENSITY,
            self.EXTREME_UPPER_DENSITY,
            self.RANDOM_LOWER_DENSITY,
            self.RANDOM_UPPER_DENSITY,
        )
        self.friction = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_FRICTION,
            self.EXTREME_UPPER_FRICTION,
            self.RANDOM_LOWER_FRICTION,
            self.RANDOM_UPPER_FRICTION,
        )
        self.power = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_POWER,
            self.EXTREME_UPPER_POWER,
            self.RANDOM_LOWER_POWER,
            self.RANDOM_UPPER_POWER,
        )

        with self.modify_xml("hopper.xml") as tree:
            for elem in tree.iterfind("worldbody/body/geom"):
                elem.set("density", str(self.density))
            for elem in tree.iterfind("default/geom"):
                elem.set("friction", str(self.friction) + " .1 .1")

    def get_obs(self, state):
        if self.oracle:
            hidden_states = self.get_hidden_states()
            state = np.concatenate([state, hidden_states])
            return state
        else:
            return state

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        state = super(RandomExtremeHopper, self)._reset(new)
        return self.get_obs(state)

    def _step(self, a):
        state, reward, done, info = super(RandomExtremeHopper, self)._step(a)
        return self.get_obs(state), reward, done, info

    @property
    def parameters(self):
        parameters = super(RandomExtremeHopper, self).parameters
        parameters.update(
            {
                "power": self.power,
                "density": self.density,
                "friction": self.friction,
            }
        )
        return parameters

    def get_hidden_states(self):
        hidden_states = np.array(
            [
                (self.density - self.EXTREME_LOWER_DENSITY)
                / (self.EXTREME_UPPER_DENSITY - self.EXTREME_LOWER_DENSITY),
                (self.friction - self.EXTREME_LOWER_FRICTION)
                / (self.EXTREME_UPPER_FRICTION - self.EXTREME_LOWER_FRICTION),
                (self.power - self.EXTREME_LOWER_POWER)
                / (self.EXTREME_UPPER_POWER - self.EXTREME_LOWER_POWER),
            ]
        )
        return hidden_states.copy()


# Walker2d (Jiang 2021)


class ModifiableRoboschoolWalker2d_MRPO(
    RoboschoolWalker2d, RoboschoolTrackDistSuccessMixin
):
    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 1
    EXTREME_UPPER_DENSITY = 750

    # NOTE: we follow Table 1 in (Jiang 2021) and changed the friction bounds in their ICML code
    RANDOM_LOWER_FRICTION = 0.2
    RANDOM_UPPER_FRICTION = 2.5
    EXTREME_LOWER_FRICTION = 0.05
    EXTREME_UPPER_FRICTION = 0.2

    def _reset(self, new=True):
        return super(ModifiableRoboschoolWalker2d_MRPO, self)._reset()

    @property
    def parameters(self):
        return {
            "id": self.spec.id,
        }


class RandomNormalWalker2d_MRPO(
    RoboschoolXMLModifierMixin, ModifiableRoboschoolWalker2d_MRPO
):
    def __init__(self, oracle: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.oracle = oracle
        if oracle == True:
            print("WARNING! YOU ARE USING MDP, NOT POMDP!\n")
            self._reset()
            tmp_hidden_states = self.get_hidden_states()
            self.observation_space = spaces.Box(
                low=np.array(
                    [*self.observation_space.low, *([0] * len(tmp_hidden_states))]
                ),  # shape will be deduced from this
                high=np.array(
                    [*self.observation_space.high, *([1] * len(tmp_hidden_states))]
                ),
                dtype=np.float32,
            )

    def randomize_env(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY
        )
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION
        )
        # self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml("walker2d.xml") as tree:
            # for elem in tree.iterfind('worldbody/body/geom'):
            #     elem.set('density', str(self.density))
            for elem in tree.iterfind("default/geom"):
                elem.set("density", str(self.density) + " .1 .1")
            for elem in tree.iterfind("default/geom"):
                elem.set("friction", str(self.friction) + " .1 .1")

    def get_obs(self, state):
        if self.oracle:
            hidden_states = self.get_hidden_states()
            state = np.concatenate([state, hidden_states])
            return state
        else:
            return state

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        state = super(RandomNormalWalker2d_MRPO, self)._reset(new)
        return self.get_obs(state)

    def _step(self, a):
        state, reward, done, info = super(RandomNormalWalker2d_MRPO, self)._step(a)
        return self.get_obs(state), reward, done, info

    @property
    def parameters(self):
        parameters = super(RandomNormalWalker2d_MRPO, self).parameters
        parameters.update(
            {
                "density": self.density,
                "friction": self.friction,
            }
        )
        return parameters

    def get_hidden_states(self):
        hidden_states = np.array(
            [
                (self.density - self.RANDOM_LOWER_DENSITY)
                / (self.RANDOM_UPPER_DENSITY - self.RANDOM_LOWER_DENSITY),
                (self.friction - self.RANDOM_LOWER_FRICTION)
                / (self.RANDOM_UPPER_FRICTION - self.RANDOM_LOWER_FRICTION),
            ]
        )
        return hidden_states.copy()


# Half Cheetah (Jiang 2021)


class ModifiableRoboschoolHalfCheetah_MRPO(
    RoboschoolHalfCheetah, RoboschoolTrackDistSuccessMixin
):

    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 1
    EXTREME_UPPER_DENSITY = 750

    # NOTE: we follow Table 1 in (Jiang 2021) and changed the friction bounds in their ICML code
    RANDOM_LOWER_FRICTION = 0.2
    RANDOM_UPPER_FRICTION = 2.25
    EXTREME_LOWER_FRICTION = 0.05
    EXTREME_UPPER_FRICTION = 0.2

    def _reset(self, new=True):
        return super(ModifiableRoboschoolHalfCheetah_MRPO, self)._reset()

    @property
    def parameters(self):
        return {
            "id": self.spec.id,
        }


class RandomNormalHalfCheetah_MRPO(
    RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah_MRPO
):
    def __init__(self, oracle: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.oracle = oracle
        if oracle == True:
            print("WARNING! YOU ARE USING MDP, NOT POMDP!\n")
            self._reset()
            tmp_hidden_states = self.get_hidden_states()
            self.observation_space = spaces.Box(
                low=np.array(
                    [*self.observation_space.low, *([0] * len(tmp_hidden_states))]
                ),  # shape will be deduced from this
                high=np.array(
                    [*self.observation_space.high, *([1] * len(tmp_hidden_states))]
                ),
                dtype=np.float32,
            )

    def randomize_env(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY
        )
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION
        )

        with self.modify_xml("half_cheetah.xml") as tree:
            for elem in tree.iterfind("worldbody/body/geom"):
                elem.set("density", str(self.density))
            for elem in tree.iterfind("default/geom"):
                elem.set("friction", str(self.friction) + " .1 .1")

    def get_obs(self, state):
        if self.oracle:
            hidden_states = self.get_hidden_states()
            state = np.concatenate([state, hidden_states])
            return state
        else:
            return state

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        state = super(RandomNormalHalfCheetah_MRPO, self)._reset(new)
        return self.get_obs(state)

    def _step(self, a):
        state, reward, done, info = super(RandomNormalHalfCheetah_MRPO, self)._step(a)
        return self.get_obs(state), reward, done, info

    @property
    def parameters(self):
        parameters = super(RandomNormalHalfCheetah_MRPO, self).parameters
        parameters.update(
            {
                "density": self.density,
                "friction": self.friction,
            }
        )
        return parameters

    def get_hidden_states(self):
        hidden_states = np.array(
            [
                (self.density - self.RANDOM_LOWER_DENSITY)
                / (self.RANDOM_UPPER_DENSITY - self.RANDOM_LOWER_DENSITY),
                (self.friction - self.RANDOM_LOWER_FRICTION)
                / (self.RANDOM_UPPER_FRICTION - self.RANDOM_LOWER_FRICTION),
            ]
        )
        return hidden_states.copy()


# Hopper (Jiang 2021)


class ModifiableRoboschoolHopper_MRPO(
    RoboschoolHopper, RoboschoolTrackDistSuccessMixin
):

    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 1
    EXTREME_UPPER_DENSITY = 750

    # NOTE: we follow Table 1 in (Jiang 2021) and changed the friction bounds in their ICML code
    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 0.5

    def _reset(self, new=True):
        return super(ModifiableRoboschoolHopper_MRPO, self)._reset()

    @property
    def parameters(self):
        return {
            "id": self.spec.id,
        }


class RandomNormalHopper_MRPO(
    RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper_MRPO
):
    def __init__(self, oracle: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.oracle = oracle
        if oracle == True:
            print("WARNING! YOU ARE USING MDP, NOT POMDP!\n")
            self._reset()
            tmp_hidden_states = self.get_hidden_states()
            self.observation_space = spaces.Box(
                low=np.array(
                    [*self.observation_space.low, *([0] * len(tmp_hidden_states))]
                ),  # shape will be deduced from this
                high=np.array(
                    [*self.observation_space.high, *([1] * len(tmp_hidden_states))]
                ),
                dtype=np.float32,
            )

    def randomize_env(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY
        )
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION
        )
        with self.modify_xml("hopper.xml") as tree:
            for elem in tree.iterfind("worldbody/body/geom"):
                elem.set("density", str(self.density))
            for elem in tree.iterfind("default/geom"):
                elem.set("friction", str(self.friction) + " .1 .1")

    def get_obs(self, state):
        if self.oracle:
            hidden_states = self.get_hidden_states()
            state = np.concatenate([state, hidden_states])
            return state
        else:
            return state

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        state = super(RandomNormalHopper_MRPO, self)._reset(new)
        return self.get_obs(state)

    def _step(self, a):
        state, reward, done, info = super(RandomNormalHopper_MRPO, self)._step(a)
        return self.get_obs(state), reward, done, info

    @property
    def parameters(self):
        parameters = super(RandomNormalHopper_MRPO, self).parameters
        parameters.update(
            {
                "density": self.density,
                "friction": self.friction,
            }
        )
        return parameters

    def get_hidden_states(self):
        hidden_states = np.array(
            [
                (self.density - self.RANDOM_LOWER_DENSITY)
                / (self.RANDOM_UPPER_DENSITY - self.RANDOM_LOWER_DENSITY),
                (self.friction - self.RANDOM_LOWER_FRICTION)
                / (self.RANDOM_UPPER_FRICTION - self.RANDOM_LOWER_FRICTION),
            ]
        )
        return hidden_states.copy()
