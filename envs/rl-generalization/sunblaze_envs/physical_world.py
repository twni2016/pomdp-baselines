import os

import Box2D as box_2d
import cocos
import numpy as np
import pyglet
from gym import error, spaces
from gym.utils import seeding

from .base import BaseGymEnvironment

ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets", "physical_world")


class PhysicalObject(cocos.sprite.Sprite):
    """Sprite which is backed by a physical object."""

    def __init__(self, image, **kwargs):
        world = kwargs.pop("world", None)
        super(PhysicalObject, self).__init__(image, **kwargs)

        if world is not None:
            # Create the physical representation of this object.
            self._world = world
            self._engine = world.engine
            self._body = self.create_physical_entity()
            self._body.userData = self
        else:
            self._world = None
            self._engine = None
            self._body = None

    @property
    def body(self):
        """Physical body."""
        return self._body

    @property
    def physical_position(self):
        """Returns physical object position."""
        if getattr(self, "_body", None) is not None:
            return self._body.position

        return (
            self.position[0] / self._world.physical_scale,
            self.position[1] / self._world.physical_scale,
        )

    @property
    def physical_rotation(self):
        """Returns physical object rotation (in radians)."""
        if getattr(self, "_body", None) is not None:
            return self._body.angle

        return -np.deg2rad(self.rotation)

    @property
    def visual_position(self):
        """Return visual object position."""
        if getattr(self, "_body", None) is None:
            return self.position

        return self._body.position * self._world.physical_scale

    @property
    def visual_rotation(self):
        """Return visual object rotation (in degrees)."""
        if getattr(self, "_body", None) is None:
            return self.rotation

        return -np.rad2deg(self._body.angle)

    def set_body_position(self, position):
        """Set object position."""
        self._body.position = (
            position[0] / self._world.physical_scale,
            position[1] / self._world.physical_scale,
        )

    def stop_body(self):
        """Stop body movement."""
        self._body.linearVelocity = (0, 0)

    def create_physical_entity(self):
        """Create the entity in the physics engine."""
        raise NotImplementedError

    def step(self):
        """Update actual object based on physical entity."""
        if not self._body:
            return

        self.position = self.visual_position
        self.rotation = self.visual_rotation

    def kill(self):
        """Kill the given object."""
        if not self._body:
            return

        if self._engine is not None:
            self._world.destroy_body(self._body)
            self._body.userData = None
            self._body = None

        super(PhysicalObject, self).kill()

    def apply_impulse(self, vector):
        """Apply linear impulse to center of mass."""
        self._body.ApplyLinearImpulse(vector, self._body.worldCenter, True)

    def on_contact(self, other):
        """Handle contact with another body."""
        pass

    def should_collide(self, other):
        """Handle collision filtering with another body."""
        return True


class ContactListener(box_2d.b2ContactListener):
    def BeginContact(self, contact):
        object_a = contact.fixtureA.body.userData
        object_b = contact.fixtureB.body.userData

        if object_a:
            object_a.on_contact(object_b)
        if object_b:
            object_b.on_contact(object_a)


class ContactFilter(box_2d.b2ContactFilter):
    def ShouldCollide(self, fixture_a, fixture_b):
        object_a = fixture_a.body.userData
        object_b = fixture_b.body.userData

        if not object_a or not object_b:
            return True

        return object_a.should_collide(object_b) and object_b.should_collide(object_a)


class PhysicalWorld(cocos.layer.Layer):
    """Physical world, which may be rendered."""

    fps = 50
    # Ratio between physical size and graphics size.
    physical_scale = 32.0
    # Number of actions.
    n_actions = 0

    def __init__(self, width, height):
        super(PhysicalWorld, self).__init__()

        # Create the world in the physics engine.
        self._contacts = ContactListener()
        self._filter = ContactFilter()
        self._engine = box_2d.b2World(
            gravity=(0, 0),
            contactListener=self._contacts,
            contactFilter=self._filter,
        )
        self._width, self._height = width, height
        self._destroy_queue = []

        self.add(cocos.layer.ColorLayer(0, 0, 0, 255))

        self._batch = cocos.batch.BatchNode()
        self.add(self._batch)

        self.seed()
        self.create_world(self._batch)
        self.reset_world()

        self._terminal = False

    def create_world(self, parent):
        """Create the physical world."""
        raise NotImplementedError

    def reset_world(self):
        """Reset the world."""
        self._terminal = False

    def act(self, action):
        """Perform an external action in the world."""
        pass

    def seed(self, seed=None):
        """Setup random number generator."""
        self.np_random, seed = seeding.np_random(seed)
        return seed

    @property
    def is_terminal(self):
        return self._terminal

    @property
    def engine(self):
        """Physics engine world."""
        return self._engine

    @property
    def ground(self):
        """Ground body."""
        return self._ground

    @property
    def parameters(self):
        """World-defining parameters."""
        return {}

    def destroy_body(self, body):
        """Queue specific body for destruction."""
        self._destroy_queue.append(body)

    def process_destroy_queue(self):
        """Process any pending object destructions."""
        for body in self._destroy_queue:
            self._engine.DestroyBody(body)

        self._destroy_queue = []

    def step(self):
        """Perform one simulation step."""
        self.process_destroy_queue()
        self._engine.Step(1.0 / self.fps, 6 * 30, 2 * 30)
        self._engine.ClearForces()

        # Step all objects.
        def step_node(node):
            if not isinstance(node, PhysicalObject):
                return

            node.step()

        self.walk(step_node)


class PhysicalEnvironment:
    """Physical environment based on Box2D/Cocos2D."""

    def __init__(self, world):
        width = 640
        height = 480
        window = getattr(cocos.director.director, "window", None)
        if window is None:
            # Update resource path.
            pyglet.resource.path = [ASSET_PATH]
            pyglet.resource.reindex()

            # Initialize new window.
            window = cocos.director.director.init(width=width, height=height)

        self._window = window
        self._width = width
        self._height = height
        self._world = world(width=width, height=height)
        self._scene = cocos.scene.Scene(
            cocos.layer.ColorLayer(0, 0, 0, 255),
            self._world,
        )

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def world(self):
        return self._world

    @property
    def is_terminal(self):
        return self._world.is_terminal

    def step(self):
        """Perform one environment update step."""
        self._world.step()
        return self.is_terminal

    def reset(self):
        """Reset the world."""
        self._world.reset_world()

    def act(self, action):
        """Perform an action on the world."""
        self._world.act(action)

    def seed(self, seed=None):
        """Seed random number generator."""
        return self._world.seed(seed)

    def render(self, mode="human"):
        """Render the environment."""
        if cocos.director.director.scene != self._scene:
            cocos.director.director._set_scene(self._scene)

        self._window.switch_to()
        self._window.dispatch_events()
        self._window.dispatch_event("on_draw")

        if mode == "human":
            self._window.flip()
        elif mode == "rgb_array":
            color_buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = color_buffer.get_image_data()
            data = np.fromstring(image_data.data, dtype=np.uint8, sep="")
            data = data.reshape(color_buffer.height, color_buffer.width, 4)
            data = data[::-1, :, 0:3]
            return data


class GymEnvironment(BaseGymEnvironment):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50,
    }
    worlds = None

    def __init__(self, obs_type="image", frameskip=4, world="baseline"):
        self._env = PhysicalEnvironment(world=self.worlds[world])
        self._world = world
        self.action_space = spaces.Discrete(self._env.world.n_actions)

        if obs_type == "image":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self._env.height, self._env.width, 3)
            )
        else:
            raise error.Error("Unrecognized observation type: {}".format(obs_type))

        self._obs_type = obs_type
        self._frameskip = frameskip

    def __getstate__(self):
        return {
            "obs_type": self._obs_type,
            "frameskip": self._frameskip,
            "world": self._world,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def _get_observation(self):
        if self._obs_type == "image":
            return self._env.render(mode="rgb_array")

        raise NotImplementedError

    def _reset(self):
        self._env.reset()
        self._env.step()
        return self._get_observation()

    def _seed(self, seed=None):
        seed = self._env.seed(seed)
        return [seed]

    def _step(self, action):
        score = self._env.world.score
        self._env.act(action)
        for _ in range(self._frameskip):
            terminal = self._env.step()
            if terminal:
                break

        observation = self._get_observation()
        reward = self._env.world.score - score
        info = {"lives": self._env.world.lives}
        return observation, reward, terminal, info

    def _render(self, mode="human", close=False):
        if close:
            return
        return self._env.render(mode)

    def get_action_meanings(self):
        return []

    @property
    def lives(self):
        return self._env.world.lives

    @property
    def parameters(self):
        parameters = super(GymEnvironment, self).parameters
        parameters.update(self._env.world.parameters)
        return parameters
