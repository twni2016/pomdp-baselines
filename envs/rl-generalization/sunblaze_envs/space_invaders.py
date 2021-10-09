import warnings

import Box2D as box_2d
import numpy as np

from .physical_world import PhysicalObject, PhysicalWorld, GymEnvironment


class Missile(PhysicalObject):
    """Missile."""

    def __init__(self, *args, **kwargs):
        super(Missile, self).__init__("missile.png", *args, **kwargs)

    def create_physical_entity(self):
        body = self._engine.CreateDynamicBody(
            position=self.physical_position, fixedRotation=True
        )
        body.CreatePolygonFixture(
            box=(
                (self.width / 2.0) / self._world.physical_scale,
                (self.height / 2.0) / self._world.physical_scale,
            ),
            density=1.0,
            friction=0.0,
            restitution=0.0,
        )

        # Constrain missile movements to Y axis.
        joint = box_2d.b2PrismaticJointDef()
        joint.Initialize(body, self._world.ground, body.worldCenter, (0.0, 1.0))
        joint.collideConnected = True
        self._engine.CreateJoint(joint)

        return body

    @classmethod
    def fire(cls, world, entity, impulse):
        """Fires a missile."""
        raise NotImplementedError


class InvaderMissile(Missile):
    """Invader's missile."""

    @classmethod
    def fire(cls, world, entity, impulse):
        """Fires a missile."""
        missile = cls(
            world=world,
            position=(
                entity.position[0],
                entity.position[1] - entity.height - 10,
            ),
        )
        missile.apply_impulse(
            (0, (-impulse / world.physical_scale) * missile.body.mass)
        )
        return missile

    def should_collide(self, other):
        """Ignore collisions with invaders and own missiles."""
        return not isinstance(other, (Invader, InvaderMissile))

    def on_contact(self, other):
        # Destroy invaders and invader missiles.
        if isinstance(other, PlayerShip):
            self._world._lives -= 1

        self.kill()


class PlayerMissile(Missile):
    """Player's missile."""

    @classmethod
    def fire(cls, world, entity, impulse):
        """Fires a missile."""
        missile = cls(
            world=world,
            position=(
                entity.position[0],
                entity.position[1] + entity.height + 10,
            ),
        )
        missile.apply_impulse((0, (impulse / world.physical_scale) * missile.body.mass))
        return missile

    def should_collide(self, other):
        """Ignore collisions with own missiles."""
        return not isinstance(other, PlayerMissile)

    def on_contact(self, other):
        # Destroy invaders and invader missiles.
        if isinstance(other, (Invader, InvaderMissile)):
            other.kill()

        # Increase player score for hitting invaders.
        if isinstance(other, Invader):
            self._world.add_kill_score()

        self.kill()


class Invader(PhysicalObject):
    """Invader."""

    # Types of invaders.
    TYPE_1 = "invader_1"
    TYPE_2 = "invader_2"
    TYPE_3 = "invader_3"

    def __init__(self, *args, **kwargs):
        self._type = kwargs.pop("invader_type")

        kwargs.setdefault("color", (0, 255, 0))
        kwargs.setdefault("scale", 1)
        super(Invader, self).__init__("{}.png".format(self._type), *args, **kwargs)

    def create_physical_entity(self):
        body = self._engine.CreateStaticBody(position=self.physical_position)
        body.CreatePolygonFixture(
            box=(
                (self.width / 2.0) / self._world.physical_scale,
                (self.height / 2.0) / self._world.physical_scale,
            ),
            density=1.0,
            friction=0.0,
            restitution=0.0,
        )

        return body


class LeftRightMovingInvader(Invader):
    """Invader which moves left and right."""

    max_delta_x = 24

    def __init__(self, *args, **kwargs):
        super(LeftRightMovingInvader, self).__init__(*args, **kwargs)

        self._direction = 1
        self._initial_x = self.position[0]

    def step(self):
        if self.position[0] - self._initial_x >= self.max_delta_x:
            self._direction = -1
        elif self.position[0] - self._initial_x <= -self.max_delta_x:
            self._direction = 1

        self.set_body_position((self.position[0] + self._direction, self.position[1]))
        super(LeftRightMovingInvader, self).step()


class CrossScreenMovingInvader(Invader):
    """Invader which moves across the whole screen."""

    def __init__(self, *args, **kwargs):
        super(CrossScreenMovingInvader, self).__init__(*args, **kwargs)

        self._direction = 1

    def step(self):
        if self.position[0] >= self._world._width - self.width:
            self._direction = -1
        elif self.position[0] <= self.width:
            self._direction = 1

        self.set_body_position((self.position[0] + self._direction, self.position[1]))
        super(CrossScreenMovingInvader, self).step()


class Shield(PhysicalObject):
    """Shield for the player."""

    def __init__(self, *args, **kwargs):
        self.health = kwargs.pop("health")

        kwargs.setdefault("color", (255, 240, 0))
        super(Shield, self).__init__("shield.png", *args, **kwargs)

    def create_physical_entity(self):
        body = self._engine.CreateStaticBody(position=self.physical_position)
        body.CreatePolygonFixture(
            box=(
                (self.width / 2.0) / self._world.physical_scale,
                (self.height / 2.0) / self._world.physical_scale,
            ),
            density=1.0,
            friction=0.0,
            restitution=0.0,
        )

        return body

    def on_contact(self, other):
        """Shield loses health if anything touches it."""

        self.health -= 1
        if self.health <= 0:
            self.kill()


class PlayerShip(PhysicalObject):
    """Player ship."""

    def __init__(self, *args, **kwargs):
        super(PlayerShip, self).__init__("ship.png", *args, **kwargs)

    def create_physical_entity(self):
        body = self._engine.CreateDynamicBody(
            position=self.physical_position, linearDamping=0.99, fixedRotation=True
        )
        body.CreatePolygonFixture(
            box=(
                (self.width / 2.0) / self._world.physical_scale,
                (self.height / 2.0) / self._world.physical_scale,
            ),
            density=1.0,
            friction=0.0,
            restitution=0.0,
        )

        # Constrain paddle movements to X axis.
        joint = box_2d.b2PrismaticJointDef()
        joint.Initialize(body, self._world.ground, body.worldCenter, (1.0, 0.0))
        joint.collideConnected = True
        self._engine.CreateJoint(joint)

        return body


class SpaceInvadersWorld(PhysicalWorld):
    missile_class = Missile
    shield_class = Shield
    player_ship_class = PlayerShip
    invader_class = LeftRightMovingInvader

    # Number of actions.
    n_actions = 4

    # Player missile parameters.
    parameters_player_missile = {
        "class": PlayerMissile,
        # Firing rate (in steps).
        "fire_rate": 20,
        # Maximum number of missiles on screen.
        "max_missiles": 2,
        # Missile impulse.
        "missile_impulse": 100,
    }

    # Invader missile parameters.
    parameters_invader_missile = {
        "class": InvaderMissile,
        # Firing rate (in steps).
        "fire_rate": 15,
        # Maximum number of missiles on screen.
        "max_missiles": 10,
        # Missile impulse.
        "missile_impulse": 100,
    }

    # Number of invaders per row.
    invaders_per_row = 11

    def create_world(self, parent):
        # Create world edges.
        p_width = self._width / self.physical_scale
        p_height = self._height / self.physical_scale

        ground = self._engine.CreateStaticBody(position=(0, 0))
        ground.CreateEdgeFixture(vertices=[(0, 0), (0, p_height)])
        ground.CreateEdgeFixture(vertices=[(0, 0), (p_width, 0)])
        ground.CreateEdgeFixture(vertices=[(0, p_height), (p_width, p_height)])
        ground.CreateEdgeFixture(vertices=[(p_width, p_height), (p_width, 0)])
        self._ground = ground

        self.create_invaders()
        self.create_shields()

        self.player_ship = self.player_ship_class(
            world=self, position=self.initial_player_ship_position()
        )
        parent.add(self.player_ship)

    def create_shields(self):
        """Create protective shields."""

        for config in self.initial_shield_configuration():
            shield = self.shield_class(world=self, **config)
            self._batch.add(shield)

    def create_invaders(self):
        """Create invader grid."""

        offset_x = 80
        offset_y = self.initial_invader_row()

        for row, invader_type in enumerate(self.initial_invader_configuration()):
            for column in range(self.invaders_per_row):
                invader = self.invader_class(
                    world=self, position=(offset_x, offset_y), invader_type=invader_type
                )
                self._batch.add(invader)

                offset_x += 48

            offset_x = 80
            offset_y -= invader.height * 2

    def fire_missile(self, entity, parameters):
        # Check if there are not too many missiles on screen already.
        def count_missiles(node):
            if not isinstance(node, parameters["class"]):
                return

            return 1

        if sum(self.walk(count_missiles)) >= parameters["max_missiles"]:
            return

        # Enforce firing rate.
        last_fire_step = self._last_fire_step.get(parameters["class"], 0)
        if self._step - last_fire_step <= parameters["fire_rate"]:
            return
        self._last_fire_step[parameters["class"]] = self._step

        # Fire missile.
        missile = parameters["class"].fire(
            world=self, entity=entity, impulse=parameters["missile_impulse"]
        )
        self._batch.add(missile)

    @property
    def lives(self):
        return self._lives

    @property
    def score(self):
        return self._score

    @property
    def parameters(self):
        parameters = super(SpaceInvadersWorld, self).parameters
        parameters.update(
            {
                "world": "space_invaders",
            }
        )
        return parameters

    def ship_impulse(self):
        """Relative paddle impulse strength on movement actions."""
        return 50

    def act(self, action):
        """Perform external action."""
        if action == 0:
            # Do nothing.
            pass
        elif action == 1:
            # Move player left.
            self.player_ship.apply_impulse(
                (
                    (-self.ship_impulse() / self.physical_scale)
                    * self.player_ship.body.mass,
                    0,
                )
            )
        elif action == 2:
            # Move player right.
            self.player_ship.apply_impulse(
                (
                    (self.ship_impulse() / self.physical_scale)
                    * self.player_ship.body.mass,
                    0,
                )
            )
        elif action == 3:
            # Fire missile.
            self.fire_missile(self.player_ship, self.parameters_player_missile)

    def initial_shield_configuration(self):
        return [
            {"health": 20, "position": (self._width // 4, 200)},
            {"health": 20, "position": (2 * self._width // 4, 200)},
            {"health": 20, "position": (3 * self._width // 4, 200)},
        ]

    def initial_invader_row(self):
        return self._height - 50

    def initial_invader_configuration(self):
        return [
            Invader.TYPE_1,
            Invader.TYPE_2,
            Invader.TYPE_2,
            Invader.TYPE_3,
            Invader.TYPE_3,
        ]

    def initial_player_ship_position(self):
        """Initial player ship position after reset."""
        return (self._width / 2, 25)

    def adjust_invader_missiles(self, n_invaders):
        """Adjust invader missile inventory."""
        if n_invaders >= 45:
            missiles = 10
        elif n_invaders >= 40:
            missiles = 9
        elif n_invaders >= 35:
            missiles = 8
        elif n_invaders >= 30:
            missiles = 7
        elif n_invaders >= 25:
            missiles = 6
        else:
            missiles = 5

        self.parameters_invader_missile["max_missiles"] = missiles

    def add_kill_score(self):
        """Add score when an invader is killed."""
        self._score += 1

    def reset_world(self):
        """Reset the game."""
        super(SpaceInvadersWorld, self).reset_world()

        self._lives = 3
        self._score = 0
        self._step = 0
        self._last_fire_step = {}

        # Destroy all non-player entities.
        def remove_nodes(node):
            if isinstance(node, (Missile, Invader, Shield)):
                node.kill()

        self.walk(remove_nodes)

        self.create_invaders()
        self.create_shields()

        self.player_ship.kill()
        self.player_ship = self.player_ship_class(
            world=self, position=self.initial_player_ship_position()
        )
        self._batch.add(self.player_ship)

    def step(self):
        """Perform one environment update step."""
        if self._lives <= 0:
            self.reset_world()

        self._terminal = False
        self._step += 1

        # Pick a random invader and make it fire.
        def collect_invaders(node):
            if isinstance(node, self.invader_class):
                return node

        invaders = self.walk(collect_invaders)
        n_invaders = len(invaders)
        if invaders:
            invader = invaders[self.np_random.randint(0, n_invaders)]
            self.fire_missile(invader, self.parameters_invader_missile)

        # Adjust invader missile inventory.
        self.adjust_invader_missiles(n_invaders)

        super(SpaceInvadersWorld, self).step()

        # Check if the player is out of lives or there are no invaders.
        if self._lives <= 0 or not n_invaders:
            self._terminal = True


class SingleLineSpaceInvadersWorld(SpaceInvadersWorld):
    def initial_invader_row(self):
        return self._height - 50

    def initial_invader_configuration(self):
        return [
            Invader.TYPE_1,
        ]

    def adjust_invader_missiles(self, n_invaders):
        """Adjust invader missile inventory."""
        pass

    def add_kill_score(self):
        """Add score when an invader is killed."""
        self._score += 5


class InfiniteShieldsSpaceInvadersWorld(SpaceInvadersWorld):
    def initial_shield_configuration(self):
        return [
            {"health": np.inf, "position": (self._width // 4, 200)},
            {"health": np.inf, "position": (2 * self._width // 4, 200)},
            {"health": np.inf, "position": (3 * self._width // 4, 200)},
        ]


class OffsetPlayerSpaceInvadersWorld(SpaceInvadersWorld):
    def initial_shield_configuration(self):
        return [
            {"health": 20, "position": (self._width // 4, 200)},
            {"health": 20, "position": (2 * self._width // 4, 200)},
            {"health": 20, "position": (3 * self._width // 4, 200)},
        ]

    def initial_player_ship_position(self):
        """Initial player ship position after reset."""
        return (self._width / 2, 100)


class OffsetPlayer150SpaceInvadersWorld(SpaceInvadersWorld):
    def initial_shield_configuration(self):
        return [
            {"health": 20, "position": (self._width // 4, 200)},
            {"health": 20, "position": (2 * self._width // 4, 200)},
            {"health": 20, "position": (3 * self._width // 4, 200)},
        ]

    def initial_player_ship_position(self):
        """Initial player ship position after reset."""
        return (self._width / 2, 150)


class RandomOffsetPlayerSpaceInvadersWorld(SpaceInvadersWorld):
    offset_range_start = 25
    offset_range_end = 125

    def initial_shield_configuration(self):
        return [
            {"health": 20, "position": (self._width // 4, 200)},
            {"health": 20, "position": (2 * self._width // 4, 200)},
            {"health": 20, "position": (3 * self._width // 4, 200)},
        ]

    def initial_player_ship_position(self):
        """Initial player ship position after reset."""
        self._player_offset = int(
            self.np_random.uniform(self.offset_range_start, self.offset_range_end)
        )
        return (self._width / 2, self._player_offset)

    @property
    def parameters(self):
        parameters = super(RandomOffsetPlayerSpaceInvadersWorld, self).parameters
        parameters.update(
            {
                "player_offset": self._player_offset,
            }
        )
        return parameters


class OffsetPlayerSetASpaceInvadersWorld(RandomOffsetPlayerSpaceInvadersWorld):
    offset_range_start = 25
    offset_range_end = 75


class OffsetPlayerSetBSpaceInvadersWorld(RandomOffsetPlayerSpaceInvadersWorld):
    offset_range_start = 75
    offset_range_end = 125


class SideObstacle(PhysicalObject):
    """Side obstacle object."""

    def __init__(self, *args, **kwargs):
        kwargs["color"] = (80, 80, 80)
        super(SideObstacle, self).__init__("side_obstacle.png", *args, **kwargs)

    def create_physical_entity(self):
        body = self._engine.CreateStaticBody(position=self.physical_position)
        body.CreatePolygonFixture(
            box=(
                (self.width / 2.0) / self._world.physical_scale,
                (self.height / 2.0) / self._world.physical_scale,
            ),
            density=10.0,
            friction=0.0,
            restitution=0.0,
        )

        return body


class SideObstacleSpaceInvadersWorld(SpaceInvadersWorld):
    def create_world(self, parent):
        super(SideObstacleSpaceInvadersWorld, self).create_world(parent)

        self.obstacle1 = SideObstacle(world=self, position=(10, self._height / 2))
        parent.add(self.obstacle1, z=1)

        self.obstacle2 = SideObstacle(
            world=self, position=(self._width - 10, self._height / 2)
        )
        parent.add(self.obstacle2, z=1)


class LeftSideObstacleSpaceInvadersWorld(SpaceInvadersWorld):
    def create_world(self, parent):
        super(LeftSideObstacleSpaceInvadersWorld, self).create_world(parent)

        self.obstacle = SideObstacle(world=self, position=(10, self._height / 2))
        parent.add(self.obstacle, z=1)


class RightSideObstacleSpaceInvadersWorld(SpaceInvadersWorld):
    def create_world(self, parent):
        super(RightSideObstacleSpaceInvadersWorld, self).create_world(parent)

        self.obstacle = SideObstacle(
            world=self, position=(self._width - 10, self._height / 2)
        )
        parent.add(self.obstacle, z=1)


class RandomSideObstacleSpaceInvadersWorld(SpaceInvadersWorld):
    def reset_world(self):
        super(RandomSideObstacleSpaceInvadersWorld, self).reset_world()

        self.reset_obstacle()

    def reset_obstacle(self):
        """Reset obstacle width and position."""
        if hasattr(self, "obstacle"):
            self.obstacle.kill()

        side = self.np_random.choice(["left", "right"])
        width = int(self.np_random.uniform(-8, 2))

        if side == "left":
            x = width
        elif side == "right":
            x = self._width - width

        self.obstacle = SideObstacle(world=self, position=(x, self._height / 2))
        self._batch.add(self.obstacle, z=1)


class SingleInvaderSpaceInvadersWorld(SpaceInvadersWorld):
    invader_class = CrossScreenMovingInvader
    invaders_per_row = 1
    parameters_invader_missile = {
        "class": InvaderMissile,
        # Firing rate (in steps).
        "fire_rate": 10,
        # Maximum number of missiles on screen.
        "max_missiles": 20,
        # Missile impulse.
        "missile_impulse": 100,
    }

    def initial_invader_row(self):
        return self._height - 200

    def initial_invader_configuration(self):
        return [
            Invader.TYPE_1,
        ]

    def adjust_invader_missiles(self, n_invaders):
        """Adjust invader missile inventory."""
        pass

    def add_kill_score(self):
        """Add score when an invader is killed."""
        self._score += 55

    def initial_shield_configuration(self):
        return [
            {"health": np.inf, "position": (4 * self._width // 5, 200)},
        ]


class WhiteShield(Shield):
    """White shield for the player."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("color", (255, 255, 255))
        super(WhiteShield, self).__init__(*args, **kwargs)


class WhiteLeftRightMovingInvader(LeftRightMovingInvader):
    """White invader which moves left and right."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("color", (255, 255, 255))
        super(WhiteLeftRightMovingInvader, self).__init__(*args, **kwargs)


class OneColorSpaceInvadersWorld(SpaceInvadersWorld):
    shield_class = WhiteShield
    invader_class = WhiteLeftRightMovingInvader


class Scaled80SpaceInvadersWorld(SpaceInvadersWorld):
    def __init__(self, *args, **kwargs):
        super(Scaled80SpaceInvadersWorld, self).__init__(*args, **kwargs)
        self.scale = 0.80


class Scaled90SpaceInvadersWorld(SpaceInvadersWorld):
    def __init__(self, *args, **kwargs):
        super(Scaled90SpaceInvadersWorld, self).__init__(*args, **kwargs)
        self.scale = 0.90


class Scaled95SpaceInvadersWorld(SpaceInvadersWorld):
    def __init__(self, *args, **kwargs):
        super(Scaled95SpaceInvadersWorld, self).__init__(*args, **kwargs)
        self.scale = 0.95


class Scaled99SpaceInvadersWorld(SpaceInvadersWorld):
    def __init__(self, *args, **kwargs):
        super(Scaled99SpaceInvadersWorld, self).__init__(*args, **kwargs)
        self.scale = 0.99


class RandomScaledSpaceInvadersWorld(SpaceInvadersWorld):
    scale_range_start = 0.90
    scale_range_end = 1.0

    def reset_world(self):
        super(RandomScaledSpaceInvadersWorld, self).reset_world()
        self.scale = self.np_random.uniform(
            self.scale_range_start, self.scale_range_end
        )

    @property
    def parameters(self):
        parameters = super(RandomScaledSpaceInvadersWorld, self).parameters
        parameters.update(
            {
                "scale": self.scale,
            }
        )
        return parameters


class ScaledSetASpaceInvadersWorld(RandomScaledSpaceInvadersWorld):
    scale_range_start = 0.95
    scale_range_end = 1.0


class ScaledSetBSpaceInvadersWorld(RandomScaledSpaceInvadersWorld):
    scale_range_start = 0.90
    scale_range_end = 0.95


class RandomActionStrengthSpaceInvadersWorld(SpaceInvadersWorld):
    impulse_range_start = 30
    impulse_range_end = 170

    def reset_world(self):
        super(RandomActionStrengthSpaceInvadersWorld, self).reset_world()
        self._impulse_strength = self.np_random.uniform(
            self.impulse_range_start, self.impulse_range_end
        )

    def ship_impulse(self):
        return self._impulse_strength

    @property
    def parameters(self):
        parameters = super(RandomActionStrengthSpaceInvadersWorld, self).parameters
        parameters.update(
            {
                "ship_impulse": self._impulse_strength,
            }
        )
        return parameters


class ActionStrengthSetASpaceInvadersWorld(RandomActionStrengthSpaceInvadersWorld):
    impulse_range_start = 30
    impulse_range_end = 100


class ActionStrengthSetBSpaceInvadersWorld(RandomActionStrengthSpaceInvadersWorld):
    impulse_range_start = 100
    impulse_range_end = 170


class MultiParameterSetASpaceInvadersWorld(
    OffsetPlayerSetASpaceInvadersWorld,
    ActionStrengthSetASpaceInvadersWorld,
    # ScaledSetASpaceInvadersWorld,
):
    """
    Parameters (all from set A):
      - player offset
      - action strength
      - scale (NOTE: removed)
    """

    pass


class MultiParameterSetBSpaceInvadersWorld(
    OffsetPlayerSetBSpaceInvadersWorld,
    ActionStrengthSetBSpaceInvadersWorld,
    # ScaledSetBSpaceInvadersWorld,
):
    """
    Parameters (all from set B):
      - player offset
      - action strength
      - scale (NOTE: removed)
    """

    pass


class SpaceInvaders(GymEnvironment):
    """Space invaders Gym environment."""

    worlds = {
        "baseline": SpaceInvadersWorld,
        "single_line": SingleLineSpaceInvadersWorld,
        "inf_shields": InfiniteShieldsSpaceInvadersWorld,
        "offset_player": OffsetPlayerSpaceInvadersWorld,
        "offset_player150": OffsetPlayer150SpaceInvadersWorld,
        "random_offset_player": RandomOffsetPlayerSpaceInvadersWorld,
        "side_obstacle": SideObstacleSpaceInvadersWorld,
        "left_side_obstacle": LeftSideObstacleSpaceInvadersWorld,
        "right_side_obstacle": RightSideObstacleSpaceInvadersWorld,
        "random_side_obstacle": RandomSideObstacleSpaceInvadersWorld,
        "single_invader": SingleInvaderSpaceInvadersWorld,
        "one_color": OneColorSpaceInvadersWorld,
        "scaled_80": Scaled80SpaceInvadersWorld,
        "scaled_90": Scaled90SpaceInvadersWorld,
        "scaled_95": Scaled95SpaceInvadersWorld,
        "scaled_99": Scaled99SpaceInvadersWorld,
        "random_scaled": RandomScaledSpaceInvadersWorld,
        "offset_player_set_a": OffsetPlayerSetASpaceInvadersWorld,
        "offset_player_set_b": OffsetPlayerSetBSpaceInvadersWorld,
        "scaled_set_a": ScaledSetASpaceInvadersWorld,
        "scaled_set_b": ScaledSetBSpaceInvadersWorld,
        "action_strength_set_a": ActionStrengthSetASpaceInvadersWorld,
        "action_strength_set_b": ActionStrengthSetBSpaceInvadersWorld,
        "multi_parameter_set_a": MultiParameterSetASpaceInvadersWorld,
        "multi_parameter_set_b": MultiParameterSetBSpaceInvadersWorld,
    }

    def get_action_meanings(self):
        return [
            "NOOP",
            "LEFT",
            "RIGHT",
            "FIRE",
        ]

    def get_keys_to_action(self):
        return {
            (): 0,
            (ord("a"),): 1,
            (ord("d"),): 2,
            (ord("s"),): 3,
            (ord("a"), ord("d")): 0,
        }
