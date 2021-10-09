import warnings

import Box2D as box_2d
import numpy as np
import pyglet

from .physical_world import PhysicalObject, PhysicalWorld, GymEnvironment


class Ball(PhysicalObject):
    """Ball object."""

    asset = "ball.png"
    max_speed = 9.0

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("scale", 0.25)
        kwargs.setdefault("color", (208, 33, 82))
        super(Ball, self).__init__(self.asset, *args, **kwargs)

    def create_physical_entity(self):
        body = self._engine.CreateDynamicBody(
            position=self.physical_position, fixedRotation=True
        )
        body.CreateCircleFixture(
            radius=(self.width / 2) / self._world.physical_scale,
            density=1.0,
            friction=0.0,
            restitution=1.0,
        )
        return body

    def step(self):
        super(Ball, self).step()

        speed = self._body.linearVelocity.length
        if speed > self.max_speed:
            self._body.linearDamping = 0.5
        elif speed < self.max_speed:
            self._body.linearDamping = 0.0

    def on_contact(self, other):
        """Prevent the ball from bouncing in a straight line up and down."""
        velocity_x = self.body.linearVelocity[0]
        if abs(velocity_x) < 0.1:
            self.apply_impulse([self._world.np_random.uniform(-0.1, 0.1), 0.0])


class Paddle(PhysicalObject):
    """Paddle object."""

    asset = "paddle.png"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("color", (255, 168, 0))
        super(Paddle, self).__init__(self.asset, *args, **kwargs)

    def create_physical_entity(self):
        body = self._engine.CreateDynamicBody(
            position=self.physical_position,
            angle=self.physical_rotation,
            linearDamping=0.99,
            fixedRotation=True,
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


class Brick(PhysicalObject):
    """Brick object."""

    def __init__(self, *args, **kwargs):
        self.row = kwargs.pop("row")
        self.column = kwargs.pop("column")
        kwargs["color"] = self.get_color()
        super(Brick, self).__init__("brick.png", *args, **kwargs)

    def get_color(self):
        """Brick color."""
        colors = {
            0: (255, 0, 0),
            1: (255, 174, 0),
            2: (252, 255, 0),
            3: (0, 255, 0),
            4: (0, 0, 255),
        }
        return colors.get(self.row, (0, 0, 0))

    def get_score(self):
        """Score if the brick is destroyed."""
        scores = {
            0: 10,
            1: 7,
            2: 5,
            3: 3,
            4: 1,
        }
        return scores.get(self.row, 0)

    def get_restitution(self):
        restitution = {
            0: 1.5,
            1: 1.3,
            2: 1.2,
            3: 1.15,
            4: 1.1,
        }
        return restitution.get(self.row, 1.0)

    def create_physical_entity(self):
        body = self._engine.CreateStaticBody(position=self.physical_position)
        body.CreatePolygonFixture(
            box=(
                (self.width / 2.0) / self._world.physical_scale,
                (self.height / 2.0) / self._world.physical_scale,
            ),
            density=1.0,
            friction=0.0,
            restitution=self.get_restitution(),
        )

        return body

    def on_contact(self, other):
        """Destroy the brick on contact with the ball."""
        if not isinstance(other, Ball):
            return

        self.kill()

        ball_velocity_x = other.body.linearVelocity[0]
        if abs(ball_velocity_x) < 0.2:
            other.apply_impulse([0.2 * np.sign(ball_velocity_x), 0.0])

        # Increase player score.
        self._world._score += self.get_score()


class BreakoutWorld(PhysicalWorld):
    # Class used for representing the paddle.
    paddle_class = Paddle
    # Class used for representing bricks.
    brick_class = Brick
    # Class used for representing the ball.
    ball_class = Ball

    # Number of actions.
    n_actions = 3

    def create_world(self, parent):
        # Create world edges.
        p_width = self._width / self.physical_scale
        p_height = self._height / self.physical_scale

        ground = self._engine.CreateStaticBody(position=(0, 0))
        ground.CreateEdgeFixture(vertices=[(0, 0), (0, p_height)])
        ground.CreateEdgeFixture(vertices=[(0, p_height), (p_width, p_height)])
        ground.CreateEdgeFixture(vertices=[(p_width, p_height), (p_width, 0)])
        self._ground = ground

        # Create ball.
        self.ball = self.ball_class(world=self, position=self.initial_ball_position())
        parent.add(self.ball)

        # Create paddle.
        self.paddle = self.paddle_class(
            world=self, position=self.initial_paddle_position()
        )
        parent.add(self.paddle)

    @property
    def lives(self):
        return self._lives

    @property
    def score(self):
        return self._score

    @property
    def parameters(self):
        parameters = super(BreakoutWorld, self).parameters
        parameters.update(
            {
                "world": "breakout",
            }
        )
        return parameters

    def paddle_impulse(self):
        """Relative paddle impulse strength on movement actions."""
        return 50

    def act(self, action):
        """Perform external action."""
        if action == 0:
            # Do nothing.
            pass
        elif action == 1:
            # Move paddle left.
            self.paddle.apply_impulse(
                (
                    (-self.paddle_impulse() / self.physical_scale)
                    * self.paddle.body.mass,
                    0,
                )
            )
        elif action == 2:
            # Move paddle right.
            self.paddle.apply_impulse(
                (
                    (self.paddle_impulse() / self.physical_scale)
                    * self.paddle.body.mass,
                    0,
                )
            )

    def initial_ball_position(self):
        """Initial ball position after reset."""
        return (self._width / 2, self._height / 2)

    def initial_paddle_position(self):
        """Initial paddle position after reset."""
        return (self._width / 2, 25)

    def initial_paddle_rotation(self):
        """Initial paddle rotation after reset (in degrees)."""
        return 0

    def initial_brick_position(self):
        """Initial brick row offset after reset."""
        return 40

    def create_bricks(self):
        """Create bricks."""
        dummy = self.brick_class(row=0, column=0)

        brick_x = dummy.width / 2
        brick_y = self._height - self.initial_brick_position()

        for row in range(5):
            for column in range(self._width // dummy.width):
                brick = self.brick_class(
                    world=self, position=(brick_x, brick_y), row=row, column=column
                )
                self._batch.add(brick)

                brick_x += dummy.width

            brick_x = dummy.width / 2
            brick_y -= dummy.height

    def reset_world(self):
        """Reset the game."""
        super(BreakoutWorld, self).reset_world()

        self._lives = 5
        self._score = 0

        # Clear any bricks still left and re-create them.
        def remove_bricks(node):
            if not isinstance(node, self.brick_class):
                return

            node.kill()

        self.walk(remove_bricks)
        self.create_bricks()

        self.reset_paddle()
        self.reset_ball()

    def reset_paddle(self):
        """Reset paddle."""
        self.paddle.kill()
        self.paddle = self.paddle_class(
            world=self,
            position=self.initial_paddle_position(),
            rotation=self.initial_paddle_rotation(),
        )
        self._batch.add(self.paddle)

    def reset_ball(self):
        """Reset ball position."""
        self.ball.stop_body()
        self.ball.set_body_position(self.initial_ball_position())
        self.ball.apply_impulse(
            (150.0 / self.physical_scale)
            * self.ball.body.mass
            * np.asarray(
                [
                    self.np_random.uniform(-0.3, 0.3),
                    -self.np_random.uniform(0.6, 1.0),
                ]
            )
        )

    def step(self):
        """Perform one environment update step."""
        if self._lives <= 0:
            self.reset_world()

        self._terminal = False

        super(BreakoutWorld, self).step()

        # Check if ball is outside the screen.
        if self.ball.position[1] < 0:
            self.reset_ball()
            self._lives -= 1
            if self._lives <= 0:
                self._terminal = True

        # Check if there are no more bricks.
        def count_bricks(node):
            if not isinstance(node, self.brick_class):
                return

            return 1

        if not sum(self.walk(count_bricks)):
            # No more bricks, the game is finished.
            self._terminal = True


class OffsetPaddleBreakoutWorld(BreakoutWorld):
    paddle_offset = 100

    def initial_paddle_position(self):
        """Initial paddle position after reset."""
        return (self._width / 2, self.paddle_offset)

    @property
    def parameters(self):
        parameters = super(OffsetPaddleBreakoutWorld, self).parameters
        parameters.update(
            {
                "paddle_offset": self.paddle_offset,
            }
        )
        return parameters


class OffsetPaddle50BreakoutWorld(OffsetPaddleBreakoutWorld):
    paddle_offset = 50


class OffsetPaddle75BreakoutWorld(OffsetPaddleBreakoutWorld):
    paddle_offset = 75


class OffsetPaddle100BreakoutWorld(OffsetPaddleBreakoutWorld):
    paddle_offset = 100


class OffsetPaddle125BreakoutWorld(OffsetPaddleBreakoutWorld):
    paddle_offset = 125


class OffsetPaddle150BreakoutWorld(OffsetPaddleBreakoutWorld):
    paddle_offset = 150


class RandomOffsetPaddleBreakoutWorld(BreakoutWorld):
    offset_range_start = 25
    offset_range_end = 110

    def initial_paddle_position(self):
        """Initial paddle position after reset."""
        self._paddle_offset = int(
            self.np_random.uniform(self.offset_range_start, self.offset_range_end)
        )
        return (self._width / 2, self._paddle_offset)

    @property
    def parameters(self):
        parameters = super(RandomOffsetPaddleBreakoutWorld, self).parameters
        parameters.update(
            {
                "paddle_offset": self._paddle_offset,
            }
        )
        return parameters


class OffsetPaddleSetABreakoutWorld(RandomOffsetPaddleBreakoutWorld):
    warnings.warn(
        "This env. parameter was dropped and should no longer be used.",
        DeprecationWarning,
    )
    offset_range_start = 25
    offset_range_end = 75


class OffsetPaddleSetBBreakoutWorld(RandomOffsetPaddleBreakoutWorld):
    warnings.warn(
        "This env. parameter was dropped and should no longer be used.",
        DeprecationWarning,
    )
    offset_range_start = 20
    offset_range_end = 125


class VisuallyFixedOffsetPaddle(Paddle):
    """Paddle, which is visually at fixed offset, but physically at some other offset."""

    @property
    def visual_position(self):
        """Return visual object position."""
        position = super(VisuallyFixedOffsetPaddle, self).visual_position
        return (position[0], 25)


class PhysicallyOffsetPaddle125BreakoutWorld(OffsetPaddleBreakoutWorld):
    paddle_class = VisuallyFixedOffsetPaddle
    paddle_offset = 125


class Obstacle(PhysicalObject):
    """Obstacle object."""

    def __init__(self, *args, **kwargs):
        kwargs["color"] = (80, 80, 80)
        super(Obstacle, self).__init__("obstacle.png", *args, **kwargs)

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


class ObstacleBreakoutWorld(BreakoutWorld):
    def create_world(self, parent):
        super(ObstacleBreakoutWorld, self).create_world(parent)

        self.obstacle = Obstacle(world=self, position=self.obstacle_position())
        parent.add(self.obstacle)

    def obstacle_position(self):
        """Position of the obstacle."""
        return (self._width / 2, 340)


class SideObstacle(PhysicalObject):
    """Side obstacle object."""

    def __init__(self, *args, **kwargs):
        image = pyglet.resource.image("side_obstacle.png")
        width = kwargs.pop("width", None)
        if width is not None:
            image = image.get_region(0, 0, width, image.height)

        kwargs["color"] = (80, 80, 80)
        super(SideObstacle, self).__init__(image, *args, **kwargs)

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


class SideObstacleBreakoutWorld(BreakoutWorld):
    def create_world(self, parent):
        super(SideObstacleBreakoutWorld, self).create_world(parent)

        self.obstacle1 = SideObstacle(world=self, position=(10, self._height / 2))
        parent.add(self.obstacle1, z=1)

        self.obstacle2 = SideObstacle(
            world=self, position=(self._width - 10, self._height / 2)
        )
        parent.add(self.obstacle2, z=1)


class LeftSideObstacleBreakoutWorld(BreakoutWorld):
    def create_world(self, parent):
        super(LeftSideObstacleBreakoutWorld, self).create_world(parent)

        self.obstacle = SideObstacle(world=self, position=(10, self._height / 2))
        parent.add(self.obstacle, z=1)


class RightSideObstacleBreakoutWorld(BreakoutWorld):
    def create_world(self, parent):
        super(RightSideObstacleBreakoutWorld, self).create_world(parent)

        self.obstacle = SideObstacle(
            world=self, position=(self._width - 10, self._height / 2)
        )
        parent.add(self.obstacle, z=1)


class RandomSideObstacleBreakoutWorld(BreakoutWorld):
    side_obstacle_width_range_start = 0
    side_obstacle_width_range_end = 20

    def reset_world(self):
        super(RandomSideObstacleBreakoutWorld, self).reset_world()

        self.reset_obstacle()

    def reset_obstacle(self):
        """Reset obstacle width and position."""
        if hasattr(self, "obstacle"):
            self.obstacle.kill()

        side = self.np_random.choice(["left", "right"])
        width = int(
            self.np_random.uniform(
                self.side_obstacle_width_range_start, self.side_obstacle_width_range_end
            )
        )

        if side == "left":
            x = width / 2
        elif side == "right":
            x = self._width - width / 2

        self.obstacle = SideObstacle(
            world=self, position=(x, self._height / 2), width=width
        )
        self._batch.add(self.obstacle, z=1)

        self._obstacle_side = side
        self._obstacle_width = width

    @property
    def parameters(self):
        parameters = super(RandomSideObstacleBreakoutWorld, self).parameters
        parameters.update(
            {
                "obstacle_side": self._obstacle_side,
                "obstacle_width": self._obstacle_width,
            }
        )
        return parameters


class SideObstacleSetABreakoutWorld(RandomSideObstacleBreakoutWorld):
    side_obstacle_width_range_start = 0
    side_obstacle_width_range_end = 15


class SideObstacleSetBBreakoutWorld(RandomSideObstacleBreakoutWorld):
    side_obstacle_width_range_start = 15
    side_obstacle_width_range_end = 20


class SmallPaddle(Paddle):
    """Small paddle object."""

    asset = "small_paddle.png"


class SmallPaddleBreakoutWorld(BreakoutWorld):
    paddle_class = SmallPaddle


class Small10Paddle(Paddle):
    """10% smaller paddle object."""

    asset = "small10_paddle.png"


class Small10PaddleBreakoutWorld(BreakoutWorld):
    paddle_class = Small10Paddle


class Small20Paddle(Paddle):
    """20% smaller paddle object."""

    asset = "small20_paddle.png"


class Small20PaddleBreakoutWorld(BreakoutWorld):
    paddle_class = Small20Paddle


class Small30Paddle(Paddle):
    """30% smaller paddle object."""

    asset = "small30_paddle.png"


class Small30PaddleBreakoutWorld(BreakoutWorld):
    paddle_class = Small30Paddle


class RandomSmallPaddleBreakoutWorld(BreakoutWorld):
    @property
    def paddle_class(self):
        return self.np_random.choice(
            [Paddle, Small10Paddle, Small20Paddle, Small30Paddle]
        )


class BigBall(Ball):
    """A bigger ball object."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("scale", 0.5)
        super(BigBall, self).__init__(*args, **kwargs)


class BigBallBreakoutWorld(BreakoutWorld):
    ball_class = BigBall


class HugeBall(Ball):
    """A huge ball object."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("scale", 0.75)
        super(HugeBall, self).__init__(*args, **kwargs)


class HugeBallBreakoutWorld(BreakoutWorld):
    ball_class = HugeBall


class SquareBall(Ball):
    """A square ball object."""

    asset = "square.png"

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
            restitution=1.0,
        )
        return body


class SquareBallBreakoutWorld(BreakoutWorld):
    ball_class = SquareBall


class WhiteBall(Ball):
    """White ball object."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("color", (255, 255, 255))
        super(WhiteBall, self).__init__(*args, **kwargs)


class WhitePaddle(Paddle):
    """White paddle object."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("color", (255, 255, 255))
        super(WhitePaddle, self).__init__(*args, **kwargs)


class WhiteBrick(Brick):
    """White brick object."""

    def get_color(self):
        """Brick color."""
        return (255, 255, 255)


class OneColorBreakoutWorld(BreakoutWorld):
    ball_class = WhiteBall
    paddle_class = WhitePaddle
    brick_class = WhiteBrick


class Scaled80BreakoutWorld(BreakoutWorld):
    def __init__(self, *args, **kwargs):
        super(Scaled80BreakoutWorld, self).__init__(*args, **kwargs)
        self.scale = 0.80


class Scaled90BreakoutWorld(BreakoutWorld):
    def __init__(self, *args, **kwargs):
        super(Scaled90BreakoutWorld, self).__init__(*args, **kwargs)
        self.scale = 0.90


class Scaled95BreakoutWorld(BreakoutWorld):
    def __init__(self, *args, **kwargs):
        super(Scaled95BreakoutWorld, self).__init__(*args, **kwargs)
        self.scale = 0.95


class Scaled99BreakoutWorld(BreakoutWorld):
    def __init__(self, *args, **kwargs):
        super(Scaled99BreakoutWorld, self).__init__(*args, **kwargs)
        self.scale = 0.99


class RandomScaledBreakoutWorld(BreakoutWorld):
    scale_range_start = 0.95
    scale_range_end = 1.0

    def reset_world(self):
        super(RandomScaledBreakoutWorld, self).reset_world()
        self.scale = self.np_random.uniform(
            self.scale_range_start, self.scale_range_end
        )

    @property
    def parameters(self):
        parameters = super(RandomScaledBreakoutWorld, self).parameters
        parameters.update(
            {
                "scale": self.scale,
            }
        )
        return parameters


class ScaledSetABreakoutWorld(RandomScaledBreakoutWorld):
    warnings.warn(
        "This env. parameter was dropped and should no longer be used.",
        DeprecationWarning,
    )
    scale_range_start = 0.90
    scale_range_end = 0.95


class ScaledSetBBreakoutWorld(RandomScaledBreakoutWorld):
    warnings.warn(
        "This env. parameter was dropped and should no longer be used.",
        DeprecationWarning,
    )
    scale_range_start = 0.95
    scale_range_end = 1.0


class RandomActionStrengthBreakoutWorld(BreakoutWorld):
    impulse_range_start = 30
    impulse_range_end = 170

    def reset_world(self):
        super(RandomActionStrengthBreakoutWorld, self).reset_world()
        self._impulse_strength = self.np_random.uniform(
            self.impulse_range_start, self.impulse_range_end
        )

    def paddle_impulse(self):
        return self._impulse_strength

    @property
    def parameters(self):
        parameters = super(RandomActionStrengthBreakoutWorld, self).parameters
        parameters.update(
            {
                "impulse_strength": self._impulse_strength,
            }
        )
        return parameters


class ActionStrengthSetABreakoutWorld(RandomActionStrengthBreakoutWorld):
    impulse_range_start = 30
    impulse_range_end = 100


class ActionStrengthSetBBreakoutWorld(RandomActionStrengthBreakoutWorld):
    impulse_range_start = 100
    impulse_range_end = 170


class RandomRotatedPaddleBreakoutWorld(BreakoutWorld):
    rotation_range_start = -90
    rotation_range_end = 90

    def initial_paddle_rotation(self):
        """Initial paddle rotation after reset."""
        self._paddle_rotation = self.np_random.uniform(
            self.rotation_range_start, self.rotation_range_end
        )
        return self._paddle_rotation

    @property
    def parameters(self):
        parameters = super(RandomRotatedPaddleBreakoutWorld, self).parameters
        parameters.update(
            {
                "paddle_rotation": self._paddle_rotation,
            }
        )
        return parameters


class RotatedPaddleSetABreakoutWorld(RandomRotatedPaddleBreakoutWorld):
    warnings.warn(
        "This env. parameter was dropped and should no longer be used.",
        DeprecationWarning,
    )
    rotation_range_start = -15
    rotation_range_end = 15


class RotatedPaddleSetBBreakoutWorld(RandomRotatedPaddleBreakoutWorld):
    warnings.warn(
        "This env. parameter was dropped and should no longer be used.",
        DeprecationWarning,
    )
    rotation_range_start = -25
    rotation_range_end = 25


class RandomOffsetBricksBreakoutWorld(BreakoutWorld):
    brick_offset_range_start = 0
    brick_offset_range_end = 80

    def initial_brick_position(self):
        """Initial brick row offset after reset."""
        self._brick_offset = int(
            self.np_random.uniform(
                self.brick_offset_range_start, self.brick_offset_range_end
            )
        )
        return self._brick_offset

    @property
    def parameters(self):
        parameters = super(RandomOffsetBricksBreakoutWorld, self).parameters
        parameters.update(
            {
                "brick_offset": self._brick_offset,
            }
        )
        return parameters


class OffsetBricksSetABreakoutWorld(RandomOffsetBricksBreakoutWorld):
    brick_offset_range_start = 0
    brick_offset_range_end = 80


class OffsetBricksSetBBreakoutWorld(RandomOffsetBricksBreakoutWorld):
    brick_offset_range_start = 80
    brick_offset_range_end = 100


class MultiParameterSetABreakoutWorld(
    ActionStrengthSetABreakoutWorld,
    OffsetBricksSetABreakoutWorld,
    SideObstacleSetABreakoutWorld,
):
    """
    Parameters (all from set A):
      - action strength
      - offset bricks
      - side obstacle
    """

    pass


class MultiParameterSetBBreakoutWorld(
    ActionStrengthSetBBreakoutWorld,
    OffsetBricksSetBBreakoutWorld,
    SideObstacleSetBBreakoutWorld,
):
    """
    Parameters (all from set B):
      - action strength
      - offset bricks
      - side obstacle
    """

    pass


class MultiParameterSetCBreakoutWorld(
    ActionStrengthSetABreakoutWorld,
    OffsetBricksSetABreakoutWorld,
    SideObstacleSetABreakoutWorld,
):
    """
    Parameters (all from set A):
      - action strength
      - offset bricks
      - side obstacle
    """

    warnings.warn(
        "This env. parameter was dropped and should no longer be used.",
        DeprecationWarning,
    )
    pass


class MultiParameterSetDBreakoutWorld(
    ActionStrengthSetABreakoutWorld, SideObstacleSetABreakoutWorld
):
    """
    Parameters (all from set A):
      - action strength
      - side obstacle
    """

    warnings.warn(
        "This env. parameter was dropped and should no longer be used.",
        DeprecationWarning,
    )
    pass


class Breakout(GymEnvironment):
    """Breakout Gym environment."""

    worlds = {
        "baseline": BreakoutWorld,
        "offset_paddle_50": OffsetPaddle50BreakoutWorld,
        "offset_paddle_75": OffsetPaddle75BreakoutWorld,
        "offset_paddle_100": OffsetPaddle100BreakoutWorld,
        "offset_paddle_125": OffsetPaddle125BreakoutWorld,
        "offset_paddle_150": OffsetPaddle150BreakoutWorld,
        "random_offset_paddle": RandomOffsetPaddleBreakoutWorld,
        "physically_offset_paddle_125": PhysicallyOffsetPaddle125BreakoutWorld,
        "obstacle": ObstacleBreakoutWorld,
        "side_obstacle": SideObstacleBreakoutWorld,
        "left_side_obstacle": LeftSideObstacleBreakoutWorld,
        "right_side_obstacle": RightSideObstacleBreakoutWorld,
        "random_side_obstacle": RandomSideObstacleBreakoutWorld,
        "small_paddle": SmallPaddleBreakoutWorld,
        "small10_paddle": Small10PaddleBreakoutWorld,
        "small20_paddle": Small20PaddleBreakoutWorld,
        "small30_paddle": Small30PaddleBreakoutWorld,
        "random_small_paddle": RandomSmallPaddleBreakoutWorld,
        "big_ball": BigBallBreakoutWorld,
        "huge_ball": HugeBallBreakoutWorld,
        "square_ball": SquareBallBreakoutWorld,
        "one_color": OneColorBreakoutWorld,
        "scaled_80": Scaled80BreakoutWorld,
        "scaled_90": Scaled90BreakoutWorld,
        "scaled_95": Scaled95BreakoutWorld,
        "scaled_99": Scaled99BreakoutWorld,
        "random_scaled": RandomScaledBreakoutWorld,
        "offset_paddle_set_a": OffsetPaddleSetABreakoutWorld,
        "offset_paddle_set_b": OffsetPaddleSetBBreakoutWorld,
        "rotated_paddle_set_a": RotatedPaddleSetABreakoutWorld,
        "rotated_paddle_set_b": RotatedPaddleSetBBreakoutWorld,
        "offset_bricks_set_a": OffsetBricksSetABreakoutWorld,
        "offset_bricks_set_b": OffsetBricksSetBBreakoutWorld,
        "scaled_set_a": ScaledSetABreakoutWorld,
        "scaled_set_b": ScaledSetBBreakoutWorld,
        "action_strength_set_a": ActionStrengthSetABreakoutWorld,
        "action_strength_set_b": ActionStrengthSetBBreakoutWorld,
        "side_obstacle_set_a": SideObstacleSetABreakoutWorld,
        "side_obstacle_set_b": SideObstacleSetBBreakoutWorld,
        "multi_parameter_set_a": MultiParameterSetABreakoutWorld,
        "multi_parameter_set_b": MultiParameterSetBBreakoutWorld,
        "multi_parameter_set_c": MultiParameterSetCBreakoutWorld,
        "multi_parameter_set_d": MultiParameterSetDBreakoutWorld,
    }

    def get_action_meanings(self):
        return [
            "NOOP",
            "LEFT",
            "RIGHT",
        ]

    def get_keys_to_action(self):
        return {
            (): 0,
            (ord("a"),): 1,
            (ord("d"),): 2,
            (ord("a"), ord("d")): 0,
        }
