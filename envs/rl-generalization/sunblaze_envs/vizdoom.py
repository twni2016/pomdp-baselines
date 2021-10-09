import copy
import itertools
import os
import tempfile

import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import vizdoom
import omg

ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets", "vizdoom")

# Texture set A.
TEXTURES_SET_A = [
    line.strip() for line in open(os.path.join(ASSET_PATH, "texture_set_a.txt"))
]
# Texture set B.
TEXTURES_SET_B = [
    line.strip() for line in open(os.path.join(ASSET_PATH, "texture_set_b.txt"))
]

# Thing set A.
THINGS_SET_A = [
    int(line.strip()) for line in open(os.path.join(ASSET_PATH, "thing_set_a.txt"))
]
# Thing set B.
THINGS_SET_B = [
    int(line.strip()) for line in open(os.path.join(ASSET_PATH, "thing_set_b.txt"))
]

# Map cache to avoid re-parsing the map multiple times.
_MAP_CACHE = {}


def sampler_with_map_editor(sampler):
    """Wrap sampler function to provide the map editor.

    Maps must be in UDMF format.
    """

    def wrapper(env, config):
        # Load source WAD.
        scenario = os.path.join(ASSET_PATH, config["scenario"])
        map_name = config.get("map", "MAP01").upper()
        cache_key = (scenario, map_name)

        if cache_key not in _MAP_CACHE:
            wad = omg.WadIO(scenario)
            editor = omg.UDMFMapEditor(wad)
            editor.load(map_name)
            _MAP_CACHE[cache_key] = editor
        else:
            editor = _MAP_CACHE[cache_key]

        editor = copy.deepcopy(editor)
        sampler(env, config, editor)

        # Create temporary WAD and write updated level there.
        updated_wad = tempfile.mktemp(suffix=".wad")
        editor.save(updated_wad)

        return updated_wad

    return wrapper


def sample_textures(textures, sides=True, sectors=True):
    """Perform texture sampling.

    :param textures: A set of textures to sample from
    :param sides: Update all side textures (walls)
    :param sectors: Update all sector textures (floor, ceiling)
    """

    @sampler_with_map_editor
    def sampler(env, config, editor):
        """Perform texture sampling.

        :param env: Environment instance
        :param config: Configuration dictionary
        :param editor: Map editor
        """

        # Update side textures.
        if sides:
            for side in editor.sidedefs:
                side.texturemiddle = str(env.np_random.choice(textures))

        # Update floor and ceiling textures for all sectors.
        if sectors:
            for sector in editor.sectors:
                sector.texturefloor = str(env.np_random.choice(textures))
                sector.textureceiling = str(env.np_random.choice(textures))

    return sampler


def sample_things(things, modify_things):
    """Perform thing sampling.

    :param things: A set of things to sample from
    :param modify_things: A set of things that can be modified
    """

    @sampler_with_map_editor
    def sampler(env, config, editor):
        """Perform thing sampling.

        :param env: Environment instance
        :param config: Configuration dictionary
        :param editor: Map editor
        """

        for thing in editor.things:
            # Ignore player start position.
            if thing.type not in modify_things:
                continue

            thing.type = int(env.np_random.choice(things))

    return sampler


class VizDoomEnvironment(gym.Env):
    metadata = {
        "render.modes": ["rgb_array"],
        "video.frames_per_second": 35,
    }
    # Scenario definitions. Within each scenario definition, configuration is inherited
    # from the baseline variant to avoid repetition.
    scenarios = {
        "basic": {
            "baseline": {
                "scenario": "basic.wad",
                "living_reward": 1,
                "death_penalty": 0,
                "reward": "health",
            },
            "floor_ceiling_flipped": {"scenario": "basic_floor_ceiling_flipped.wad"},
            "torches": {"scenario": "basic_torches.wad"},
            "random_textures_set_a": {"sampler": sample_textures(TEXTURES_SET_A)},
            "random_textures_set_b": {"sampler": sample_textures(TEXTURES_SET_B)},
            "random_things_set_a": {
                "scenario": "basic_torches.wad",
                "sampler": sample_things(THINGS_SET_A, modify_things=[56]),
            },
            "random_things_set_b": {
                "scenario": "basic_torches.wad",
                "sampler": sample_things(THINGS_SET_B, modify_things=[56]),
            },
        },
        "navigation": {
            "baseline": {
                "scenario": "navigation.wad",
                "living_reward": 1,
                "death_penalty": 0,
                "reward": "health",
            },
            "new_layout": {"scenario": "navigation_new_layout.wad"},
            "floor_ceiling_flipped": {
                "scenario": "navigation_floor_ceiling_flipped.wad"
            },
            "torches": {"scenario": "navigation_torches.wad"},
            "random_textures_set_a": {"sampler": sample_textures(TEXTURES_SET_A)},
            "random_textures_set_b": {"sampler": sample_textures(TEXTURES_SET_B)},
            "random_things_set_a": {
                "scenario": "navigation_torches.wad",
                "sampler": sample_things(THINGS_SET_A, modify_things=[56]),
            },
            "random_things_set_b": {
                "scenario": "navigation_torches.wad",
                "sampler": sample_things(THINGS_SET_B, modify_things=[56]),
            },
        },
    }
    # Available buttons.
    buttons = [
        vizdoom.Button.MOVE_FORWARD,
        vizdoom.Button.MOVE_BACKWARD,
        vizdoom.Button.MOVE_RIGHT,
        vizdoom.Button.MOVE_LEFT,
        vizdoom.Button.TURN_LEFT,
        vizdoom.Button.TURN_RIGHT,
        vizdoom.Button.ATTACK,
        vizdoom.Button.SPEED,
    ]
    opposite_button_pairs = [
        (vizdoom.Button.MOVE_FORWARD, vizdoom.Button.MOVE_BACKWARD),
        (vizdoom.Button.MOVE_RIGHT, vizdoom.Button.MOVE_LEFT),
        (vizdoom.Button.TURN_LEFT, vizdoom.Button.TURN_RIGHT),
    ]

    def __init__(self, scenario, variant, obs_type="image", frameskip=4):
        if scenario not in self.scenarios:
            raise error.Error("Unsupported scenario: {}".format(scenario))

        if variant not in self.scenarios[scenario]:
            raise error.Error("Unsupported scenario variant: {}".format(variant))

        # Generate config (extend from baseline).
        config = {}
        config.update(self.scenarios[scenario]["baseline"])
        config.update(self.scenarios[scenario][variant])
        self._config = config

        self._vizdoom = vizdoom.DoomGame()
        self._vizdoom.set_doom_scenario_path(
            os.path.join(ASSET_PATH, config["scenario"])
        )
        self._vizdoom.set_doom_map(config.get("map", "MAP01"))
        self._vizdoom.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        self._vizdoom.set_screen_format(vizdoom.ScreenFormat.BGR24)
        self._vizdoom.set_mode(vizdoom.Mode.PLAYER)

        self._width = 640
        self._height = 480
        self._depth = 3

        # Entity visibility.
        self._vizdoom.set_render_hud(False)
        self._vizdoom.set_render_minimal_hud(False)
        self._vizdoom.set_render_crosshair(False)
        self._vizdoom.set_render_weapon(False)
        self._vizdoom.set_render_decals(False)
        self._vizdoom.set_render_particles(False)
        self._vizdoom.set_render_effects_sprites(False)
        self._vizdoom.set_render_messages(False)
        self._vizdoom.set_render_corpses(False)
        self._vizdoom.set_window_visible(False)
        self._vizdoom.set_sound_enabled(False)

        # Rewards.
        self._vizdoom.set_living_reward(config.get("living_reward", 1))
        self._vizdoom.set_death_penalty(config.get("death_penalty", 100))

        # Duration.
        self._vizdoom.set_episode_timeout(config.get("episode_timeout", 2100))

        # Generate action space from buttons.
        for button in self.buttons:
            self._vizdoom.add_available_button(button)

        self._action_button_map = []
        for combination in itertools.product([False, True], repeat=len(self.buttons)):
            # Exclude any pairs where opposite buttons are pressed.
            valid = True
            for a, b in self.opposite_button_pairs:
                if (
                    combination[self.buttons.index(a)]
                    and combination[self.buttons.index(b)]
                ):
                    valid = False
                    break

            if valid:
                self._action_button_map.append(list(combination))

        self.action_space = spaces.Discrete(len(self._action_button_map))

        if obs_type == "image":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self._height, self._width, self._depth)
            )
        else:
            raise error.Error("Unrecognized observation type: {}".format(obs_type))

        self._scenario = scenario
        self._variant = variant
        self._obs_type = obs_type
        self._frameskip = frameskip
        self._initialized = False
        self._temporary_scenario = None
        self._seed()

    def __getstate__(self):
        return {
            "scenario": self._scenario,
            "variant": self._variant,
            "obs_type": self._obs_type,
            "frameskip": self._frameskip,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self._vizdoom.set_seed(seed % 2 ** 32)
        return [seed]

    def _get_observation(self):
        state = self._vizdoom.get_state()

        if self._obs_type == "image":
            if not state:
                return np.zeros([self._height, self._width, self._depth])

            return state.screen_buffer

        raise NotImplementedError

    def _reset(self):
        # Sample scenario when configured.
        sampler = self._config.get("sampler", None)
        if sampler:
            #  Remove previous temporary scenario.
            if self._temporary_scenario:
                try:
                    os.remove(self._temporary_scenario)
                except OSError:
                    pass

                self._temporary_scenario = None

            self._temporary_scenario = sampler(self, self._config)
            self._vizdoom.set_doom_scenario_path(self._temporary_scenario)

        if not self._initialized:
            self._vizdoom.init()
            self._initialized = True

        self._vizdoom.new_episode()

        return self._get_observation()

    def _get_state_variables(self):
        return {
            "health": self._vizdoom.get_game_variable(vizdoom.GameVariable.HEALTH),
            "frags": self._vizdoom.get_game_variable(vizdoom.GameVariable.FRAGCOUNT),
        }

    def _step(self, action):
        previous_info = self._get_state_variables()
        action = self._action_button_map[action]
        scenario_reward = self._vizdoom.make_action(action, self._frameskip)
        terminal = self._vizdoom.is_episode_finished() or self._vizdoom.is_player_dead()
        observation = self._get_observation()
        info = self._get_state_variables()

        reward_value = self._config.get("reward", "reward")
        if reward_value == "reward":
            reward = scenario_reward
        else:
            reward = info[reward_value] - previous_info[reward_value]

        return observation, reward, terminal, info

    def get_keys_to_action(self):
        # TODO.
        return {
            (): 0,
        }
