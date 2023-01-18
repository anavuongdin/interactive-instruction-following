#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json

from ast import Num
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import numpy as np
from gym import spaces
from gym.spaces.box import Box

# if TYPE_CHECKING:
from torch import Tensor

import habitat_sim
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)
from habitat.core.spaces import Space
from habitat.sims.habitat_simulator.human_trajectory import EllipseTrajectory


def overwrite_config(
    config_from: Config,
    config_to: Any,
    ignore_keys: Optional[Set[str]] = None,
    trans_dict: Optional[Dict[str, Callable]] = None,
) -> None:
    r"""Takes Habitat Lab config and Habitat-Sim config structures. Overwrites
    Habitat-Sim config with Habitat Lab values, where a field name is present
    in lowercase. Mostly used to avoid :ref:`sim_cfg.field = hapi_cfg.FIELD`
    code.
    Args:
        config_from: Habitat Lab config node.
        config_to: Habitat-Sim config structure.
        ignore_keys: Optional set of keys to ignore in config_to
        trans_dict: A Dict of str, callable which can be used on any value that has a matching key if not in ignore_keys.
    """

    def if_config_to_lower(config):
        if isinstance(config, Config):
            return {key.lower(): val for key, val in config.items()}
        else:
            return config

    for attr, value in config_from.items():
        low_attr = attr.lower()
        if ignore_keys is None or low_attr not in ignore_keys:
            if hasattr(config_to, low_attr):
                if trans_dict is not None and low_attr in trans_dict:
                    setattr(config_to, low_attr, trans_dict[low_attr](value))
                else:
                    setattr(config_to, low_attr, if_config_to_lower(value))
            else:
                raise NameError(
                    f"""{low_attr} is not found on habitat_sim but is found on habitat_lab config.
                    It's also not in the list of keys to ignore: {ignore_keys}
                    Did you make a typo in the config?
                    If not the version of Habitat Sim may not be compatible with Habitat Lab version: {config_from}
                    """
                )

def extract_scene_id(scene_path: str):
    return scene_path.split("/")[-1].split(".")[0], scene_path.split("/")[-3]


class BotConfig:
    def __init__(self, **kwargs):
        self.bot_config_path = kwargs["bot_config_path"]
        self.scene_id = kwargs["scene_id"]
        self.prefix = kwargs["prefix"]
        self.default_number_of_bots = kwargs["default_number_of_bots"]
        self.default_bot_trajectory = kwargs["default_bot_trajectory"]
        self.config = None
        self._set_up_bot_trajectory()
    
    @staticmethod
    def add_suffix(file: str):
        return file + ".json"

    def get_bot_position(self):
        return self._get_property("position")
    
    def get_number_of_bots(self):
        try:
            return self._get_property("number_of_bots")
        except:
            return self.default_number_of_bots
    
    def _set_up_bot_trajectory(self):
        bot_trajectory_config = None
        try:
            bot_trajectory_config = self._get_property("bot_trajectory")
        except:
            bot_trajectory_config = [self.default_bot_trajectory for i in range(self.default_number_of_bots)]
        
        self.num_bots = self.get_number_of_bots()
        self.bot_trajectory = [None for _ in range(self.num_bots)]
        for i in range(self.num_bots):
            # Semi lengths, centers, initial angle and angular velocity
            a, b, c, d, alpha, omega = bot_trajectory_config[i]
            try:
                c, d = self.get_bot_position()[i][0], self.get_bot_position()[i][2]
            except:
                pass
            self.bot_trajectory[i] = EllipseTrajectory((a, b), (c, d), alpha, omega)
    
    def get_linear_vel(self, t):
        return [self.bot_trajectory[i].get_linear_vel(t) for i in range(self.num_bots)]
    
    def get_self_rotation_vel(self, t):
        return [self.bot_trajectory[i].get_self_rotation_vel(t) for i in range(self.num_bots)]
    
    def _get_property(self, property=None):
        scene_config = BotConfig.add_suffix(self.scene_id)
        if self.config != None:
            return self.config[property]
        try:
            with open(os.path.join(self.bot_config_path, self.prefix, scene_config)) as f:
                self.config = json.load(f)
                return self.config[property]        
        except:
            raise("Invalid scene: {}, maybe the scene has not been configured properly.".format(self.scene_id))
            exit()



class HabitatSimSensor:
    sim_sensor_type: habitat_sim.SensorType
    _get_default_spec = Callable[..., habitat_sim.sensor.SensorSpec]
    _config_ignore_keys = {"height", "type", "width"}


@registry.register_sensor
class HabitatSimRGBSensor(RGBSensor, HabitatSimSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    sim_sensor_type = habitat_sim.SensorType.COLOR

    RGBSENSOR_DIMENSION = 3

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(
                self.config.HEIGHT,
                self.config.WIDTH,
                self.RGBSENSOR_DIMENSION,
            ),
            dtype=np.uint8,
        )

    def get_observation(
        self, sim_obs: Dict[str, Union[np.ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, : self.RGBSENSOR_DIMENSION]  # type: ignore[index]
        return obs


@registry.register_sensor
class HabitatSimDepthSensor(DepthSensor, HabitatSimSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    _config_ignore_keys = {
        "max_depth",
        "min_depth",
        "normalize_depth",
    }.union(HabitatSimSensor._config_ignore_keys)
    sim_sensor_type = habitat_sim.SensorType.DEPTH

    min_depth_value: float
    max_depth_value: float

    def __init__(self, config: Config) -> None:
        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32,
        )

    def get_observation(
        self, sim_obs: Dict[str, Union[np.ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        check_sim_obs(obs, self)
        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)

            obs = np.expand_dims(
                obs, axis=2
            )  # make depth observation a 3D array
        else:
            obs = obs.clamp(self.config.MIN_DEPTH, self.config.MAX_DEPTH)  # type: ignore[attr-defined, unreachable]

            obs = obs.unsqueeze(-1)  # type: ignore[attr-defined]

        if self.config.NORMALIZE_DEPTH:
            # normalize depth observation to [0, 1]
            obs = (obs - self.config.MIN_DEPTH) / (
                self.config.MAX_DEPTH - self.config.MIN_DEPTH
            )

        return obs


@registry.register_sensor
class HabitatSimSemanticSensor(SemanticSensor, HabitatSimSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    sim_sensor_type = habitat_sim.SensorType.SEMANTIC

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.int32,
        )

    def get_observation(
        self, sim_obs: Dict[str, Union[np.ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        check_sim_obs(obs, self)
        # make semantic observation a 3D array
        if isinstance(obs, np.ndarray):
            obs = obs[..., None].astype(np.int32)
        else:
            obs = obs[..., None]
        return obs


# TODO Sensor Hierarchy needs to be redone here. These should not subclass camera sensors
@registry.register_sensor
class HabitatSimEquirectangularRGBSensor(HabitatSimRGBSensor):
    _get_default_spec = habitat_sim.EquirectangularSensorSpec


@registry.register_sensor
class HabitatSimEquirectangularDepthSensor(HabitatSimDepthSensor):
    _get_default_spec = habitat_sim.EquirectangularSensorSpec


@registry.register_sensor
class HabitatSimEquirectangularSemanticSensor(HabitatSimSemanticSensor):
    _get_default_spec = habitat_sim.EquirectangularSensorSpec


@registry.register_sensor
class HabitatSimFisheyeRGBSensor(HabitatSimRGBSensor):
    _get_default_spec = habitat_sim.FisheyeSensorDoubleSphereSpec


@registry.register_sensor
class HabitatSimFisheyeDepthSensor(HabitatSimDepthSensor):
    _get_default_spec = habitat_sim.FisheyeSensorDoubleSphereSpec


@registry.register_sensor
class HabitatSimFisheyeSemanticSensor(HabitatSimSemanticSensor):
    _get_default_spec = habitat_sim.FisheyeSensorDoubleSphereSpec


def check_sim_obs(obs: Optional[np.ndarray], sensor: Sensor) -> None:
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )


@registry.register_simulator(name="Sim-v0")
class HabitatSim(habitat_sim.Simulator, Simulator):
    r"""Simulator wrapper over habitat-sim

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """

    def _initialize_bots(self):
        self.rigid_obj_mgr = self.get_rigid_object_manager()
        self.obj_templates_mgr = self.get_object_template_manager()

        # Initialize settings for bots
        self.default_num_bots = self.habitat_config.NUMBER_OF_BOTS
        self.bot_template_id = [None for _ in range(self.default_num_bots)]
        self.bot_obj = [None for _ in range(self.default_num_bots)]
        self.vel_control = [None for _ in range(self.default_num_bots)]
        self.starting_point = [None for _ in range(self.default_num_bots)]

        # Initialize configurations for bots
        scene_id, prefix = extract_scene_id(self.habitat_config.SCENE)
        self.bot_config = BotConfig(bot_config_path=self.habitat_config.BOT_CONFIG_PATH, 
                               scene_id=scene_id, prefix=prefix, default_number_of_bots=self.default_num_bots,
                               default_bot_trajectory=self.habitat_config.BOT_TRAJECTORY)
        bot_position = self.bot_config.get_bot_position()
        num_bots = self.bot_config.get_number_of_bots()

        for idx in range(num_bots):
            # Initialize for each bot
            self.bot_template_id[idx] = self.obj_templates_mgr.load_configs(
                self.habitat_config.BOT_PATH + str(idx)
            )[0]
            self.bot_obj[idx] = self.rigid_obj_mgr.add_object_by_template_id(self.bot_template_id[idx])
            self.bot_obj[idx].motion_type = habitat_sim.physics.MotionType.DYNAMIC
            self.vel_control[idx] = self.bot_obj[idx].velocity_control
            self.vel_control[idx].controlling_lin_vel = True
            self.vel_control[idx].controlling_ang_vel = True
            self.starting_point[idx] = bot_position[idx]
            self.bot_obj[idx].translation = bot_position[idx]

        self._update_bot_vel(reset=True)
        del bot_position
    
    def _reset_position(self):
        for idx in range(self.bot_config.num_bots):
            self.bot_obj[idx].translation = self.starting_point[idx]
        
        self._update_bot_vel(reset=True)

    def _update_bot_vel(self, reset=False):
        if reset:
            self.counter = 0

        self.counter += 1
        current_linear_vel = self.bot_config.get_linear_vel(self.counter)
        current_angular_vel = self.bot_config.get_self_rotation_vel(self.counter)
        for idx in range(self.bot_config.num_bots):
            self.vel_control[idx].linear_velocity = current_linear_vel[idx]
            self.vel_control[idx].angular_velocity = current_angular_vel[idx]

    def _destroy_bots(self):
        del self.rigid_obj_mgr
        del self.obj_templates_mgr
        del self.bot_template_id
        del self.bot_obj
        del self.vel_control
        del self.bot_config
        del self.starting_point

    def __init__(self, config: Config) -> None:
        self.habitat_config = config
        agent_config = self._get_agent_config()
        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.habitat_config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        super().__init__(self.sim_config)
        # load additional object paths specified by the dataset
        # TODO: Should this be moved elsewhere?
        obj_attr_mgr = self.get_object_template_manager()
        for path in self.habitat_config.ADDITIONAL_OBJECT_PATHS:
            obj_attr_mgr.load_configs(path)
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._prev_sim_obs: Optional[Observations] = None
        self._initialize_bots()

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config(
            config_from=self.habitat_config.HABITAT_SIM_V0,
            config_to=sim_config,
            # Ignore key as it gets propogated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_dataset_config_file = (
            self.habitat_config.SCENE_DATASET
        )
        sim_config.scene_id = self.habitat_config.SCENE
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(),
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "start_position",
                "start_rotation",
            },
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[misc]
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys=sensor._config_ignore_keys,
                # TODO consider making trans_dict a sensor class var too.
                trans_dict={
                    "sensor_model_type": lambda v: getattr(
                        habitat_sim.FisheyeSensorModelType, v
                    ),
                    "sensor_subtype": lambda v: getattr(
                        habitat_sim.SensorSubType, v
                    ),
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.habitat_config.HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.habitat_config.ACTION_SPACE_CONFIG
        )(self.habitat_config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.habitat_config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated
    
    def _get_current_bot_locations(self):
        try:
            if self.bot_obj is not None:
                return np.array([self.bot_obj[idx].translation for idx in range(self.bot_config.num_bots)])
        except:
            return np.ones((self.bot_config.num_bots,3))
    
    def _get_relative_bot_locations(self):
        absolute_bot_locations = self._get_current_bot_locations()
        relative_bot_locations = absolute_bot_locations - self.agents[0].get_state().position
        if absolute_bot_locations.shape[0] != self.default_num_bots:
            missing_length = self.default_num_bots - absolute_bot_locations.shape[0]
            relative_bot_locations = np.append(relative_bot_locations, np.zeros((missing_length, 3)), axis=0)
        
        return relative_bot_locations

    def reset(self) -> Observations:
        sim_obs = super().reset()
        self._reset_position()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        return self._sensor_suite.get_observations(sim_obs)

    def step(self, action: Union[str, np.ndarray, int]) -> Observations:
        sim_obs = super().step(action)
        self._prev_sim_obs = sim_obs
        self._update_bot_vel()
        
        observations = self._sensor_suite.get_observations(sim_obs)
        return observations

    def render(self, mode: str = "rgb") -> Any:
        r"""
        Args:
            mode: sensor whose observation is used for returning the frame,
                eg: "rgb", "depth", "semantic"

        Returns:
            rendered frame according to the mode
        """
        sim_obs = self.get_sensor_observations()
        observations = self._sensor_suite.get_observations(sim_obs)

        output = observations.get(mode)
        assert output is not None, "mode {} sensor is not active".format(mode)
        if not isinstance(output, np.ndarray):
            # If it is not a numpy array, it is a torch tensor
            # The function expects the result to be a numpy array
            output = output.to("cpu").numpy()

        return output

    def reconfigure(self, habitat_config: Config) -> None:
        # TODO(maksymets): Switch to Habitat-Sim more efficient caching
        is_same_scene = habitat_config.SCENE == self._current_scene
        self.habitat_config = habitat_config
        self.sim_config = self.create_sim_config(self._sensor_suite)
        if not is_same_scene:
            self._destroy_bots()
            self._current_scene = habitat_config.SCENE
            self.close()
            super().reconfigure(self.sim_config)
            self._initialize_bots()
        
        self._update_agents_state()

    def geodesic_distance(
        self,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[
            Sequence[float], Sequence[Sequence[float]], np.ndarray
        ],
        episode: Optional[Episode] = None,
    ) -> float:
        if episode is None or episode._shortest_path_cache is None:
            path = habitat_sim.MultiGoalShortestPath()
            if isinstance(position_b[0], (Sequence, np.ndarray)):
                path.requested_ends = np.array(position_b, dtype=np.float32)
            else:
                path.requested_ends = np.array(
                    [np.array(position_b, dtype=np.float32)]
                )
        else:
            path = episode._shortest_path_cache

        path.requested_start = np.array(position_a, dtype=np.float32)

        self.pathfinder.find_path(path)

        if episode is not None:
            episode._shortest_path_cache = path

        return path.geodesic_distance

    def action_space_shortest_path(
        self,
        source: AgentState,
        targets: Sequence[AgentState],
        agent_id: int = 0,
    ) -> List[ShortestPathPoint]:
        r"""
        Returns:
            List of agent states and actions along the shortest path from
            source to the nearest target (both included). If one of the
            target(s) is identical to the source, a list containing only
            one node with the identical agent state is returned. Returns
            an empty list in case none of the targets are reachable from
            the source. For the last item in the returned list the action
            will be None.
        """
        raise NotImplementedError(
            "This function is no longer implemented. Please use the greedy "
            "follower instead"
        )

    @property
    def up_vector(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    @property
    def forward_vector(self) -> np.ndarray:
        return -np.array([0.0, 0.0, 1.0])

    def get_straight_shortest_path_points(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self.pathfinder.find_path(path)
        return path.points

    def sample_navigable_point(self) -> List[float]:
        return self.pathfinder.get_random_navigable_point().tolist()

    def is_navigable(self, point: List[float]) -> bool:
        return self.pathfinder.is_navigable(point)

    def semantic_annotations(self):
        r"""
        Returns:
            SemanticScene which is a three level hierarchy of semantic
            annotations for the current scene. Specifically this method
            returns a SemanticScene which contains a list of SemanticLevel's
            where each SemanticLevel contains a list of SemanticRegion's where
            each SemanticRegion contains a list of SemanticObject's.

            SemanticScene has attributes: aabb(axis-aligned bounding box) which
            has attributes aabb.center and aabb.sizes which are 3d vectors,
            categories, levels, objects, regions.

            SemanticLevel has attributes: id, aabb, objects and regions.

            SemanticRegion has attributes: id, level, aabb, category (to get
            name of category use category.name()) and objects.

            SemanticObject has attributes: id, region, aabb, obb (oriented
            bounding box) and category.

            SemanticScene contains List[SemanticLevels]
            SemanticLevel contains List[SemanticRegion]
            SemanticRegion contains List[SemanticObject]

            Example to loop through in a hierarchical fashion:
            for level in semantic_scene.levels:
                for region in level.regions:
                    for obj in region.objects:
        """
        return self.semantic_scene

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.habitat_config.DEFAULT_AGENT_ID
        agent_name = self.habitat_config.AGENTS[agent_id]
        agent_config = getattr(self.habitat_config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        return self.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        r"""Sets agent state similar to initialize_agent, but without agents
        creation. On failure to place the agent in the proper position, it is
        moved back to its previous pose.

        Args:
            position: list containing 3 entries for (x, y, z).
            rotation: list with 4 entries for (x, y, z, w) elements of unit
                quaternion (versor) representing agent 3D orientation,
                (https://en.wikipedia.org/wiki/Versor)
            agent_id: int identification of agent from multiagent setup.
            reset_sensors: bool for if sensor changes (e.g. tilt) should be
                reset).

        Returns:
            True if the set was successful else moves the agent back to its
            original pose and returns false.
        """
        agent = self.get_agent(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors)
        return True

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            sim_obs = self.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def distance_to_closest_obstacle(
        self, position: np.ndarray, max_search_radius: float = 2.0
    ) -> float:
        return self.pathfinder.distance_to_closest_obstacle(
            position, max_search_radius
        )

    def island_radius(self, position: Sequence[float]) -> float:
        return self.pathfinder.island_radius(position)

    @property
    def previous_step_collided(self):
        r"""Whether or not the previous step resulted in a collision

        Returns:
            bool: True if the previous step resulted in a collision, false otherwise

        Warning:
            This feild is only updated when :meth:`step`, :meth:`reset`, or :meth:`get_observations_at` are
            called.  It does not update when the agent is moved to a new loction.  Furthermore, it
            will _always_ be false after :meth:`reset` or :meth:`get_observations_at` as neither of those
            result in an action (step) being taken.
        """
        # return self._prev_sim_obs.get("collided", False)
        return self._is_collided_with_bots()
    
    def _is_collided_with_bots(self):
        rel_bot_loc = self._get_relative_bot_locations()
        dist = np.square(rel_bot_loc).sum(axis=1)
        return dist[dist<self.habitat_config.COLLIDE_DIST].size > 0
