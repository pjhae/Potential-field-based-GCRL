import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    ORI_IND: int = 2
    MANUAL_COLLISION: bool = True
    RADIUS: float = 0.4
    OBJBALL_TYPE: str = "hinge"
    VELOCITY_LIMITS: float = 10.0

    def __init__(
        self,
        xml_file="point.xml",
        terminate_when_unhealthy=True,

    ):
        utils.EzPickle.__init__(**locals())

        high = np.inf * np.ones(6, dtype=np.float32)
        high[3:] = self.VELOCITY_LIMITS * 1.2
        high[self.ORI_IND] = np.pi
        low = -high
        self.observation_space = gym.spaces.Box(low, high)

        # For initialization variables
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._goal_pos = np.array([30,0])
        self._initial_pos = np.array([0,0])
        self._map_size = 30


        # For timestep and collision check
        self.time_step = 0
        self.coll_num = 0
        self.random_seed = 0
        self.random_seed2 = 0

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def is_healthy(self):
        is_healthy = (np.linalg.norm(self._goal_pos - np.array(self.get_body_com("torso")[:2])) > 3)
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def step(self, action):
        
        self.time_step += 1
        
        # Position before simulation
        xy_position_before = self.get_body_com("torso")[:2].copy()

        ## Do Simulation
        qpos = self.sim.data.qpos.copy()
        qpos[2] += action[1]
        # Clip orientation
        if qpos[2] < -np.pi:
            qpos[2] += np.pi * 2
        elif np.pi < qpos[2]:
            qpos[2] -= np.pi * 2
        ori = qpos[2]
        # Compute increment in each direction
        qpos[0] += np.cos(ori) * action[0]
        qpos[1] += np.sin(ori) * action[0]
        qvel = np.clip(self.sim.data.qvel, -self.VELOCITY_LIMITS, self.VELOCITY_LIMITS)
        self.set_state(qpos, qvel)
        for _ in range(0, self.frame_skip):
            self.sim.step()

        # Position after simulation
        xy_position_after = self.get_body_com("torso")[:2].copy()
        
        # Calculate distance reward
        dist_reward =   np.linalg.norm(self._goal_pos - np.array([xy_position_before]))  - np.linalg.norm(self._goal_pos - np.array([xy_position_after]))   
        if np.linalg.norm(self._goal_pos - np.array(self.get_body_com("torso")[:2])) < 1.5:
            dist_reward = 500000
            print("!!!!Goal in!!!!", "Init pos : ", self._initial_pos, "Goal pos : ", self._goal_pos)

        # Absorbing reward
        if np.linalg.norm(self._goal_pos - np.array(self.get_body_com("torso")[:2])) < 10:
            dist_reward = 3*(np.linalg.norm(self._goal_pos - np.array([xy_position_before]))  - np.linalg.norm(self._goal_pos - np.array([xy_position_after])))   

        # Calculate collision reward
        col_reward = 0
        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]
            # print("geom1 : " ,str(self.sim.model.geom_id2name(sim_contact.geom1)))
            # print("geom2 : " ,str(self.sim.model.geom_id2name(sim_contact.geom2)))
        
            if (str(self.sim.model.geom_id2name(sim_contact.geom2)) == "pointbody" or "pointarrow"):
                if str(self.sim.model.geom_id2name(sim_contact.geom1)) != "floor":
                    self.coll_num += 1
                    col_reward = -2.5
                    # print("Collision! : Reward -= 40")
                    break

        reward = dist_reward + col_reward

        done = self.done
        observation = self._get_obs()
        info = {
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),

        }
        
        # # for grid 1
        # R = 30
        # W = self.random_seed*0.005
        # phs = self.random_seed2*np.pi/3
        # self._goal_pos = np.array([R*np.cos(self.time_step*W+phs), R*np.sin(self.time_step*W+phs)])
        # box_id = self.sim.model.geom_name2id('Goal')
        # self.sim.model.geom_pos[box_id][0:2] = self._goal_pos 

        # for grid 2
        R = 30
        W = self.random_seed*0.004
        phs = self.random_seed2*np.pi/6
        self._goal_pos = np.array([R*np.cos(self.time_step*W+phs), R*np.sin(self.time_step*W+phs) ])
        box_id = self.sim.model.geom_name2id('Goal')
        self.sim.model.geom_pos[box_id][0:2] = self._goal_pos 


        # # for forest
        # R = 25
        # W = self.random_seed*0.008
        # phs = self.random_seed2*np.pi/6
        # box_id = self.sim.model.geom_name2id('wall_29_29')
        # self.sim.model.geom_pos[box_id][0:2] = np.array([R*np.cos(self.time_step*W+phs), R*np.sin(self.time_step*W+phs)])
        # box_id = self.sim.model.geom_name2id('wall_29_25')
        # self.sim.model.geom_pos[box_id][0:2] = np.array([R*np.cos(self.time_step*W+phs+np.pi/2), R*np.sin(self.time_step*W+phs+np.pi/2)])
        # box_id = self.sim.model.geom_name2id('wall_29_27')
        # self.sim.model.geom_pos[box_id][0:2] = np.array([R*np.cos(self.time_step*W+phs+np.pi), R*np.sin(self.time_step*W+phs+np.pi)])
        # box_id = self.sim.model.geom_name2id('wall_25_11')
        # self.sim.model.geom_pos[box_id][0:2] = np.array([R*np.cos(self.time_step*W+phs-np.pi/2), R*np.sin(self.time_step*W+phs-np.pi/2)])

        return observation, reward, done, info

    def _get_obs(self):
        velocity = self.sim.data.qvel.flat.copy()
        observations = np.concatenate([self._goal_pos - self.sim.data.qpos[:2] ,self.obstable_detector(30, self._map_size),  [np.cos(self.sim.data.qpos[2]) , np.sin(self.sim.data.qpos[2]) ] , velocity  ])
        return observations



    def reset_model(self):


        print("Number of collision :", self.coll_num)
        self.time_step = 0
        self.coll_num = 0
        self.random_seed = 2*(np.random.randint(2)-0.5)
        self.random_seed2 = np.random.randint(6)
        qpos = self.init_qpos + self.np_random.uniform(size=self.sim.model.nq, low=-0.1, high=0.1)

        qpos[2] = self.np_random.uniform(size=1, low=-10, high=10)

        # Agent_position_setting
        agent_pos_candidate = np.array([[0,0]]) 
        idx = np.random.randint(1)
        self._initial_pos = np.array(agent_pos_candidate[idx][:])
        qpos[:2] = self._initial_pos
        qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * 0.1
        self.set_state(qpos, qvel)
        observation = self._get_obs()

        # # For forest
        # # Update the position of the goal
        # box_id = self.sim.model.geom_name2id('Goal')
        # # Goal_position_setting
        # goal_pos_candidate = np.array([[-20,40],[0, -50] ,[40, -20],[-40, -10]])
        # idx = np.random.randint(4)
        # self._goal_pos = np.array(goal_pos_candidate[idx][:])
        # print("This goal is : ", self._goal_pos)
        # self.sim.model.geom_pos[box_id][0:2] = self._goal_pos 

        return observation

    def sort_array_by_norm(array, number):
        norms = np.linalg.norm(array, axis=1)  # Calculate norms of each row
        sorted_indices = np.argsort(norms)  # Get sorted indices based on norms
        sorted_array = array[sorted_indices]  # Sort the array based on the indices
        return sorted_array[:number][:]

    def obstable_detector(self, window_size, map_size):
        
        geometry_names = [self.sim.model.geom_id2name(geom_id) for geom_id in range(self.sim.model.ngeom)]

        distance_vectors = np.array([0,0])
        distance = 1000

        for i in range(map_size):
            for j in range(map_size):
                geom_name = "wall_" + str(i) + "_" + str(j)
                if geom_name in  geometry_names:
                    box_id = self.sim.model.geom_name2id(geom_name)
                    pos_difference = np.abs(self.sim.model.geom_pos[box_id][0:2] - self.sim.data.qpos[:2])
                    if np.all(pos_difference <= window_size):
                        if distance > np.linalg.norm(pos_difference):
                            distance = np.linalg.norm(pos_difference)
                            distance_vectors = np.array(self.sim.model.geom_pos[box_id][0:2] - self.sim.data.qpos[:2])

        return distance_vectors


    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)