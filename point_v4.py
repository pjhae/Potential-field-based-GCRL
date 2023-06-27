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
        xml_file="point_local.xml",
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

        # obstacles
        self.obs1 = np.array([5,0])
        self.obs2 = np.array([0,-5])
        self.obs3 = np.array([0,-5])

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def is_healthy(self):
        is_healthy = (np.linalg.norm(self._goal_pos - np.array(self.get_body_com("torso")[:2])) > 3.5)
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
        if np.linalg.norm(self._goal_pos - np.array(self.get_body_com("torso")[:2])) < 1.2:
            dist_reward = 500000
            print("!!!!Goal in!!!!", "Init pos : ", self._initial_pos, "Goal pos : ", self._goal_pos)

        # Absorbing reward
        if np.linalg.norm(self._goal_pos - np.array(self.get_body_com("torso")[:2])) < 5:
            dist_reward = 2*(np.linalg.norm(self._goal_pos - np.array([xy_position_before]))  - np.linalg.norm(self._goal_pos - np.array([xy_position_after])))   


        # Calculate collision reward
        col_reward = 0
        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]

            if (str(self.sim.model.geom_id2name(sim_contact.geom2)) == "pointbody" or "pointarrow"):
                if str(self.sim.model.geom_id2name(sim_contact.geom1)) != "floor":
                    self.coll_num += 1
                    col_reward = -3
                    # print("Collision! : Reward -= 40")
                    break
        
        if (np.linalg.norm(self.obs1 - np.array(self.get_body_com("torso")[:2])) < 5) \
            or (np.linalg.norm(self.obs2 - np.array(self.get_body_com("torso")[:2])) < 7.5) \
            or (np.linalg.norm(self.obs3 - np.array(self.get_body_com("torso")[:2])) < 5) :
            col_reward -= 0.025

        reward = dist_reward + col_reward

        done = self.done
        observation = self._get_obs()
        info = {
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),

        }
        

        return observation, reward, done, info

    def _get_obs(self):
        velocity = self.sim.data.qvel.flat.copy()
        observations = np.concatenate([self._goal_pos - self.sim.data.qpos[:2] ,  self.obs1  - self.sim.data.qpos[:2]  , self.obs2 - self.sim.data.qpos[:2], self.obs3 - self.sim.data.qpos[:2]     ,  [np.cos(self.sim.data.qpos[2]) , np.sin(self.sim.data.qpos[2]) ] , velocity  ])
        return observations


    def reset_model(self):
        self.time_step = 0
        self.coll_num = 0

        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * 0.1

        agent_pos_candidate = np.array([[20,0], [-20,0 ], [-20,0],  [-20,0], [-20,0], [-20,0], [0,20], [0,20], [0,20], [0,-20], [0,-20], [0,-20], [0,-20], [0, 20]]) 
        goal_pos_candidate  = np.array([[0,20], [0,-20], [20,0], [ 20,0], [ 0,20], [ 0,-20],[0,-20],[20,0],[-20,0], [0,20], [20,0] , [-20,0] , [0, 20], [0,-20] ])

        # Agent_position_setting
        idx = np.random.randint(14)
        self._initial_pos = np.array(agent_pos_candidate[idx][:])
        qpos[:2] = self._initial_pos + self.np_random.uniform(size=2, low=-1, high=1)
        self.set_state(qpos, qvel)


        self._goal_pos = np.array(goal_pos_candidate[idx][:]) + self.np_random.uniform(size=2, low=-1, high=1)
        box_id = self.sim.model.geom_name2id('Goal')
        self.sim.model.geom_pos[box_id][0:2] = self._goal_pos 

        print("Initial pos : ",self._initial_pos, "Goal pos :", np.array(goal_pos_candidate[idx][:]) )

        observation = self._get_obs()

        return observation


    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)