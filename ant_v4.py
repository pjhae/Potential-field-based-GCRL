import numpy as np
import matplotlib.pyplot as plt
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="ant.xml",
        ctrl_cost_weight=0.5,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.3, 1.0),
        reset_noise_scale=0.1,
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z and np.linalg.norm(np.array([10,0]) - np.array(self.get_body_com("torso")[:2])) > 0.5
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def step(self, action):
        # print(self.state_vector()[0], self.state_vector()[2])
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()
        
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)

        forward_reward = x_velocity

        rewards = forward_reward
        
        costs = ctrl_cost 

        reward = 10*rewards + 0.01

        # reward = -np.linalg.norm(np.array([10,0]) - np.array([xy_position_after]))**2
        # if np.linalg.norm(np.array([10,0]) - np.array(self.get_body_com("torso")[:2])) < 0.5:
        #     reward = 100
        #     print("Goal in!")

        done = self.done
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        #########################################################################
        # # 1. For Training
        # camera_data = np.array(self.render("rgb_array", 84, 84, 0))
        # CHW = np.transpose(camera_data, (2, 0, 1))

        # print(camera_data, CHW)

        # # If you wanna check the input image
        # plt.imshow(camera_data)
        # plt.show()

        # ## 2. For rendering check
        # data = self._get_viewer("rgb_array").read_pixels(52, 52, depth=False)
        # CHW = np.transpose(data[::-1, :, :], (2, 0, 1))

        # obs_dct = {}
        # obs_dct['image'] = np.array(CHW) / 255.0
        # obs_dct['vector'] = np.concatenate([self.action_buffer, [self.state_vector()[4]]])
        # #########################################################################





        observations = np.concatenate((np.array([10,0]), position , velocity))
        return observations

    def reset_model(self):

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        # # Update the position in the model
        # box_id = self.sim.model.geom_name2id('Goal')
        # new_position = np.random.uniform(low=-1.0, high=1.0, size=3)
        # self.sim.model.geom_pos[box_id] = new_position

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)