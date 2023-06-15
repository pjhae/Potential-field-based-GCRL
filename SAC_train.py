import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
# For evaluation
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback



def lin_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

date = "0614"
trial = "C"

checkpoint_on_event = CheckpointCallback(
    save_freq=1,
    save_path='./save_model_'+date+'/'+trial,
    verbose=2,
    name_prefix='Point_model_'+date+trial
)

event_callback = EveryNTimesteps(
    n_steps=int(1e4),  # every n_steps, save the model
    callback=checkpoint_on_event
)


env = make_vec_env("Point-v1", n_envs=1)
# class EvaluateCallback(BaseCallback):
#     def __init__(self, eval_env, eval_freq=10, n_eval_episodes=10, verbose=0):
#         super(EvaluateCallback, self).__init__(verbose)
#         self.eval_env = eval_env
#         self.eval_freq = eval_freq
#         self.n_eval_episodes = n_eval_episodes

#     def _on_training_start(self) -> None:
#         # Perform an initial evaluation before training starts
#         self._eval_policy()

#     def _on_step(self) -> bool:
#         # Check if it's time to evaluate the policy
#         if self.num_timesteps % self.eval_freq == 0:
#             self._eval_policy()
#         return True

#     def _eval_policy(self):
#         episode_rewards = []
#         for _ in range(self.n_eval_episodes):
#             obs = self.eval_env.reset()
#             done = False
#             episode_reward = 0.0
#             while not done:
#                 action, _ = self.model.predict(obs, deterministic=True)
#                 obs, reward, done, _ = self.eval_env.step(action)
#                 episode_reward += reward
#             episode_rewards.append(episode_reward)
#         mean_reward = np.mean(episode_rewards)
#         self.logger.record("eval/mean_reward", mean_reward)

#         if self.verbose > 0:
#             print(f"Eval: mean_reward={mean_reward:.2f}")

# eval_callback = EvaluateCallback(env, eval_freq=50, n_eval_episodes=10, verbose=1)


#env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
model = SAC("MlpPolicy", env=env, device = 'cuda' ,verbose=2, tensorboard_log='./Point_tb_log_'+ date, learning_starts=100,
            learning_rate=5e-5, tau=0.005, buffer_size=50000, gamma=0.99, gradient_steps=1, train_freq= 1, action_noise=None, replay_buffer_class=None, replay_buffer_kwargs=None,
            optimize_memory_usage=False, ent_coef='auto',target_update_interval=1, target_entropy='auto', use_sde=False, sde_sample_freq=-1, batch_size=256*2)


model.learn(total_timesteps=30000000,
		callback=event_callback,  # every n_steps, save the model.
		tb_log_name='point_tb_'+date+trial
		,reset_num_timesteps=True   # if you need to continue learning by loading existing model, use this option.
		
		)


model.save("Point_model")
del model # remove to demonstrate saving and loading
model = SAC.load("Point_model")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    
    env.render()
