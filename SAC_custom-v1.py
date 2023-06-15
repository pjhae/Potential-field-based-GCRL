import gym
import torch as th
import torch.nn as nn
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack


def lin_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func


date = "0615"
trial = "B"

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


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                n_input_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(nn.Conv2d(n_input_channels, 8, 4, stride=4, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 2, 3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                )

                total_concat_size += 2*2*2
                
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 7)
                total_concat_size += 7

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


env = make_vec_env("Point-v2", n_envs=1)
# env = VecFrameStack(env, n_stack=3,  channels_order = "first")

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
)

model = SAC("MultiInputPolicy", env=env, device = 'cuda' ,verbose=2, tensorboard_log='./Point_tb_log_'+ date, learning_starts=100,
            learning_rate=5e-5, tau=0.005, buffer_size=50000, gamma=0.99, gradient_steps=1, train_freq= 1, action_noise=None, replay_buffer_class=None, replay_buffer_kwargs=None,
            optimize_memory_usage=False, ent_coef='auto',target_update_interval=1, target_entropy='auto', use_sde=False, sde_sample_freq=-1, batch_size=256*2)


model.learn(total_timesteps=100000000,
		callback=event_callback,  # every n_steps, save the model.
		tb_log_name='Point_tb_'+date+trial
		# ,reset_num_timesteps=False   # if you need to continue learning by loading existing model, use this option.
		)

model.save("Point_model")
del model # remove to demonstrate saving and loading
model = SAC.load("Point_model")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    
    env.render()
