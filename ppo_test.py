# https://pytorch.org/rl/tutorials/coding_ppo.html
# This tutorial does not inspire confidence...
# Missing stuff is here, I guess:
# https://github.com/pytorch/rl/blob/main/tutorials/sphinx-tutorials/coding_ppo.py
#
# Better tutorial: https://github.com/pytorch/rl/blob/main/tutorials/sphinx-tutorials/pendulum.py
# Apparently, torch uses it's own class 'EnvBase' which is assumed to always be compatible with gymnasium envs, but isn't

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from gym_env import gym_env
import guerrilla_checkers

#is_fork = multiprocessing.get_start_method() == "fork"
is_fork = False
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

player = 1
env = gym_env(guerrilla_checkers.game(), player)


