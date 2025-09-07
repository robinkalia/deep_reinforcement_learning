# `CartPole-v1` and `FrozenLake-v1` game environments: Cross Entropy - Model-Free, Policy Based, On-Policy Method
#
#   a) Model Free -> Does not build a model of the environment to predict next action or reward.
#   b) Policy Based π(a|s) -> Builds a probability distribution over actions with observations as input.
#      Different from Value based methods that check all actions to select the action which gives the best reward.
#   c) On-Policy -> Uses observations from actions that we get from the current policy that we are updating. Does not use
#      the observations from previous episodes.

import argparse
import numpy as np

import gymnasium as gym
import pygame

import enum
import time
from typing import List, Tuple, Generator

from dataclasses import dataclass

import torch
from torch import optim
import torch.nn as nn

from torch.utils.tensorboard.writer import SummaryWriter

from utils import get_elapsed_time


HIDDEN_LAYER_SIZE = 256

CARTPOLE_BATCH_SIZE = 16
CARTPOLE_REWARD_BOUNDARY_PERCENTILE = 70
CARTPOLE_MAX_REWARD_VALUE = 475
CARTPOLE_DISCOUNT_FACTOR = 1.0

FROZENLAKE_ENV_TYPE = "Non-Slippery"
FROZENLAKE_BATCH_SIZE = 100
FORZENLAKE_REWARD_BOUNDARY_PERCENTILE = 30
FROZENLAKE_MAX_REWARD_VALUE = 0.8
FROZENLAKE_DISCOUNT_FACTOR = 0.9


class GameEnv(enum.Enum):
    CARTPOLE_V1 = 0
    FROZENLAKE_V1 = 1

class AgentInitializationApproach(enum.Enum):
    BASIC_ENV = 0
    HUMAN_RENDERING = 1
    RECORD_VIDEO = 2

@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int


@dataclass
class Episode:
    reward: float
    steps: List[EpisodeStep]


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Discrete)
        shape = (self.observation_space.n, )
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape, dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


# A very simple neural network mapping from observations to actions.
# Policy Based π(a|s) -> Outputs a probability distribution over actions with observations as input.
class Policy(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden_channels: int):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            # In: 4 x Out: 256
            nn.Linear(in_features=in_ch, out_features=hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            # In: 256 x Out: 512
            nn.Linear(in_features=hidden_channels, out_features=2*hidden_channels),
            nn.LayerNorm(2*hidden_channels),
            nn.ReLU(),
            # In: 512 x Out: 2
            nn.Linear(in_features=2*hidden_channels, out_features=out_ch)
        )

    def forward(self, x: List):
        y = self.network(x)
        return y


def iterate_batches(env: gym.Env, policy: Policy, device: torch.device, batch_size: int) -> Generator[List[Episode], None, None]:
    obs, info = env.reset()
    episode_reward = 0.0
    episodes = []
    episode_steps = []
    softmax = nn.Softmax(dim=1)

    while True:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        obs_tensor = obs_tensor.to(device)
        action_pred = softmax(policy(obs_tensor.unsqueeze(0)))
        predicted_actions = action_pred.cpu().data.numpy()[0]
        selected_action = np.random.choice(len(predicted_actions), p=predicted_actions)
        next_obs, reward, done, trunc, info = env.step(selected_action)
        episode_reward += float(reward)
        episode_step = EpisodeStep(observation=obs, action=selected_action)
        episode_steps.append(episode_step)

        if done or trunc:
            episode = Episode(reward=episode_reward, steps=episode_steps)
            episodes.append(episode)
            next_obs, info = env.reset()
            episode_steps = []
            episode_reward = 0.0
            if len(episodes) == batch_size:
                yield episodes
                episodes = []

        obs = next_obs


def filter_episodes(episodes: List[Episode], device: torch.device, reward_boundary_percentile: float, discount_factor: float) -> \
    Tuple[torch.FloatTensor, torch.LongTensor, List[Episode], float, float]:
    # For `CartPole` this becomes `episode_rewards = list(map(lambda episode: episode.reward, episodes))`
    episode_rewards = list(map(lambda episode: episode.reward * (discount_factor ** len(episode.steps)), episodes))
    reward_mean = np.mean(episode_rewards)
    reward_boundary_thresh = np.percentile(episode_rewards, reward_boundary_percentile)

    filtered_episode_observations: List[np.ndarray] = []
    filtered_episode_actions: List[int] = []
    filtered_elite_episodes: List[Episode] = []

    for episode in episodes:
        if episode.reward < reward_boundary_thresh:
            continue
        filtered_episode_observations.extend(map(lambda episode_step: episode_step.observation, episode.steps))
        filtered_episode_actions.extend(map(lambda episode_step: episode_step.action, episode.steps))
        filtered_elite_episodes.append(episode)

    filtered_episode_observations_tensor = torch.tensor(np.stack(filtered_episode_observations), dtype=torch.float32, device=device)
    filtered_episode_actions_tensor = torch.tensor(filtered_episode_actions, dtype=torch.long, device=device)

    return filtered_episode_observations_tensor, filtered_episode_actions_tensor, filtered_elite_episodes, reward_boundary_thresh, reward_mean


def setup_environment(agent_init_approach: AgentInitializationApproach, game_env: GameEnv, env_type: str) -> gym.Env:
    env_str = "CartPole-v1" if game_env == GameEnv.CARTPOLE_V1 else "FrozenLake-v1"
    if game_env == GameEnv.FROZENLAKE_V1:
        is_slippery = False if env_type == FROZENLAKE_ENV_TYPE else True
    if agent_init_approach == AgentInitializationApproach.BASIC_ENV:
        if game_env == GameEnv.CARTPOLE_V1:
            env = gym.make(env_str)
        else:
            env = gym.make(env_str, is_slippery=is_slippery)
    else:
        if game_env == GameEnv.CARTPOLE_V1:
            env = gym.make(env_str, render_mode="rgb_array")
        else:
            env = gym.make(env_str, is_slippery=is_slippery, render_mode="rgb_array")

        # Observation is discrete (one of 16 cells) for the `FrozenLake` environment. Hence,
        # it can be encoded as a 1-hot vector with a value of 1 at the specific position
        # where the current position is and 0 at every other position in the vector. On
        # the other hand observation is a collection of 4 floating point numbers for the
        # `CartPole` environment that would generally differ drastically from other
        # observations in values.
        if game_env == GameEnv.FROZENLAKE_V1:
            env = DiscreteOneHotWrapper(env)
        if agent_init_approach == AgentInitializationApproach.HUMAN_RENDERING:
            env = gym.wrappers.HumanRendering(env)
        else:
            env = gym.wrappers.RecordVideo(env, video_folder="video")

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["CartPole", "FrozenLake"], default="CartPole", help="Game environment choice")
    parser.add_argument("--env_type", choices=[FROZENLAKE_ENV_TYPE], help="FrozenLake environment type")
    args = parser.parse_args()

    game_env = GameEnv.CARTPOLE_V1 if args.env == "CartPole" else GameEnv.FROZENLAKE_V1
    agent_init_approach = AgentInitializationApproach.HUMAN_RENDERING
    env = setup_environment(agent_init_approach, game_env, args.env_type)

    game_env_str = args.env + "-v1"
    pygame.display.set_caption(game_env_str + ": Cross Entropy based RL Agent")

    env_obs_space_shape = env.observation_space.shape
    action_size = int(env.action_space.n)

    print("\nPlaying the game environment: " + game_env_str)
    if game_env == GameEnv.FROZENLAKE_V1:
        print("Is Slippery =", "False" if args.env_type==FROZENLAKE_ENV_TYPE else "True")
    print("\nenv_obs_space_shape =", env_obs_space_shape, "\taction.size =", action_size)

    if torch.cuda.is_available():
        # If you have multiple GPUs, you can assign specific GPU via indices, for example `cuda:0`, `cuda:1`.
        device = torch.device("cuda")
    elif torch.mps.is_available():
        # For Apple's Silicon chip beginning from M2 with support for Metal Performance Shaders (mps).
        device = torch.device("mps")
    else:
        # Just use CPU if you don't have either GPU support or an Apple Computer
        # with M2+ series chips that come with MPS support already enabled.
        device = torch.device("cpu")

    print("\nRunning the code on device:", device)

    policy = Policy(in_ch=env_obs_space_shape[0], out_ch=action_size, hidden_channels=HIDDEN_LAYER_SIZE)
    print("\nPolicy Network:-\n", policy)
    policy.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(policy.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.99))
    writer = SummaryWriter()

    start_time = time.time()

    if game_env == GameEnv.CARTPOLE_V1:
        batch_size = CARTPOLE_BATCH_SIZE
        reward_boundary_percentile = CARTPOLE_REWARD_BOUNDARY_PERCENTILE
        max_reward_value = CARTPOLE_MAX_REWARD_VALUE
        discount_factor = CARTPOLE_DISCOUNT_FACTOR
    else:
        batch_size = FROZENLAKE_BATCH_SIZE
        reward_boundary_percentile = FROZENLAKE_BATCH_SIZE
        max_reward_value = FROZENLAKE_BATCH_SIZE
        discount_factor = FROZENLAKE_DISCOUNT_FACTOR

    filtered_elite_episodes = []
    for iter_no, episodes in enumerate(iterate_batches(env, policy, device, batch_size), start=1):
        # We store previous elite episodes for the `FrozenLake` game environment but discard them
        # for the `CartPole` game environment since it generally has an abundance of them.
        past_elite_episodes = [] if game_env == GameEnv.CARTPOLE_V1 else filtered_elite_episodes
        filtered_episode_observations, filtered_episode_action_labels, filtered_elite_episodes, reward_boundary_thresh, \
            reward_mean = filter_episodes(episodes + past_elite_episodes, device, reward_boundary_percentile, discount_factor)

        if len(filtered_elite_episodes) == 0:
            continue

        if game_env == GameEnv.FROZENLAKE_V1:
            filtered_elite_episodes = filtered_elite_episodes[-500:]

        optimizer.zero_grad()

        action_logits = policy(filtered_episode_observations)
        loss = loss_func(action_logits, filtered_episode_action_labels)
        loss.backward()
        optimizer.step()

        curr_time = time.time()
        elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs = get_elapsed_time(start_time, curr_time)

        print(f"Iteration no {iter_no}: Avg Loss from {len(filtered_elite_episodes)} filtered elite episodes with {len(filtered_episode_observations)} " +
              f"episode steps = {loss.item()}  Reward Threshold = {reward_boundary_thresh}  Reward Mean = {reward_mean}  " +
              f"Elapsed Time: {elapsed_time_hrs} hours, {elapsed_time_mins} mins, and {elapsed_time_secs} secs")

        writer.add_scalar("Avg Loss per filtered batch", loss.item(), iter_no)
        writer.add_scalar("Reward Threshold", reward_boundary_thresh, iter_no)
        writer.add_scalar("Reward Mean", reward_mean, iter_no)

        if reward_mean > max_reward_value:
            print("\nFinished solving %s env in %d iterations with final episode loss = %.4f. Elapsed Time: %d hours, %d mins, and %d secs." % \
                  (game_env_str, iter_no, loss.item(), elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs))
            break

    writer.close()
