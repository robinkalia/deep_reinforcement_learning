# CartPole-v1: Cross Entropy - Model-Free, Policy Based, On-Policy Method
#
#   a) Model Free -> Does not build a model of the environment to predict next action or reward.
#   b) Policy Based π(a|s) -> Builds a probability distribution over actions with observations as input.
#      Different from Value based methods that check all actions to select the action which gives the best reward.
#   c) On-Policy -> Uses observations from actions that we get from the policy that we are updating. Does not use
#      the observations from previous episodes.

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

HIDDEN_LAYER_SIZE = 256
BATCH_SIZE = 16
REWARD_BOUNDARY_PERCENTILE = 70

MAX_REWARD_VALUE = 475

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


def filter_episodes(episodes: List[Episode], device: torch.device, reward_boundary_percentile: float) -> \
    Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    episode_rewards = list(map(lambda episode: episode.reward, episodes))
    reward_boundary_thresh = np.percentile(episode_rewards, reward_boundary_percentile)
    reward_mean = np.mean(episode_rewards)

    filtered_episode_observations: List[np.ndarray] = []
    filtered_episode_actions: List[int] = []
    for episode in episodes:
        if episode.reward < reward_boundary_thresh:
            continue
        filtered_episode_observations.extend(map(lambda episode_step: episode_step.observation, episode.steps))
        filtered_episode_actions.extend(map(lambda episode_step: episode_step.action, episode.steps))

    filtered_episode_observations_tensor = torch.tensor(np.stack(filtered_episode_observations), dtype=torch.float32, device=device)
    filtered_episode_actions_tensor = torch.tensor(filtered_episode_actions, dtype=torch.long, device=device)

    return filtered_episode_observations_tensor, filtered_episode_actions_tensor, reward_boundary_thresh, reward_mean


def get_elapsed_time(start_time, curr_time) -> Tuple[int, int, int]:
    train_time_secs = curr_time - start_time
    elapsed_time_hrs = int(train_time_secs / 3600.0)
    elapsed_time_mins = int((train_time_secs - 3600 * elapsed_time_hrs) / 60.0)
    elapsed_time_secs = int(train_time_secs - 60 * elapsed_time_mins - 3600 * elapsed_time_hrs)

    return elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs


def setup_environment(agent_init_approach: AgentInitializationApproach) -> gym.Env:
    if agent_init_approach == AgentInitializationApproach.BASIC_ENV:
        env = gym.make("CartPole-v1")
    else:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        if agent_init_approach == AgentInitializationApproach.HUMAN_RENDERING:
            env = gym.wrappers.HumanRendering(env)
        else:
            env = gym.wrappers.RecordVideo(env, video_folder="video")

    return env


if __name__ == "__main__":
    agent_init_approach = AgentInitializationApproach.HUMAN_RENDERING
    env = setup_environment(agent_init_approach)

    pygame.display.set_caption("Cartpole-v1: Cross Entropy based RL Agent")

    env_obs_space_shape = env.observation_space.shape
    action_size = int(env.action_space.n)

    print("env_obs_space_shape =", env_obs_space_shape, "\taction.size =", action_size)

    if torch.cuda.is_available():
        # If you have multiple GPUs, you can assign specific GPU via indices, for example `cuda:0`, `cuda:1`.
        device = torch.device("cuda")
    elif torch.mps.is_available():
        # For Apple's Silicon chip beginning from M2 with support for Metal Performance Shaders (mps).
        device = torch.device("mps")
    else:
        # Just use CPU if you don't have either GPU support or an Apple Computer with M2+ series chips that come with MPS support already enabled.
        device = torch.device("cpu")

    print("\nRunning the code on device:", device)

    policy = Policy(in_ch=env_obs_space_shape[0], out_ch=action_size, hidden_channels=HIDDEN_LAYER_SIZE)
    print("\nPolicy Network:-\n", policy)
    policy.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(policy.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.99))
    writer = SummaryWriter()

    start_time = time.time()

    for iter_no, batch in enumerate(iterate_batches(env, policy, device, BATCH_SIZE), start=1):
        filtered_episode_observations, filtered_episode_action_labels, reward_boundary_thresh, reward_mean = \
            filter_episodes(batch, device, REWARD_BOUNDARY_PERCENTILE)

        optimizer.zero_grad()

        action_logits = policy(filtered_episode_observations)
        loss = loss_func(action_logits, filtered_episode_action_labels)
        loss.backward()
        optimizer.step()

        curr_time = time.time()
        elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs = get_elapsed_time(start_time, curr_time)

        print(f"Iteration no {iter_no}: Avg Loss from filtered {len(filtered_episode_observations)} episodes = {loss.item()}  " +
              f"Reward Threshold = {reward_boundary_thresh}  Reward Mean = {reward_mean}  " +
              f"Elapsed Time: {elapsed_time_hrs} hours, {elapsed_time_mins} mins, and {elapsed_time_secs} secs")

        writer.add_scalar("Avg Loss per filtered batch", loss.item(), iter_no)
        writer.add_scalar("Reward Threshold", reward_boundary_thresh, iter_no)
        writer.add_scalar("Reward Mean", reward_mean, iter_no)

        if reward_mean > MAX_REWARD_VALUE:
            print("\nFinished solving CartPole-v1 env in %d iterations with final episode loss = %.4f. Elapsed Time: %d hours, %d mins, and %d secs." % \
                  (iter_no, loss.item(), elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs))
            break

    writer.close()
