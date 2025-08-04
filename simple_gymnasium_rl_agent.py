# Robin Kalia
# robinkalia@berkeley.edu
#
# Using Gymnasium API to simulate interactions between environment
# and an RL agent for the sample CartPole environment.

import gymnasium as gym
import random
import enum


class AgentInitializationApproach(enum.Enum):
    BASIC_ENV = 0
    ENV_WRAPPER = 1
    HUMAN_RENDERING = 2
    RECORD_VIDEO = 3


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, epsilon: float = 0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: gym.core.WrapperActType) -> gym.core.WrapperActType:
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
            print("Selected random action: ", action)
            return action
        return action


def setup_environment(agent_init_approach: AgentInitializationApproach) -> gym.Env:
    if agent_init_approach == AgentInitializationApproach.BASIC_ENV:
        env = gym.make("CartPole-v1")
    elif agent_init_approach == AgentInitializationApproach.ENV_WRAPPER:
        env = RandomActionWrapper(gym.make("CartPole-v1"))
    else:
        env = RandomActionWrapper(gym.make("CartPole-v1", render_mode="rgb_array"))
        if agent_init_approach == AgentInitializationApproach.HUMAN_RENDERING:
            env = gym.wrappers.HumanRendering(env)
        else:
            env = gym.wrappers.RecordVideo(env, video_folder="results")

    return env


class RLAgent():
    def __init__(self):
        self.total_steps = 0.0
        self.total_reward = 0

    def step(self, env: gym.Env, action: int):
        obs, reward, done, trunc, info = env.step(action)
        self.total_reward += reward
        self.total_steps += 1
        return done

    def get_total_reward(self) -> float:
        return self.total_reward

    def get_total_steps(self) -> int:
        return self.total_steps


if __name__ == "__main__":
    agent_init_approach = AgentInitializationApproach.HUMAN_RENDERING
    env = setup_environment(agent_init_approach)
    obs, info = env.reset()
    rl_agent = RLAgent()

    MAX_NUM_STEPS = 1000

    for _ in range(MAX_NUM_STEPS):
        action = env.action_space.sample()
        done = rl_agent.step(env, action)
        if done:
            print("Current Episode finished. Steps executed = %d" % rl_agent.get_total_steps())
            obs, _ = env.reset()

    print("RL Agent interaction complete after %d steps with total reward = %.2f" % (rl_agent.get_total_steps(), rl_agent.get_total_reward()))
