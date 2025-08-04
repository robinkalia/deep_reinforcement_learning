import gymnasium as gym
import random

approach = 3

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

if __name__ == "__main__":
    if approach == 1:
        env = gym.make("CartPole-v1")
    elif approach == 2:
        env = RandomActionWrapper(gym.make("CartPole-v1"))
    else:
        env = RandomActionWrapper(gym.make("CartPole-v1", render_mode="rgb_array"))
        if approach == 3:
            env = gym.wrappers.HumanRendering(env)
        else:
            env = gym.wrappers.RecordVideo(env, video_folder="results")

    total_reward = 0.0
    total_steps = 0
    obs, extra_info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            print("Episode finished")
            obs, _ = env.reset()

    print("Episode complete after %d steps with total reward = %.2f" % (total_steps, total_reward))
