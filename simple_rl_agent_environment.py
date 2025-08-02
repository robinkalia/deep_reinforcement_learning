# Robin Kalia
# robinkalia@berkeley.edu
# 
# A dummy implementation of a reinforcement learning problem 
# with a dummy environment and a simple agent.

from typing import List
import random

# Define the environment.
class Environment:
    # Internal state of the environment.
    def __init__(self):
        # Corresponds to maximum number of steps 
        # allowed in an episode.
        self.rem_steps = 20
    
    # Dummy Observations.
    def get_observations(self) -> List[float]:
        return [0.0, 0.0, 0.0]
    
    # Dummy Actions.
    def get_actions(self) -> List[int]:
        return [0, 1]
    
    # Check if the environment is still receptive to 
    # communication by the agent.
    def is_done(self) -> bool:
        return self.rem_steps == 0
    
    # For a given action taken by the agent,
    # return the reward from the environment.
    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception("The game is finished")
        self.rem_steps -= 1
        return random.random()

    # Reset the environment to its initial state.
    def reset(self) -> List[float]:
        self.rem_steps = 20
        return [0.0, 0.0, 0.0]

# Define the agent.
class Agent:
    # Internal state of the agent.
    def __init__(self):
        self.total_reward = 0

    # Take a step: 
    # a) Get the list of observations from the environment.
    # b) Get the possible actions that can be taken.
    # c) Choose a specific action.
    # d) Get reward from the environment for choosing that specific action.
    # e) Update the agent's state, which is the total reward from the episode
    #    till now, via adding the computed reward value to it.
    def step(self, env: Environment) -> None:
        curr_observations = env.get_observations()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward

    # Return the agent's current state, which is the total rewards
    # accumulated from the episode till a specific time step.
    def get_reward(self) -> float:
        return self.total_reward

if __name__ == "__main__":
    print("\nBeginning the RL Agent's Operations...")
    env = Environment()
    agent = Agent()
    print("\nAgent's Current Reward:-")
    num_steps = 0
    while not env.is_done():
        num_steps += 1
        agent.step(env)
        print(f"\tStep {num_steps}: {agent.get_reward()}")

    print(f"\nRL Agent's Total Reward at the end of the episode with {num_steps} steps: {agent.get_reward()}\n")
