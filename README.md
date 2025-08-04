### Simple RL Agent and Environment.

A dummy implementation of a reinforcement learning problem 
with a dummy environment and a simple agent.

Just run:-
```
$ python3 simple_rl_agent_environment.py
```

Install conda and create a virtual environment to install gymnasium 
and other corresponding gymnasium environments.
```
$ conda create --name deep_rl
$ conda activate deep_rl
$ conda config --add channels conda-forge
$ conda config --set channel_priority strict
$ conda install gymnasium gymnasium-all gymnasium-atari gymnasium-box2d gymnasium-classic_control gymnasium-mujoco gymnasium-other gymnasium-toy_text
```

Now run 
```
$ python3 simple_gymnasium_rl_agent.py
```