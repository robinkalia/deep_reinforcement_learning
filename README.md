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

Sample Output with rendered graphics:-
```
opt/anaconda3/envs/deep_rl/lib/python3.12/site-packages/pygame/pkgdata.py:25: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_stream, resource_exists
2025-08-04 15:54:00.478 python3[36605:14153423] +[IMKClient subclass]: chose IMKClient_Modern
2025-08-04 15:54:00.478 python3[36605:14153423] +[IMKInputSession subclass]: chose IMKInputSession_Modern
Selected random action:  1
Current Episode finished. Steps executed = 12
Selected random action:  1
Current Episode finished. Steps executed = 37
Selected random action:  1
Current Episode finished. Steps executed = 53
Selected random action:  1
Selected random action:  0
Selected random action:  1
Current Episode finished. Steps executed = 86
Selected random action:  0
Selected random action:  1
Current Episode finished. Steps executed = 113
Selected random action:  0
Selected random action:  0
Selected random action:  0
Selected random action:  1
Current Episode finished. Steps executed = 157
...
...
Current Episode finished. Steps executed = 966
Selected random action:  1
Selected random action:  0
Selected random action:  0
Current Episode finished. Steps executed = 983
Selected random action:  0
Current Episode finished. Steps executed = 995
Selected random action:  1
RL Agent interaction complete after 1000 steps with total reward = 1000.00
```