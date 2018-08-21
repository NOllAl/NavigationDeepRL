# Introduction

# Environment

The environment solved in this repository is similar to the [Banana collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) environment of Unity with a single agent (aka brain). In this environment, the agent moves in a two-dimensional plane that is populated with *blue* and *yellow* bananas. The agent's goal is to collect yellow bananas, which gives a reward of $+1$ for each one, while avoiding blue banas giving a reward of $-1$ when collected.

The state space is $37$-dimensional and contains the agent's velocity and 6 measurement along 6 rays. The action space is discrete with four choices:

+ `0`: move forward faster
+ `1`: move backward
+ `2`: turn left
+ `3`: turn right

The task is episodic (it ends after a fixed amount of time) and is considered solved when the agent receives a mean reward of 13 over 100 episodes.

[![IMAGE ALT TEXT](environment.jpg)](https://www.youtube.com/watch?v=3x2TjeRQb2Q)

# Setup

