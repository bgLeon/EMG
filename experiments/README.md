# Experiments

The 'map' folder contains 10 randomly generated maps. The first 5 maps (from map_0.txt to map_4.txt) were unbiasedly sampled. The last 5 maps (from map_5.txt to map_9.txt) were biased towards maps with a big gap between locally optimal and globally optimal policies over 10 tasks. Each grid position might contain a resource, workstation, wall, or being empty. A brief explanation of each symbol follows:

- 'A' is the agent
- 'X' is a wall
- 'a' is a tree
- 'b' is a toolshed
- 'c' is a workbench
- 'd' is grass
- 'e' is a factory
- 'f' is iron
- 'g' is gold
- 'h' is gem
- 's' is part of the shelter

We defined 3 set of tasks to *teach* the agent in these maps (called *sequence*, *interleaving*, and *safety*). These tasks are based on the 10 tasks defined by [Andreas et al.](https://arxiv.org/abs/1611.01796) for the crafting environment. We also included the optimal number of steps needed to solve each task per map in the 'optimal_policies' folder. We use those values to normalize the discounted rewards in our experiments.
