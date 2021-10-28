# Is  Transfer  Learning  helpful  for  Neural  Combinatorial  Optimization applied to Vehicle Routing Problems?

This repository contains our code and results of our experiments on applying transfer learning in the case of neural combinatorial optimisation applied for routing problems. Our use case is transferring the routing policy learned in the case of TSP to the CVRP.

The model used is a personal implementation of [Attention Model](https://github.com/wouterkool/attention-learn-to-route) with small modifications in order to have the same number of parameters learned between the problems.

![The model used in our experiments](ResultsTL/tl-tsp-vrp.png)

Some of our findings suggest that when using transfer learning from TSP to CVRP, there is a speed up in learning when a relatively small number of instances are used per epoch.

![Average tour lengths per epoch](ResultsTL/average-tour-lengths-per-epoch-in-training.png)

The link to the paper will be available when the paper will be published.
