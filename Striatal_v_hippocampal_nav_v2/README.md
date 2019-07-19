# striatum-hippocampus

This repo contains code implementing a striatal and hippocampal learning model in the Morris water maze. To run a comparison between the models, open and run the jupyter notebook. The simulated experiments mentioned in the manuscript can be found in the Experiments folder. 

Behavioural and neural data suggest that multiple strategies and brain systems support learning about reward. 
Reinforcement learning (RL) considers these strategies in terms of model-based (MB) learning, which involves learning an explicit transition model of the available state-space, or model-free (MF) learning, which involves directly estimating the cumulative future reward associated with each state. 
The MB-MF distinction parallels that between  place learning and response learning within spatial navigation. 
Lesion and inactivation studies suggests that dorsal hippocampus (dHC) underlies both MB and place learning, whereas dorsolateral striatum (DLS) underlies MF and response learning. 
Here, we present a computational model of the dHC and DLS contributions to reward learning that applies to  spatial and nonspatial domains. 
In the model, hippocampal place cell firing reflects the geodesic distance of other states from its preferred state along the graph of task structure. Accordingly, this population can support a value function or goal cell firing rate, via one-shot learning, on which gradient ascent corresponds to taking the shortest path on the graph, behaviour associated with MB planning. 
In contrast, the DLS learns through MF stimulus-response associations with egocentric environmental cues. 
We show that this model reproduces animal behaviour on spatial navigation tasks using the Morris Water Maze and the Plus Maze, and human behaviour on non-spatial two-step decision tasks. 
We discuss how the geodesic place cell fields could be learnt, and how this type of representation helps to span the gap between MB and MF learning. 
The generality of our model, originally shaped by detailed constraints in the spatial literature, suggests that the hippocampal-striatal system is a general-purpose learning device that adaptively combines MB and MF mechanisms.  


Requirements:

- numpy
- matplotlib
- jupyter
- pandas
- tqdm
- scipy
