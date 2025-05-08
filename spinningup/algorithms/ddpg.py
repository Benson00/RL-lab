import torch

# Set up function for computing DDPG Q-loss
def loss_q_fn(data, ac, ac_targ, gamma):

    return 0

# Set up function for computing DDPG pi loss
def loss_pi_fn(data, ac):
    
    return 0
    
def update_rule(data, q_optimizer, pi_optimizer, logger, ac, ac_targ, gamma, polyak):
    return 0
