import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gym

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from utils import run_episole, mlp, cumulate_return

def get_policy_model(sizes):
    layers = []
    for j in range(len(sizes)-1):
        layers += [nn.Linear(sizes[j], sizes[j+1]), nn.Tanh()]
    return nn.Sequential(*layers)

class Method:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
    
    def compute_loss(self, obs, act, weights):
        return -(self.agent.policy_logp(obs, act) * torch.as_tensor(weights, dtype=torch.float)).mean()

    def train(self, epoch=50, batch_size=4000, gamma=0.99, lr=1e-2, render=False):
        optimizer = Adam(self.agent.policy_model.parameters(), lr=lr)
        
        for epoch_i in range(epoch):

            batch_S = []          # for observations
            batch_A = []         # for actions
            batch_ret = []      # for R(tau) weighting in policy gradient

            batch_rets = []         # for measuring episode returns
            batch_lens = []         # for measuring episode lengths

            first_episode_rendered = False

            with torch.no_grad():

                while len(batch_S) < batch_size:
                    
                    S, A, R, _, done = run_episole(self.env, self.agent, 1000, render and not first_episode_rendered)
                    if not first_episode_rendered:
                            first_episode_rendered = True

                    ret = cumulate_return(R[1:], gamma)
                    batch_S += S[:-1]
                    batch_A += A[:-1]
                    batch_ret += ret

                    batch_rets += [ret[0]]
                    batch_lens += [len(S)]

            optimizer.zero_grad()
            batch_loss = self.compute_loss(batch_S, batch_A, batch_ret)
            batch_loss.backward()
            optimizer.step()
            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (epoch_i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


class myAgent():
    def __init__(self, env):
        self.env = env
        feature_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.n
        hidden_dims = 32
        self.policy_model = mlp([feature_dims, hidden_dims, action_dims])

    def policy_distribution(self, state):
        return Categorical(logits=self.policy_model(torch.as_tensor(state, dtype=torch.float32)))

    def policy_logp(self, state, action):
        return self.policy_distribution(state).log_prob(torch.as_tensor(action, dtype=torch.float32))
    
    def select_distribution(self, state):
        return self.policy_distribution(state)
    
    def policy_select(self, state):
        dist = self.select_distribution(state)
        action = dist.sample()
        p_a = dist.log_prob(action)
        return action.item(), p_a.item()

    def state_value(self, state):
        pass


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    ag = myAgent(env)
    method = Method(env, ag)
    method.train()#, render=True)