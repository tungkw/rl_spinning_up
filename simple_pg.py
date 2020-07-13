import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gym

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

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
        logp = self.agent.policy_distribution(obs).log_prob(act)
        return -(logp * weights).mean()

    def train(self, epoch, batch_size, lr=1e-2, render=False):
        optimizer = Adam(self.agent.policy_model.parameters(), lr=lr)
        
        for epoch_i in range(epoch):

            # batch_obs = []          # for observations
            # batch_acts = []         # for actions
            # batch_weights = []      # for R(tau) weighting in policy gradient
            batch_rets = []         # for measuring episode returns
            batch_lens = []         # for measuring episode lengths

            first_episode_rendered = False
            batch_loss = torch.as_tensor(0, dtype=torch.float32)
            for i in range(batch_size):
            # while len(batch_obs) < batch_size:
                t = 0
                S = [self.env.reset()]
                A = [self.agent.policy_select(S[t])]
                R = [0]
                while True:
                    if render and (not first_episode_rendered):
                        self.env.render()
                    S_next, r, done, _ = self.env.step(A[t])
                    S.append(S_next)
                    A.append(self.agent.policy_select(S_next))
                    R.append(r)
                    t+=1
                    if done:
                        first_episode_rendered = True
                        break
                G = np.sum(R)

                logp = self.agent.policy_distribution(torch.as_tensor(S, dtype=torch.float32)).log_prob(torch.as_tensor(A, dtype=torch.float32))
                weights = torch.as_tensor([G] * len(S), dtype=torch.float32)
                batch_loss += - torch.sum(logp * weights)
                
                # batch_obs += S
                # batch_acts += A
                # batch_weights += [G] * len(S)
                batch_rets += [G]
                batch_lens += [len(S)]

            optimizer.zero_grad()
            # batch_loss = self.compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            #                         act=torch.as_tensor(batch_acts, dtype=torch.int32),
            #                         weights=torch.as_tensor(batch_weights, dtype=torch.float32)
            #                         )
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
        self.policy_model = get_policy_model([feature_dims, hidden_dims, action_dims])

    def policy_distribution(self, state):
        return Categorical(logits=self.policy_model(state))

    def policy(self, state, action):
        pmf = self.policy_distribution(torch.as_tensor(state, dtype=torch.float32)).probs
        return pmf[action]
    
    def policy_select(self, state):
        return Categorical(logits=self.policy_model(torch.as_tensor(state, dtype=torch.float32))).sample().item()
        

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    ag = myAgent(env)
    method = Method(env, ag)
    method.train(50, 500)#, render=True)