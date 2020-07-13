import algo
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gym

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

def mlp(sizes):
    layers = []
    for j in range(len(sizes)-1):
        layers += [nn.Linear(sizes[j], sizes[j+1]), nn.Tanh()]
    return nn.Sequential(*layers)

def cumulate_return(R, discount):
    G = 0
    ret = []
    for i in reversed(range(len(R)-1)):
        G = R[i+1] + discount * G
        ret = [G] + ret
    return ret

class Method:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
    
    def compute_policy_loss(self, obs, act, target):
        logp = self.agent.policy_distribution(torch.as_tensor(obs, dtype=torch.float32)).log_prob(torch.as_tensor(act, dtype=torch.float32))
        return -(logp * torch.as_tensor(target, dtype=torch.float32)).mean()
    
    def compute_state_value_loss(self, obs, ret):
        return ((self.agent.state_value(torch.as_tensor(obs, dtype=torch.float32)) - torch.as_tensor(ret, dtype=torch.float32))**2).mean()

    def train(self, epoch=50, batch_size=4000, lambd=0.95, gamma=0.99, p_lr=3e-4, v_lr=1e-3, train_v_iters=80, max_ep_len=1000, render=False):
        p_optimizer = Adam(self.agent.policy_model.parameters(), lr=p_lr)
        v_optimizer = Adam(self.agent.state_value_model.parameters(), lr=v_lr)

        for epoch_i in range(epoch):

            batch_S = []          # for observations
            batch_A = []         # for actions
            batch_adv = []      # for advantage
            batch_ret = []      # for return

            batch_rets = []         # for measuring episode returns
            batch_lens = []         # for measuring episode lengths

            first_episode_rendered = False

            for i in range(batch_size):
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
                    if done or t > max_ep_len:
                        first_episode_rendered = True
                        break
                
                ret = cumulate_return(R[1:], gamma)
                v = [self.agent.state_value(s) for s in S]
                delta = [R[i] + gamma * v[i+1] - v[i] for i in range(len(v)-1)]
                adv = cumulate_return(delta, gamma*lambd)
                
                batch_ret += ret
                batch_adv += adv
                
                batch_rets += [ret[0]]
                batch_lens += [len(S)]

            p_optimizer.zero_grad()
            batch_p_loss = self.compute_policy_loss(batch_S, batch_A, batch_adv)
            batch_p_loss.backward()
            p_optimizer.step()

            for i in range(train_v_iters):
                v_optimizer.zero_grad()
                batch_v_loss = self.compute_state_value_loss(batch_S, batch_ret)
                batch_v_loss.backward()
                v_optimizer.step()

            print('epoch: %3d \t policy_loss: %.3f \t policy_loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (epoch_i, batch_p_loss, batch_v_loss, np.mean(batch_rets), np.mean(batch_lens)))

class myAgent():
    def __init__(self, env):
        self.env = env
        feature_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.n
        hidden_dims = 32
        self.policy_model = mlp([feature_dims, hidden_dims, action_dims])
        self.state_value_model = mlp([feature_dims, hidden_dims, 1])

    def policy_distribution(self, state):
        return Categorical(logits=self.policy_model(state))

    def policy(self, state, action):
        pmf = self.policy_distribution(torch.as_tensor(state, dtype=torch.float32)).probs
        return pmf[action]
    
    def policy_select(self, state):
        return Categorical(logits=self.policy_model(torch.as_tensor(state, dtype=torch.float32))).sample().item()

    def state_value(self, state):
        return self.state_value_model(torch.as_tensor(state, dtype=torch.float32)).squeeze(-1)
        

if __name__ == "__main__":
    env = gym.make('HalfCheetah-v2')
    ag = myAgent(env)
    method = Method(env, ag)
    method.train(50, 500)#, render=True)