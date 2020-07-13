import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gym

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from utils import mlp, cumulate_return, run_episole, parameters_assignment

class myAgent():
    def __init__(self, env):
        self.env = env
        feature_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.n
        self.policy_model = mlp([feature_dims, 64, 64, action_dims])
        self.state_value_model = mlp([feature_dims, 64, 64, 1])

    def policy_distribution(self, state):
        return Categorical(logits=self.policy_model(torch.as_tensor(state, dtype=torch.float32)))

    def policy_p(self, state, action):
        return self.policy_distribution(state).log_prob(torch.as_tensor(action, dtype=torch.float32)).exp()
    
    def policy_select(self, state):
        return self.policy_distribution(state).sample().item()

    def state_value(self, state):
        return self.state_value_model(torch.as_tensor(state, dtype=torch.float32)).squeeze(-1)
    
        

class Method:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
    
    def compute_policy_loss(self, ratio, target):
        return -(ratio * target).mean()
    
    def compute_state_value_loss(self, obs, ret):
        return ((self.agent.state_value(obs) - torch.as_tensor(ret, dtype=torch.float32))**2).mean()

    def train(self, epoch=50, batch_size=4000, lambd=0.97, gamma=0.99, p_lr=1e-2, v_lr=1e-3, train_v_iters=80, max_ep_len=1000, render=False):
        # p_optimizer = Adam(self.agent.policy_model.parameters(), lr=p_lr)
        v_optimizer = Adam(self.agent.state_value_model.parameters(), lr=v_lr)
        print(
            "policy model size: {} \t state-value model size {}".format(
            sum([np.prod(p.shape) for p in self.agent.policy_model.parameters()]),
            sum([np.prod(p.shape) for p in self.agent.state_value_model.parameters()])
            )
        )

        for epoch_i in range(epoch):

            batch_S = []          # for observations
            batch_A = []         # for actions
            batch_adv = []      # for advantage
            batch_ret = []      # for return

            batch_rets = []         # for measuring episode returns
            batch_lens = []         # for measuring episode lengths

            first_episode_rendered = False

            with torch.no_grad():

                while len(batch_S) < batch_size:

                    S,A,R,done = run_episole(self.env, self.agent, max_ep_len, not first_episode_rendered)
                    
                    ret = cumulate_return(R[1:], gamma)
                    v = [self.agent.state_value(s) for s in S]
                    delta = [R[i+1] + gamma * v[i+1] - v[i] for i in range(len(v)-1)]
                    adv = cumulate_return(delta, gamma*lambd)
                    
                    batch_S += S[:-1]
                    batch_A += A[:-1]
                    batch_ret += ret
                    batch_adv += adv
                    
                    batch_rets += [ret[0]]
                    batch_lens += [len(S)]
            

            # conjugate gradient policy improvement
            # p_optimizer.zero_grad()
            rate = None
            batch_p_loss = self.compute_policy_loss(rate, batch_adv)
            batch_p_loss.backward()
            grad = batch_p_loss.parameters.grad()
            parameters_assignment()
            # p_optimizer.step()

            # value function iteration
            print(np.max(batch_ret))
            for i in range(train_v_iters):
                v_optimizer.zero_grad()
                batch_v_loss = self.compute_state_value_loss(batch_S, batch_ret)
                batch_v_loss.backward()
                v_optimizer.step()

            print('epoch: %3d \t policy_loss: %.3f \t state_value_loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (epoch_i, batch_p_loss, batch_v_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    ag = myAgent(env)
    method = Method(env, ag)
    method.train()#render=True)