import agent
import algo
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

class myAgent(agent.Agent):
    def __init__(self):
        super().__init__(discount=1.0)
        
        self.env = gym.make('CartPole-v0')
        self.done = False

        feature_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.n
        hidden_dims = 32
        self.policy_model = get_policy_model([feature_dims, hidden_dims, action_dims])

        self.optimizer = Adam(self.policy_model.parameters(), lr=1e-2)
    
    def feature(self, state, action):
        return np.array([[0, 1],[1, 0]],dtype=float)[:,int(action)].reshape(2,1)

    def new_episode(self):
        self.done = False
        return self.env.reset()

    def policy_distribution(self, state):
        return Categorical(logits=self.policy_model(state))

    def policy(self, state, action):
        pmf = self.policy_distribution(torch.as_tensor(state, dtype=torch.float32)).probs
        return pmf[action]
    
    def policy_select(self, state):
        return Categorical(logits=self.policy_model(torch.as_tensor(state, dtype=torch.float32))).sample().item()

    def update(self, t, state, action, target):
        pass

    def act(self, state, action):
        new_state, reward, self.done, _ = self.env.step(action)
        return new_state, reward
    
    def stop_state(self, state):
        return self.done

    def print_t(self,t,St,At,Rtn,Stn,Atn,Gt):
        # x = self.feature(state, action)
        
        # diff = target
        # # diff = target - np.sum([self.policy(state, i) * self.action_value(state, i) for i in range(2)])
        # grad_ln_p = x - np.sum([self.feature(state, i)*self.policy(state, i) for i in range(2)])
        # self.z_policy = self.discount * self.lambd_policy * self.z_policy + (self.discount**t) * grad_ln_p
        # self.theta += self.step_size_policy * diff * self.z_policy
        pass

    def print_e(self,e,S,A,R):
        self.optimizer.zero_grad()
        As = torch.as_tensor(A, dtype=torch.int32)
        logp = self.policy_distribution(torch.as_tensor(S, dtype=torch.float32)).log_prob(As)
        Rs = torch.as_tensor([np.sum(R)] * len(S), dtype=torch.float32)
        batch_loss = -(logp * Rs).mean()
        batch_loss.backward()
        self.optimizer.step()
        

if __name__ == "__main__":
    ag = myAgent()
    method = algo.Method(ag)
    method.learn(1, step=float('inf'))