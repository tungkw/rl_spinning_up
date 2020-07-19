import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gym

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torch.optim import Adam
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from utils import mlp, cumulate_return, run_episole

class myAgent():
    def __init__(self, env):
        self.env = env
        feature_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.n
        self.policy_model = mlp([feature_dims, 64, 64, action_dims])
        self.state_value_model = mlp([feature_dims, 64, 64, 1])

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
        return self.state_value_model(torch.as_tensor(state, dtype=torch.float32)).squeeze(-1)
    

class Method:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
    
    def compute_policy_loss(self, S, A, target, sample_logp):
        new_logp = self.agent.policy_logp(S, A)
        ratio = torch.exp(new_logp - torch.as_tensor(sample_logp, dtype=torch.float32))
        return -(torch.as_tensor(ratio, dtype=torch.float32) * torch.as_tensor(target, dtype=torch.float32)).mean()
    
    def compute_state_value_loss(self, obs, ret):
        return ((self.agent.state_value(obs) - torch.as_tensor(ret, dtype=torch.float32))**2).mean()
    
    def policy_update(self, S, A, sample_logp, target, cg_iters, step, delta, damping_coeff=0.1):
        with torch.no_grad():
            old_logp = self.agent.policy_logp(S, A)
            old_policy = self.agent.policy_distribution(S)

        old_paras = parameters_to_vector(self.agent.policy_model.parameters())
        old_loss = self.compute_policy_loss(S, A, target, sample_logp)
        g = parameters_to_vector(torch.autograd.grad(old_loss, self.agent.policy_model.parameters(), retain_graph=True))

        kl = kl_divergence(old_policy, self.agent.policy_distribution(S)).mean()
        # commput hessian-vector product
        def Hx(x):
            g1 = parameters_to_vector(torch.autograd.grad(kl, self.agent.policy_model.parameters(), create_graph=True))
            hvp = parameters_to_vector(torch.autograd.grad((g1*x.detach()).sum(), self.agent.policy_model.parameters(), retain_graph=True))
            if damping_coeff > 0:
                hvp += damping_coeff * x
            return hvp
        # conjugate gradient
        def cg(Ax, b):
            x = torch.zeros_like(b)
            r = b.clone()
            p = b.clone()
            r_dot_old = torch.dot(r, r)
            for _ in range(cg_iters):
                z = Ax(p)
                alpha = r_dot_old / (torch.dot(p, z) + 0.00001)
                x += alpha * p
                r -= alpha * z
                r_dot_new = torch.dot(r, r)
                p = r + (r_dot_new / r_dot_old) * p
                r_dot_old = r_dot_new
            return x
        # search direction
        x = cg(Hx, g)

        # max step
        alpha = torch.sqrt(2 * delta / (torch.dot(x, g) + 0.00001)) * x
        for i in range(step):
            # minimizing loss rather than maximizing surrogate function, so the minus rather than plus
            vector_to_parameters(old_paras - 0.8**i * alpha, self.agent.policy_model.parameters())
            kl = kl_divergence(old_policy, self.agent.policy_distribution(S)).mean()
            new_loss = self.compute_policy_loss(S, A, target, sample_logp)
            # print(i, old_loss, new_loss, kl)
            if new_loss <= old_loss and kl <= delta:
                break
            if i == step-1:
                vector_to_parameters(old_paras, self.agent.policy_model.parameters())

        return old_loss

    def train(self, epoch=50, batch_size=4000, lambd=0.97, gamma=0.99, v_lr=1e-3, train_v_iters=80, max_ep_len=1000, render=False):
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
            batch_P_A = []      # for selection policy prob
            batch_adv = []      # for advantage
            batch_ret = []      # for return

            batch_rets = []         # for measuring episode returns
            batch_lens = []         # for measuring episode lengths

            first_episode_rendered = False

            with torch.no_grad():

                while len(batch_S) < batch_size:

                    S, A, R, P_A, done = run_episole(self.env, self.agent, max_ep_len, (not first_episode_rendered) and render)
                    if not first_episode_rendered:
                        first_episode_rendered = True

                    ret = cumulate_return(R[1:], gamma)
                    v = [self.agent.state_value(s) for s in S]
                    delta = [R[i+1] + gamma * v[i+1] - v[i] for i in range(len(v)-1)]
                    adv = cumulate_return(delta, gamma*lambd)
                    
                    batch_S += S[:-1]
                    batch_A += A[:-1]
                    batch_P_A += P_A[:-1]
                    batch_ret += ret
                    batch_adv += adv
                    
                    batch_rets += [ret[0]]
                    batch_lens += [len(S)]
            

            # conjugate gradient policy improvement
            batch_p_loss = self.policy_update(batch_S, batch_A, batch_P_A, batch_adv, cg_iters=10, step=10, delta=0.01)

            # value function iteration
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