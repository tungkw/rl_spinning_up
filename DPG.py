import numpy as np
import gym
import torch
from torch.distributions.normal import Normal
from torch.optim import Adam

from utils.utils import cumulate_return, run_episole


class myAgent():
    def __init__(self, env):
        self.env = env
        feature_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]
        self.theta = torch.nn.Parameter(torch.rand((64), dtype=torch.float32))
        self.theta_mean = torch.nn.Parameter(torch.rand((64), dtype=torch.float32))
        self.theta_std = torch.nn.Parameter(torch.rand((64), dtype=torch.float32))
        self.w = torch.nn.Parameter(torch.rand((64,1), dtype=torch.float32))

    def policy_distribution(self, state):
        mean = torch.dot(torch.as_tensor(state, dtype=torch.float32), self.theta_mean)
        mean = torch.dot(torch.as_tensor(state, dtype=torch.float32), self.theta_std)
        Normal()
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

    def compute_policy_loss(self, obs, act, target):
        logp = self.agent.policy_distribution(torch.as_tensor(obs, dtype=torch.float32)).log_prob(
            torch.as_tensor(act, dtype=torch.float32))
        return -(logp * torch.as_tensor(target, dtype=torch.float32)).mean()

    def compute_state_value_loss(self, obs, ret):
        return ((self.agent.state_value(obs) - torch.as_tensor(ret, dtype=torch.float32)) ** 2).mean()

    def train(self, epoch=50, batch_size=4000, lambd=0.97, gamma=0.99, p_lr=1e-2, v_lr=1e-3, train_v_iters=80,
              max_ep_len=1000, render=False):
        p_optimizer = Adam(self.agent.policy_model.parameters(), lr=p_lr)
        v_optimizer = Adam(self.agent.state_value_model.parameters(), lr=v_lr)
        print(
            "policy model size: {} state value model size {}".format(
                sum([np.prod(p.shape) for p in self.agent.policy_model.parameters()]),
                sum([np.prod(p.shape) for p in self.agent.state_value_model.parameters()])
            )
        )

        for epoch_i in range(epoch):

            batch_S = []  # for observations
            batch_A = []  # for actions
            batch_adv = []  # for advantage
            batch_ret = []  # for return

            batch_rets = []  # for measuring episode returns
            batch_lens = []  # for measuring episode lengths

            first_episode_rendered = False

            with torch.no_grad():

                while len(batch_S) < batch_size:

                    S, A, R, _, done = run_episole(self.env, self.agent, 1000, render and not first_episode_rendered)
                    if not first_episode_rendered:
                        first_episode_rendered = True

                    ret = cumulate_return(R[1:], gamma)
                    v = [self.agent.state_value(s) for s in S]
                    delta = [R[i + 1] + gamma * v[i + 1] - v[i] for i in range(len(v) - 1)]
                    adv = cumulate_return(delta, gamma * lambd)

                    batch_S += S[:-1]
                    batch_A += A[:-1]
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

            print('epoch: %3d \t policy_loss: %.3f \t state_value_loss: %.3f \t return: %.3f \t ep_len: %.3f' %
                  (epoch_i, batch_p_loss, batch_v_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == "__main__":
    env = gym.make('MountainCarContinuous-v0')
    ag = myAgent(env)
    method = Method(env, ag)
    method.train()  # , render=True)