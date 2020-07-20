import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.uniform import Uniform
from torch.nn.functional import mse_loss
import gym
from utils import *
from copy import deepcopy

class myAgent():
    def __init__(self, env):
        self.env = env
        self.feature_dims = self.env.observation_space.shape[0]
        self.action_dims = self.env.action_space.shape[0]
        self.action_low = torch.as_tensor(self.env.action_space.low, dtype=torch.float32)
        self.action_high = torch.as_tensor(self.env.action_space.high, dtype=torch.float32)
        self.policy_model = mlp([self.feature_dims, 256, 256, self.action_dims], output_activation=nn.Tanh)
        self.action_value_model = mlp([self.feature_dims + self.action_dims, 256, 256, 1])
        self.act_noise = 0.1

    def policy_select_greedy(self, state):
        action = self.policy_model(torch.as_tensor(state, dtype=torch.float32))
        action = action * (self.action_high - self.action_low) + self.action_low
        return action

    def select_distribution(self, state):
        dist = Uniform(self.action_low, self.action_high)
        return dist

    def policy_select(self, state):
        dist = self.select_distribution(state)
        action = dist.sample()
        action = action * (self.action_high - self.action_low) + self.action_low
        action += self.act_noise * torch.randn(self.action_dims)
        action = torch.clamp(action, self.action_low.item(), self.action_high.item())
        p_a = dist.log_prob(action)
        return action.numpy(), p_a.item()

    def action_value(self, state, action):
        feature = torch.cat([torch.as_tensor(state, dtype=torch.float32), torch.as_tensor(action, dtype=torch.float32)], dim=-1)
        return self.action_value_model(feature).squeeze(-1)


class Method:
    def __init__(self, env, agent, p_lr=3e-4, v_lr=1e-3):
        self.env = env
        self.agent = agent
        # target model
        self.agent_c = deepcopy(self.agent)

        self.p_optimizer = Adam(self.agent.policy_model.parameters(), lr=p_lr)
        self.q_optimizer = Adam(self.agent.action_value_model.parameters(), lr=v_lr)
        print(
            "policy model size: {} \t state-value model size {}".format(
                sum([np.prod(p.shape) for p in self.agent.policy_model.parameters()]),
                sum([np.prod(p.shape) for p in self.agent.action_value_model.parameters()])
            )
        )

    def compute_policy_loss(self, S, A):
        return - self.agent.action_value(S, A).mean()

    def compute_action_value_loss(self, S, A, target):
        return mse_loss(self.agent.action_value(S, A), torch.as_tensor(target, dtype=torch.float32))

    def test(self):
        for i in range(10):
            pass

    def get_batch(self):
        return [],[],[]

    def upda(self, polyak=0.995):
        batch_S, batch_A, batch_Y = self.get_batch()
        # value function iteration
        self.q_optimizer.zero_grad()
        batch_q_loss = self.compute_action_value_loss(batch_S, batch_A, batch_Y)
        batch_q_loss.backward()
        self.q_optimizer.step()

        # policy iteration
        # todo : do not compute the grad of action_value_model for efficiency
        self.p_optimizer.zero_grad()
        batch_p_loss = self.compute_policy_loss(batch_S, batch_A)
        batch_p_loss.backward()
        self.p_optimizer.step()

        # target model
        with torch.no_grad():
            for model_para, target_model_para in zip(self.agent.policy_model.parameters(),
                                                     self.agent_c.policy_model.parameters()):
                target_model_para.data.mul_(polyak)
                target_model_para.data.add_((1 - polyak) * model_para)
            for model_para, target_model_para in zip(self.agent.action_value_model.parameters(),
                                                     self.agent_c.action_value_model.parameters()):
                target_model_para.data.mul_(polyak)
                target_model_para.data.add_((1 - polyak) * model_para.data)

    def train(self, epoch=50, batch_size=4000, lambd=0.97, gamma=0.99, , polyak=0.995, \
              train_policy_iters=80, train_v_iters=80, target_kl=0.01, max_ep_len=1000, render=False):

        total_S = []
        total_A = []
        total_Y = []

        for epoch_i in range(epoch):

            batch_S = []  # for observations
            batch_A = []  # for actions
            batch_Y = []  # for Q learning target

            batch_rets = []  # for measuring episode returns
            batch_lens = []  # for measuring episode lengths

            first_episode_rendered = False

            with torch.no_grad():

                while len(batch_S) < batch_size:

                    S, A, R, _, done = run_episole(self.env, self.agent, max_ep_len,
                                                     render and not first_episode_rendered)
                    if not first_episode_rendered:
                        first_episode_rendered = True

                    G = cumulate_return(R[1:], gamma)
                    Y = [R[i+1] + gamma * agent_c.action_value([S[i+1]], agent_c.policy_select_greedy([S[i+1]])).item()
                         if i < len(S)-2 else R[i+1]
                         for i in range(len(S)-1)]\
                        + [0]

                    batch_S += S
                    batch_A += A
                    batch_Y += Y

                    batch_rets += [G]
                    batch_lens += [len(S)]


            print('epoch: %3d \t policy_loss: %.3f \t action_value_loss: %.3f \t mean return: %.3f \t mean ep_len: %.3f' %
                  (epoch_i, batch_p_loss, batch_q_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")
    # env = gym.make("HalfCheetah-v2")
    ag = myAgent(env)
    method = Method(env, ag)
    method.train()#render=True)