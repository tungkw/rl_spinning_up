import torch
import torch.nn as nn
from torch.optim import Adam
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
        self.policy_model = mlp([self.feature_dims, 256, 256, self.action_dims], activation=nn.ReLU, output_activation=nn.Tanh)
        self.action_value_model = mlp([self.feature_dims + self.action_dims, 256, 256, 1], activation=nn.ReLU)

    def policy_select(self, state):
        action = self.policy_model(torch.as_tensor(state, dtype=torch.float32))
        # action = action * (self.action_high - self.action_low) + self.action_low
        action = action * self.action_high
        return action

    def action_value(self, state, action):
        feature = torch.cat([torch.as_tensor(state, dtype=torch.float32), torch.as_tensor(action, dtype=torch.float32)], dim=-1)
        return self.action_value_model(feature).squeeze(-1)

    def sample_select(self, state):
        action = self.env.action_space.sample()
        logp = np.log(1/(self.action_high - self.action_low))
        return action, logp

    def greedy_select(self, state):
        with torch.no_grad():
            action = self.policy_select(state)
            logp = np.log([1.0] * self.action_dims)
            return action.numpy(), logp

class ReplayBuffer():

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((int(size), int(obs_dim)), dtype=np.float32)
        self.obs_next_buf = np.zeros((int(size), int(obs_dim)), dtype=np.float32)
        self.act_buf = np.zeros((int(size), int(act_dim)), dtype=np.float32)
        self.r_buf = np.zeros((int(size)), dtype=np.float32)
        self.done_buf = np.zeros((int(size)), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, obs_next, act, r, done):
        self.obs_buf[self.ptr] = obs
        self.obs_next_buf[self.ptr] = obs_next
        self.act_buf[self.ptr] = act
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs_next=self.obs_next_buf[idxs],
                     act=self.act_buf[idxs],
                     r=self.r_buf[idxs],
                     done=self.done_buf[idxs]
                     )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

class Method:
    def __init__(self, env, agent, p_lr=1e-3, v_lr=1e-3):
        self.env = env
        self.agent = agent

        # target model
        self.agent_c = deepcopy(self.agent)
        for paras in self.agent_c.policy_model.parameters():
            paras.requires_grad = False
        for paras in self.agent_c.action_value_model.parameters():
            paras.requires_grad = False
        # replay buffer
        self.dataset = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], int(1e6))
        # stage sampling
        self.random_sample = self.agent.sample_select
        noise_scale = 0.1
        def func(s):
            with torch.no_grad():
                action, logp = self.agent.greedy_select(s)
            action += noise_scale * np.random.randn(self.agent.action_dims)
            action = np.clip(action, self.agent.action_low.numpy(),
                             self.agent.action_high.numpy())
            return action, logp
        self.noisy_greedy_sample = func

        self.p_optimizer = Adam(self.agent.policy_model.parameters(), lr=p_lr)
        self.q_optimizer = Adam(self.agent.action_value_model.parameters(), lr=v_lr)
        print(
            "policy model size: {} \t state-value model size {}".format(
                sum([np.prod(p.shape) for p in self.agent.policy_model.parameters()]),
                sum([np.prod(p.shape) for p in self.agent.action_value_model.parameters()])
            )
        )

    def compute_policy_loss(self, S):
        A = self.agent.policy_select(S)
        return - self.agent.action_value(S, A).mean()

    def compute_action_value_loss(self, S, A, target):
        return mse_loss(self.agent.action_value(S, A), torch.as_tensor(target, dtype=torch.float32))

    def test(self, max_ep_len=1000, gamma=0.99, render=False):
        test_size = 10
        rets = []  # for measuring episode returns
        lens = []  # for measuring episode lengths
        loss_ps = []
        loss_qs = []

        for i in range(test_size):

            S = []
            Sn = []
            A = []
            R = []
            Done = []

            # episole
            for s, a, r, sn, an, _, _, done in run_episole(self.env, self.agent.greedy_select, max_ep_len, render):
                self.dataset.store(s, sn, a, r, done)
                S.append(s)
                Sn.append(sn)
                A.append(a)
                R.append(r)
                Done.append(done)

            S = torch.as_tensor(S, dtype=torch.float32)
            Sn = torch.as_tensor(Sn, dtype=torch.float32)
            A = torch.as_tensor(A, dtype=torch.float32)
            R = torch.as_tensor(R, dtype=torch.float32)
            Done = torch.as_tensor(Done, dtype=torch.float32)

            with torch.no_grad():
                Q = self.agent_c.action_value(Sn, self.agent_c.policy_select(Sn))
                Y = R + gamma * (1-Done) * Q
                loss_q = self.compute_action_value_loss(S, A, Y)
                loss_p = self.compute_policy_loss(S)
                rets.append(torch.sum(R).item())
                lens.append(len(S))
                loss_ps.append(loss_p.item())
                loss_qs.append(loss_q.item())
        return np.mean(loss_ps), np.mean(loss_qs), np.mean(rets), np.mean(lens)

    def update(self, data, gamma=0.99, polyak=0.995):
        S, Sn, A, R, Done = \
            data['obs'], data['obs_next'], data['act'], data['r'], data['done']

        # value function iteration
        Q = self.agent_c.action_value(Sn, self.agent_c.policy_select(Sn))
        Y = R + gamma * (1-Done) * Q
        self.q_optimizer.zero_grad()
        loss_q = self.compute_action_value_loss(S, A, Y)
        loss_q.backward()
        self.q_optimizer.step()

        # policy iteration
        for paras in self.agent.action_value_model.parameters():
            paras.requires_grad = False
        self.p_optimizer.zero_grad()
        loss_p = self.compute_policy_loss(S)
        loss_p.backward()
        self.p_optimizer.step()
        for paras in self.agent.action_value_model.parameters():
            paras.requires_grad = True

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

    def train(self, epoch=100, step_per_epoch=4000, batch_size=100, gamma=0.99, polyak=0.995, max_ep_len=1000,
              update_after=1000, update_every=50, start_steps=10000, render=False):

        sample_func = self.random_sample

        total_step = 0
        for epoch_i in range(epoch):

            epoch_step = 0
            while epoch_step < step_per_epoch:

                # episole
                for s, a, r, sn, an, _, _, done in run_episole(self.env, sample_func, max_ep_len, render):
                    self.dataset.store(s, sn, a, r, done)

                    total_step += 1
                    epoch_step += 1

                    # train
                    if total_step > update_after and total_step % update_every == 0:
                        for i in range(update_every):
                            data = self.dataset.sample_batch(batch_size=batch_size)
                            self.update(data, polyak=polyak)
                    if total_step > start_steps:
                        sample_func = self.noisy_greedy_sample

            # test
            mean_p_loss, mean_q_loss, mean_batch_rets, mean_batch_lens = self.test(max_ep_len=max_ep_len, gamma=gamma, render=render)
            print('epoch: %3d \t policy_loss: %.3f \t action_value_loss: %.3f \t mean return: %.3f \t mean ep_len: %.3f' %
                  (epoch_i, mean_p_loss, mean_q_loss, mean_batch_rets, mean_batch_lens))



if __name__ == '__main__':
    # env = gym.make("MountainCarContinuous-v0")
    env = gym.make("HalfCheetah-v2")
    ag = myAgent(env)
    method = Method(env, ag)
    method.train()#render=True)