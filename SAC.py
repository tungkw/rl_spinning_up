import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import mse_loss
from torch.distributions.normal import Normal
import torch.nn.functional as F
import gym
from utils import *
from logger import Logger
from copy import deepcopy


class ReplayBuffer():

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((int(size), int(obs_dim)), dtype=np.float32)
        self.act_buf = np.zeros((int(size), int(act_dim)), dtype=np.float32)
        self.r_buf = np.zeros((int(size)), dtype=np.float32)
        self.obs_next_buf = np.zeros((int(size), int(obs_dim)), dtype=np.float32)
        self.done_buf = np.zeros((int(size)), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, r, obs_next, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.r_buf[self.ptr] = r
        self.obs_next_buf[self.ptr] = obs_next
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     act=self.act_buf[idxs],
                     r=self.r_buf[idxs],
                     obs_next=self.obs_next_buf[idxs],
                     done=self.done_buf[idxs]
                     )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

LOG_STD_MAX = 2
LOG_STD_MIN = -20
class myAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.feature_dims = self.env.observation_space.shape[0]
        self.action_dims = self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.policy_mean = mlp([self.feature_dims, 256, 256, self.action_dims], nn.ReLU)
        self.policy_log_std = mlp([self.feature_dims, 256, 256, self.action_dims], nn.ReLU)
        self.state_value_model = mlp([self.feature_dims , 256, 256, 1], nn.ReLU)
        self.action_value_model = mlp([self.feature_dims + self.action_dims, 256, 256, 1], nn.ReLU)

    def policy_distribution(self, state):
        s = torch.as_tensor(state, dtype=torch.float32)
        mean = self.policy_mean(s)
        log_std = self.policy_log_std(s)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist

    def policy_select(self, state, with_logp=False, greedy=False):
        dist = self.policy_distribution(state)
        if greedy:
            action = self.policy_mean(f32tensor(state))
        else:
            action = dist.sample()
        # print(greedy, action)

        if with_logp:
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)

        action = torch.tanh(action)
        action = (f32tensor(self.action_high + self.action_low)
                  + action * f32tensor(self.action_high - self.action_low)) / 2

        if with_logp:
            return action, logp_pi
        else:
            return action

    def state_value(self, state):
        return self.state_value_model(torch.as_tensor(state, dtype=torch.float32)).squeeze(-1)

    def action_value(self, state, action):
        feature = torch.cat([torch.as_tensor(state, dtype=torch.float32),
                             torch.as_tensor(action, dtype=torch.float32)],dim=-1)
        return self.action_value_model(feature).squeeze(-1)


class Method:
    def __init__(self, env, test_env, agent, params, render=False):
        self.env = env
        self.test_env = test_env
        self.agent = agent
        self.params = params
        self.render = render

        # optimizer
        self.p_optimizer = Adam([{'params': self.agent.policy_mean.parameters()},
                                 {'params': self.agent.policy_log_std.parameters()}],lr=self.params['pi_lr'])
        self.v_optimizer = Adam(self.agent.state_value_model.parameters(), lr=self.params['v_lr'])
        self.q_optimizer = Adam(self.agent.action_value_model.parameters(), lr=self.params['q_lr'])

        # target model
        self.agent_c = deepcopy(self.agent)
        for paras in self.agent_c.parameters():
            paras.requires_grad = False

        # replay buffer
        self.dataset = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0],
                                    self.params['replay_size'])

        # log
        self.logger = Logger()
        print(
            "policy mean model size: {} \t policy std model size: {} \t state-value model size {} \t action-value model size {}".format(
                sum([np.prod(p.shape) for p in self.agent.policy_mean.parameters()]),
                sum([np.prod(p.shape) for p in self.agent.policy_log_std.parameters()]),
                sum([np.prod(p.shape) for p in self.agent.state_value_model.parameters()]),
                sum([np.prod(p.shape) for p in self.agent.action_value_model.parameters()])
            )
        )

    def compute_state_value_loss(self, S, target):
        V = self.agent.state_value(S)
        self.logger.log("v", V.detach().numpy())
        self.logger.log("q-pi", target.detach().numpy())
        return mse_loss(V, target)

    def compute_action_value_loss(self, S, A, target):
        Q = self.agent.action_value(S, A)
        self.logger.log("q", Q.detach().numpy())
        self.logger.log("r+v", target.detach().numpy())
        return mse_loss(Q, target)

    def compute_policy_loss(self, S):
        A, logp = self.agent.policy_select(S, with_logp=True)
        Q = self.agent.action_value(S, A)
        return (logp - Q).mean()

    def update(self, data):
        S, A, R, Sn, Done = \
            data['obs'], data['act'], data['r'], data['obs_next'], data['done']

        # state value
        with torch.no_grad():
            A_, logp = self.agent.policy_select(S, with_logp=True)
            v_target = self.agent.action_value(S, A_) - logp
        self.v_optimizer.zero_grad()
        loss_v = self.compute_state_value_loss(S, v_target)
        loss_v.backward()
        self.v_optimizer.step()

        # action value
        with torch.no_grad():
            V = self.agent_c.state_value(Sn)
            q_target = R + self.params['gamma'] * (1 - Done) * V
        self.q_optimizer.zero_grad()
        loss_q = self.compute_action_value_loss(S, A, q_target)
        loss_q.backward()
        self.q_optimizer.step()

        # policy
        for paras in self.agent.action_value_model.parameters():
            paras.requires_grad = False
        self.p_optimizer.zero_grad()
        loss_p = self.compute_policy_loss(S)
        loss_p.backward()
        self.p_optimizer.step()
        for paras in self.agent.action_value_model.parameters():
            paras.requires_grad = True

        self.logger.log("loss_q", loss_q.item())
        self.logger.log("loss_p", loss_p.item())

        # target model
        with torch.no_grad():
            for model_para, target_model_para in zip(self.agent.parameters(),
                                                     self.agent_c.parameters()):
                target_model_para.data.mul_(self.params['polyak'])
                target_model_para.data.add_((1 - self.params['polyak']) * model_para.data)

    def test(self):
        def get_action(s):
            with torch.no_grad():
                action, logp = self.agent.policy_select(s, with_logp=True, greedy=True)
                return action.detach().numpy(), logp.detach().numpy()
        first_ep_rendered = False
        for i in range(self.params['num_test_episodes']):
            R = []
            # episole
            for s, a, r, sn, _, done in run_episole(self.test_env, get_action, self.params['max_ep_len'],
                                                    self.render and not first_ep_rendered):
                R.append(r)
            first_ep_rendered = True
            self.logger.log("rets", np.sum(R).item())
            self.logger.log("lens", len(R))

    def train(self):

        total_step = 0
        for epoch_i in range(self.params['epochs']):

            epoch_step = 0
            while epoch_step < self.params['steps_per_epoch']:

                def get_action(s):
                    if total_step > 0:#self.params['start_steps']:
                        with torch.no_grad():
                            action, logp = self.agent.policy_select(s, with_logp=True, greedy=True)
                            action, logp = action.detach().numpy(), logp.detach().numpy()
                        action += self.params['act_noise'] * np.random.randn(self.agent.action_dims)
                        action = np.clip(action, self.agent.action_low, self.agent.action_high)
                        return action, logp
                    else:
                        action = self.env.action_space.sample()
                        logp = 1 / (self.env.action_space.high - self.env.action_space.low)
                        logp = np.log(np.prod(logp, axis=-1))
                        return action, logp

                # episole
                ep_len = 0
                for s, a, r, sn, _, done in run_episole(self.env, get_action, self.params['max_ep_len']):
                    ep_len += 1
                    # ignore the done cause by hitting the env time horizon
                    done = False if ep_len == self.params['max_ep_len'] else done
                    self.dataset.store(s, a, r, sn, done)

                    total_step += 1
                    epoch_step += 1

                    # train
                    if total_step >= self.params['update_after'] and total_step % self.params['update_every'] == 0:
                        for i in range(self.params['update_every']):
                            batch_data = self.dataset.sample_batch(batch_size=self.params['batch_size'])
                            self.update(batch_data)

            # test
            self.test()

            # print msg
            map = {'mean': np.mean, 'max': np.max, 'min': np.min, 'std': np.std}
            ops = {
                "loss_p": ["mean"],
                "loss_q": ["mean"],
                "rets": ["mean"],
                "lens": ['mean'],
                'v': ['mean', 'max', 'min', 'std'],
                'q-pi': ['mean', 'max', 'min', 'std'],
                'q': ['mean', 'max', 'min', 'std'],
                'r+v': ['mean', 'max', 'min', 'std'],
            }
            self.logger.show("epoch: \t {:3d}".format(epoch_i), ops, map)


if __name__ == '__main__':
    # env = gym.make("MountainCarContinuous-v0")
    env = gym.make("HalfCheetah-v2")
    test_env = gym.make("HalfCheetah-v2")
    ag = myAgent(env)
    params = dict(steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, pi_lr=1e-3, v_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, noise=0.1,
         max_ep_len=1000)
    method = Method(env, test_env, ag, params)#, render=True)
    method.train()
