import gym
import torch
from torch.distributions.categorical import Categorical

from pg.algo import Algo as PG
from utils.utils import *

class Agent(torch.nn.Module):
    def __init__(self, env, params):
        super(Agent, self).__init__()
        self.env = env
        self.obs_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.n
        self.hidden_list = [params['hidden_dim']] * params['layers']
        self.policy_model = mlp([self.obs_dims] + self.hidden_list + [self.action_dims])

    def policy_distribution(self, obs):
        return Categorical(logits=self.policy_model(obs))

    def policy_sample(self, obs):
        dist = self.policy_distribution(obs)
        action = dist.sample()
        p_a = dist.log_prob(action)
        return action, p_a


class Buffer:
    def __init__(self, env, params):
        self.env = env
        self.obs_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.n
        self.max_size = params['batch_size']
        self.gamma = params['gamma']
        self.obs_buf = np.zeros((self.max_size, self.obs_dims), dtype=np.float32)
        self.a_buf = np.zeros((self.max_size), dtype=np.float32)
        self.r_buf = np.zeros((self.max_size), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.ep_start = 0

    def store(self, o, a, r):
        self.obs_buf[self.ptr] = o
        self.a_buf[self.ptr] = a
        self.r_buf[self.ptr] = r
        self.ptr += 1
        self.size += 1

    def ep_done(self):
        # discounted reward-to-go return
        self.r_buf[self.ep_start:self.ptr] = cumulate_return(self.r_buf[self.ep_start:self.ptr], self.gamma)
        self.ep_start = self.ptr

    def sample(self, batch_size):
        batch = dict(
            obs=self.obs_buf,
            a=self.a_buf,
            ret=self.r_buf,
        )
        self.ptr = 0
        self.size = 0
        self.ep_start = 0
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

if __name__ == "__main__":
    params = dict(
        task="CartPole-v0",
        hidden_dim=32,
        layers=1,
        epochs=50,
        batch_size=4000,
        gamma=0.99,
        lr=1e-2,
        render=False,
        show_log=True,
        # render=True,
        # show_log=False,
    )
    env = gym.make(params['task'])
    ag = Agent(env, params)
    buffer = Buffer(env, params)
    pg = PG(env, ag, buffer, params)
    pg.train()