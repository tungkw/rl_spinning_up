import numpy as np
import gym

import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.nn.functional import mse_loss

from utils.utils import mlp, cumulate_return, run_episole


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
    
    def compute_policy_loss(self, S, A, sample_logp, target):
        e = 0.2
        new_logp = self.agent.policy_logp(S, A)
        ratio = (new_logp - torch.as_tensor(sample_logp, dtype=torch.float32)).exp()
        loss = torch.as_tensor(ratio, dtype=torch.float32) * torch.as_tensor(target, dtype=torch.float32)
        loss_clip = torch.clamp(ratio, 1-e, 1+e) * torch.as_tensor(target, dtype=torch.float32)
        return -torch.min(loss, loss_clip).mean()
    
    def compute_state_value_loss(self, obs, ret):
        return mse_loss(self.agent.state_value(obs), torch.as_tensor(ret, dtype=torch.float32))

    def train(self, epoch=50, batch_size=4000, lambd=0.97, gamma=0.99, p_lr=3e-4, v_lr=1e-3, \
        train_policy_iters=80, train_v_iters=80, target_kl=0.01, max_ep_len=1000, render=False):
        
        p_optimizer = Adam(self.agent.policy_model.parameters(), lr=p_lr)
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

                    S = []
                    A = []
                    R = []
                    Sn = []
                    P_A = []
                    Done = []
                    # ts = time.time()
                    for s,a,r,sn,logp,done in run_episole(self.env, self.agent.policy_select,
                                                           max_ep_len, render and not first_episode_rendered):
                        S.append(s)
                        A.append(a)
                        R.append(r)
                        Sn.append(sn)
                        P_A.append(logp)
                        Done.append(done)

                    # print("time", time.time() - ts)
                    if not first_episode_rendered:
                        first_episode_rendered = True

                    ret = cumulate_return(R, gamma)
                    v_s = self.agent.state_value(S).numpy()
                    v_sn = self.agent.state_value(Sn).numpy()
                    delta = np.array(R) + gamma * v_sn - v_s
                    adv = cumulate_return(delta.tolist(), gamma*lambd)
                    
                    batch_S += S
                    batch_A += A
                    batch_P_A += P_A
                    batch_ret += ret
                    batch_adv += adv
                    # print(len(batch_S),len(batch_A),len(batch_P_A),len(batch_ret),len(batch_adv))

                    batch_rets += [ret[0]]
                    batch_lens += [len(S)]

            # policy iteration
            batch_p_loss = self.compute_policy_loss(batch_S, batch_A, batch_P_A, batch_adv)
            old_logp = torch.as_tensor(batch_P_A, dtype=torch.float)
            for i in range(train_policy_iters):
                p_optimizer.zero_grad()
                new_loss = self.compute_policy_loss(batch_S, batch_A, batch_P_A, batch_adv)
                new_loss.backward()
                p_optimizer.step()

                new_logp = self.agent.policy_logp(batch_S, batch_A)
                kl = (old_logp - new_logp).mean()
                if kl > 1.5 * target_kl:
                    print("step {} early break in policy iteration".format(i))
                    break

            # value function iteration
            # print(len(batch_S), len(batch_ret))
            batch_v_loss = self.compute_state_value_loss(batch_S, batch_ret)
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