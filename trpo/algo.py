import numpy as np
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
from utils.logger import Logger


class Algo:
    def __init__(self, env, agent, buffer, params):
        self.env = env
        self.agent = agent
        self.params = params
        self.buffer = buffer
        self.logger = Logger()
        self.optimizer_p = Adam(self.agent.policy_model.parameters(), lr=self.params['lr_p'])
        self.optimizer_v = Adam(self.agent.state_value_model.parameters(), lr=self.params['lr_v'])
        print("Vanilla Policy Gradient\n"
              "policy model size {}\n"
              "state-value model size {}".format(
            np.sum([np.prod(p.data.shape) for p in self.agent.policy_model.parameters()]),
            np.sum([np.prod(p.data.shape) for p in self.agent.state_value_model.parameters()])
        ))

    def policy_loss(self, obs, a, target):
        dist = self.agent.policy_distribution(obs)
        logp = dist.log_prob(a)
        return -(logp * target).mean()

    def state_value_loss(self, obs, ret):
        return mse_loss(self.agent.state_value(obs), ret)

    def get_action(self, obs):
        with torch.no_grad():
            a, logp = self.agent.policy_sample(torch.as_tensor([obs], dtype=torch.float32))
            return a.item(), logp.item()

    def update(self, data):
        obs, a, ret, target = data['obs'], data['a'], data['ret'], data['target']

        self.optimizer_p.zero_grad()
        loss_p = self.policy_loss(obs, a, target)
        loss_p.backward()
        self.optimizer_p.step()

        for i in range(self.params['train_v_iters']):
            self.optimizer_v.zero_grad()
            loss_v = self.state_value_loss(obs, ret)
            loss_v.backward()
            self.optimizer_v.step()

        self.logger.log("loss_p", loss_p.detach().item())
        self.logger.log("loss_v", loss_v.detach().item())

    def train(self, ):

        for epoch_i in range(self.params['epochs']):

            # init status
            first_episode_rendered = False
            o = self.env.reset()
            done = False
            r_ep = []
            t_ep = 0
            while self.buffer.size < self.params["batch_size"]:
                t_ep += 1

                a, logp = self.get_action(o)
                o_next, r, done, _ = self.env.step(a)
                done = False if t_ep > self.params['ep_max_len'] else done
                if not first_episode_rendered and self.params['render']:
                    self.env.render()

                # save step data
                with torch.no_grad():
                    v = self.agent.state_value(torch.as_tensor([o], dtype=torch.float32)).item()
                    self.buffer.store(o, a, r, v)
                    r_ep.append(r)

                # end episode
                if done or t_ep > self.params['ep_max_len']:
                    # compute episode return
                    self.buffer.ep_done()
                    self.logger.log('rets', np.sum(r_ep))
                    self.logger.log('lens', len(r_ep))
                    # update all status
                    first_episode_rendered = True
                    o = self.env.reset()
                    r_ep = []
                    t_ep = 0
                else:
                    o = o_next
            if not done:
                self.buffer.ep_done()

            # loss and backward
            batch_data = self.buffer.sample(self.params['batch_size'])
            self.update(batch_data)

            # print log msg
            if self.params['show_log']:
                msg = "epoch: \t {:3d}".format(epoch_i)
                ops = {
                    "loss_p": ["mean"],
                    "loss_v": ["mean"],
                    "rets": ["mean"],
                    "lens": ['mean'],
                }
                map = {'mean': np.mean}
                self.logger.show(msg, ops, map)