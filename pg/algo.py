import numpy as np
import torch
from torch.optim import Adam
from utils.logger import Logger


class Algo:
    def __init__(self, env, agent, buffer, params):
        self.env = env
        self.agent = agent
        self.params = params
        self.buffer = buffer
        self.logger = Logger()
        self.optimizer = Adam(self.agent.parameters(), lr=self.params['lr'])
        print("Simple Policy Gradient model size", np.sum([np.prod(p.data.shape) for p in self.agent.parameters()]))

    def policy_loss(self, obs, a, ret):
        dist = self.agent.policy_distribution(obs)
        logp = dist.log_prob(a)
        # policy gradient
        return -(logp * ret).mean()

    def get_action(self, obs):
        with torch.no_grad():
            a, logp = self.agent.policy_sample(torch.as_tensor([obs], dtype=torch.float32))
            return a.item(), logp.item()

    def update(self, data):
        obs, a, ret = data['obs'], data['a'], data['ret']

        self.optimizer.zero_grad()
        batch_loss = self.policy_loss(obs, a, ret)
        batch_loss.backward()
        self.optimizer.step()

        self.logger.log("loss_p", batch_loss.detach().item())

    def train(self, ):

        for epoch_i in range(self.params['epochs']):

            # init status
            first_episode_rendered = False
            o = self.env.reset()
            done = False
            r_ep = []
            while self.buffer.size < self.params["batch_size"]:

                a, logp = self.get_action(o)
                o_next, r, done, _ = self.env.step(a)
                if not first_episode_rendered and self.params['render']:
                    self.env.render()

                # save step data
                self.buffer.store(o, a, r)
                r_ep.append(r)

                # end episode
                if done:
                    # compute episode return
                    self.buffer.ep_done()
                    self.logger.log('rets', np.sum(r_ep))
                    self.logger.log('lens', len(r_ep))
                    # update all status
                    first_episode_rendered = True
                    o = self.env.reset()
                    r_ep = []
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
                    "rets": ["mean"],
                    "lens": ['mean'],
                }
                map = {'mean': np.mean}
                self.logger.show(msg, ops, map)