import numpy as np
import gym
import torch
import torch.nn as nn
from torch.optim import Adam

from pg.agent import Agent as PGAgent
from pg.buffer import Buffer as PGBuffer
from utils.logger import Logger

class Method:
    def __init__(self, env, agent, buffer, params):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.params = params
        self.optimizer = Adam(self.agent.parameters(), lr=self.params['lr'])
        self.logger = Logger()
        print("model size", np.sum([np.prod(p.data.shape) for p in self.agent.parameters()]))

    def get_action(self, obs):
        a, logp = self.agent.policy_select(torch.as_tensor([obs], dtype=torch.float32))
        return a, logp

    def update(self):
        self.optimizer.zero_grad()
        batch_data = self.buffer.sample(self.params['batch_size'])
        batch_loss = self.agent.policy_loss(batch_data['obs'], batch_data['a'], batch_data['r'])
        batch_loss.backward()
        self.optimizer.step()
        self.logger.log("loss_p", batch_loss.detach().item())

    def train(self, ):

        for epoch_i in range(self.params['epochs']):

            # init status
            first_episode_rendered = False
            o = self.env.reset()
            done = False
            while self.buffer.size < self.params["batch_size"]:
                a, logp = self.get_action(o)
                o_next, r, done, _ = self.env.step(a)
                if not first_episode_rendered and self.params['render']:
                    self.env.render()

                # save step msg
                self.buffer.store(o, a, r)

                # end episode
                if done:
                    # compute episode return
                    R, l = self.buffer.ep_done()
                    # print(R)
                    # log
                    self.logger.log('rets', R)
                    self.logger.log('lens', l)
                    # update all status
                    first_episode_rendered = True
                    o = self.env.reset()
                else:
                    o = o_next
            if not done:
                self.buffer.ep_done()

            # loss and backward
            self.update()

            # print log msg
            msg = "epoch: \t {:3d} \n".format(
                epoch_i
            )
            ops = {
                "loss_p": ["mean"],
                "rets": ["mean"],
                "lens": ['mean'],
            }
            map = {'mean': np.mean}
            self.logger.show(msg, ops, map)



if __name__ == "__main__":
    params = dict(
        task="CartPole-v0",
        hidden_dim=32,
        layers=1,
        epochs=50,
        batch_size=4000,
        gamma=0.99,
        lr=1e-2,
        render=False
    )

    # gym environment
    env = gym.make(params['task'])

    # agent
    obs_dims = env.observation_space.shape[0]
    action_dims = env.action_space.n
    hidden_dims = [params["hidden_dim"]] * params["layers"]
    ag = PGAgent(obs_dims, action_dims, hidden_dims)

    # data keeper
    buffer = PGBuffer(obs_dims, action_dims, params['batch_size'])

    # train
    method = Method(env, ag, buffer, params)
    method.train()