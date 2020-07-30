import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import mse_loss
import gym
from utils import *
from logger import Logger
from copy import deepcopy
import matplotlib.pyplot as plt
import cv2 as cv
import time

device = 'cuda:0'
# device = 'cpu'

def downsampling(img, dsize):
    osize = img.shape[:2]
    img_d = np.zeros(dsize, dtype=np.float)
    for i in range(dsize[0]):
        x = i / dsize[0] * osize[0]
        u = np.floor(x).astype(np.int)
        x = x-u
        for j in range(dsize[1]):
            y = j / dsize[1] * osize[1]
            v = np.floor(y).astype(np.int)
            y = y-v
            img_d[i][j] = bilinear([x,y], img[u:np.min([u+2,osize[0]]), v:np.min([v+2,osize[1]])].reshape(-1).tolist())
    return img_d

def bilinear(coor, value):
    x,y = coor
    return value[0] * (1 - x) * (1 - y) + \
           value[1] * x * (1 - y) + \
           value[2] * (1 - x) * y + \
           value[3] * x * y

class ActionValueModel(nn.Module):
    def __init__(self, action_dims):
        super(ActionValueModel, self).__init__()
        self.action_dims = action_dims
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32*9*9, 256)
        self.fc2 = nn.Linear(256, self.action_dims)
        # self.conv1 = nn.Conv2d(4, 4, 8, 4)
        # self.conv2 = nn.Conv2d(4, 4, 4, 2)
        # self.fc1 = nn.Linear(4*9*9, 4)
        # self.fc2 = nn.Linear(4, self.action_dims)

    def forward(self, x):
        batch_size = x.shape[0]
        u = nn.functional.tanh(self.conv1(x))
        u = nn.functional.tanh(self.conv2(u))
        u = nn.functional.tanh(self.fc1(u.reshape(batch_size,-1)))
        u = self.fc2(u)
        return u

class ReplayBuffer():

    def __init__(self, obs_dim, size):
        self.obs_buf = np.zeros((int(size), *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((int(size)), dtype=np.float32)
        self.r_buf = np.zeros((int(size)), dtype=np.float32)
        self.obs_next_buf = np.zeros((int(size), *obs_dim), dtype=np.float32)
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


class myAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.feature_dims = list(self.env.observation_space.shape)
        self.action_dims = self.env.action_space.n
        self.action_value_model = ActionValueModel(self.action_dims)

    def policy_select(self, state):
        action_values = self.action_value_model(torch.as_tensor(state, dtype=torch.float32).to(device))
        action_best = torch.argmax(action_values, dim=-1)
        return action_best

    def action_value(self, state, action):
        action_values = self.action_value_model(f32tensor(state).to(device))
        value = action_values[
            torch.arange(0,action_values.shape[0],dtype=torch.long, device=device),
            torch.as_tensor(action, dtype=torch.long).to(device)]
        return value

    def get_action(self, state):
        with torch.no_grad():
            action = self.policy_select(state)
            return action.cpu().numpy()


class Method:
    def __init__(self, env, test_env, agent, params, render=False):
        self.env = env
        self.test_env = test_env
        self.agent = agent
        self.params = params
        self.render = render

        self.agent.to(device=device)
        for p in self.agent.parameters():
            nn.init.normal_(p.data, 0, 1)
        # optimizer
        self.q_optimizer = Adam(self.agent.action_value_model.parameters(), lr=self.params['q_lr'])
        # replay buffer
        self.crop_size = (84,84)
        self.dataset = ReplayBuffer(list([4,*self.crop_size]), self.params['replay_size'])
        # test state-action
        self.test_S, self.test_A = self.get_test_sa()
        print('test data prepared (size {})'.format(len(self.test_S)))
        # log
        self.logger = Logger()
        print(
            "action-value model size {}".format(
                sum([np.prod(p.shape) for p in self.agent.action_value_model.parameters()])
            )
        )

    def preprocess(self, o):
        gray_size = (110,84)
        gray = downsampling(cv.cvtColor(o, cv.COLOR_RGB2GRAY), gray_size)
        [h_d, w_d] = (np.subtract(gray_size, self.crop_size) / 2).astype(np.int)
        cropped = gray[h_d + 5:gray_size[0]-h_d+5, w_d:gray_size[1]-w_d]
        cropped /= 255;
        return cropped

    def get_test_sa(self):
        test_S = []
        test_A = []

        true_s = [np.zeros((84,84),dtype=np.float32)] * 4
        def get_action(s):
            true_s.append(self.preprocess(s))
            thi = np.stack(true_s[-4:], axis=-1).transpose(2,0,1)
            action, logp = self.agent.get_action(f32tensor([thi]).to(device)), 0.0
            return action, logp

        i = 0
        for s, a, r, sn, _, done in run_episole(self.env, get_action, self.params['max_ep_len'], step=4):
            i+=1
            true_s = true_s[-4:]
            thi = np.stack(true_s, axis=-1).transpose(2,0,1)
            test_S.append(thi)
            test_A.append(a)
        return test_S, test_A

    def compute_action_value_loss(self, S, A, target):
        Q = self.agent.action_value(S, A)
        self.logger.log("q", Q.cpu().detach().numpy())
        self.logger.log("y", target.cpu().detach().numpy())
        return mse_loss(Q, target)

    def test(self):
        print('testing test state-action')
        with torch.no_grad():
            S = f32tensor(self.test_S).to(device)
            A = f32tensor(self.test_A).to(device)
            Q = self.agent.action_value(S, A)
            self.logger.log('test_q', Q.cpu().numpy())

    def update(self, data):
        S, A, R, Sn, Done = \
            data['obs'].to(device), data['act'].to(device), data['r'].to(device), data['obs_next'].to(device), data['done'].to(device)

        # value function iteration
        with torch.no_grad():
            A_max = self.agent.policy_select(Sn)
            Q = self.agent.action_value(Sn, A_max)
            Y = R + self.params['gamma'] * (1 - Done) * Q
        self.q_optimizer.zero_grad()
        loss_q = self.compute_action_value_loss(S, A, Y)
        loss_q.backward()
        self.q_optimizer.step()

        self.logger.log("loss_q", loss_q.item())

    def train(self):
        e = 1
        x = []
        y = []
        total_step = 0
        for epoch_i in range(self.params['epochs']):
            ts = time.time()

            epoch_step = 0
            while epoch_step < self.params['steps_per_epoch']:
                print("step {} in epoch {}".format(epoch_step, epoch_i))

                # episole
                R = []
                true_s = [np.zeros((84,84), dtype=np.float32)] * 4
                ep_len = 0
                def get_action(s):
                    true_s.append(self.preprocess(s))
                    if np.random.rand() <= e:
                        action, logp = self.env.action_space.sample(), 0.0
                    else:
                        thi = np.stack(true_s[-4:], axis=-1).transpose(2,0,1)
                        action, logp = self.agent.get_action(f32tensor([thi]).to(device)), 0.0
                    return action, logp
                for s, a, r, sn, _, done in run_episole(self.env, get_action, self.params['max_ep_len'], step=4):
                    ep_len += 1
                    # clip reward
                    if r > 0:
                        r = 1
                    elif r < 0:
                        r = -1
                    R.append(r)

                    # ignore the done cause by hitting the env time horizon
                    done = False if ep_len == self.params['max_ep_len'] else done

                    true_s = true_s[-4:]
                    thi = np.stack(true_s, axis=-1).transpose(2, 0, 1)
                    thi_next = np.stack(true_s[-3:] + [self.preprocess(sn)], axis=-1).transpose(2, 0, 1)
                    self.dataset.store(thi, a, r, thi_next, done)

                    total_step += 1
                    epoch_step += 1
                    e = np.max([0.1, 1 - (1-0.1)/1e6 * total_step])

                    # train
                    if total_step >= self.params['update_after'] and total_step % self.params['update_every'] == 0:
                        for i in range(self.params['update_steps']):
                            batch_data = self.dataset.sample_batch(batch_size=self.params['batch_size'])
                            self.update(batch_data)

                self.logger.log("rets", np.sum(R).item())
                self.logger.log("lens", len(R))

            self.test()
            msg = "epoch: \t {:3d} \n" \
                  "time: \t {:.3f} \n" \
                  "greedy e: \t {:.3f}".format(
                epoch_i,
                time.time() - ts,
                e
            )
            ops = {
                "loss_q": ["mean"],
                'test_q':['mean'],
                "rets": ["mean"],
                "lens": ['mean'],
                'q': ['mean', 'max', 'min', 'std'],
                'y': ['mean', 'max', 'min', 'std'],
            }
            map = {'mean': np.mean, 'max': np.max, 'min': np.min, 'std': np.std}
            x.append(epoch_i)
            y.append(np.mean(self.logger.log_data['test_q']))
            # plt.ion()
            # plt.plot(x,y)
            # plt.pause(0.1)
            self.logger.show(msg, ops, map)
        # plt.ioff()
        plt.plot(x,y)
        plt.show()


if __name__ == '__main__':
    task = "Breakout-v0"
    env = gym.make(task)
    test_env = gym.make(task)
    ag = myAgent(env)
    params = dict(
        epochs=100,
        steps_per_epoch=500,
        replay_size=int(1e4),
        batch_size=32,
        update_after=100,
        update_every=1,
        update_steps=1,
        gamma=0.99,
        q_lr=1e-3,
        num_test_episodes=10,
        max_ep_len=1000
    )
    method = Method(env, test_env, ag, params)#, render=True)
    method.train()
