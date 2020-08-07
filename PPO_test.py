import cv2 as cv
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.nn.functional import mse_loss

from utils.utils import *


def bilinear(coor, value):
    x,y = coor
    return value[0] * (1 - x) * (1 - y) + \
           value[1] * x * (1 - y) + \
           value[2] * (1 - x) * y + \
           value[3] * x * y

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


class StateValueModel(nn.Module):
    def __init__(self):
        super(StateValueModel, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32*9*9, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        u = nn.functional.tanh(self.conv1(x))
        u = nn.functional.tanh(self.conv2(u))
        u = nn.functional.tanh(self.fc1(u.reshape(batch_size,-1)))
        u = self.fc2(u)
        return u


class PolicyModel(nn.Module):
    def __init__(self, action_dims):
        super(PolicyModel, self).__init__()
        self.action_dims = action_dims
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32*9*9, 256)
        self.fc2 = nn.Linear(256, self.action_dims)

    def forward(self, x):
        batch_size = x.shape[0]
        u = nn.functional.tanh(self.conv1(x))
        u = nn.functional.tanh(self.conv2(u))
        u = nn.functional.tanh(self.fc1(u.reshape(batch_size,-1)))
        u = self.fc2(u)
        return u


class myAgent():
    def __init__(self, env):
        self.env = env
        self.feature_dims = self.env.observation_space.shape[0]
        self.action_dims = self.env.action_space.n
        self.policy_model = PolicyModel(self.action_dims)
        self.state_value_model = StateValueModel()

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
        self.crop_size = (84,84)

    def preprocess(self, o):
        gray_size = (110,84)
        gray = downsampling(cv.cvtColor(o, cv.COLOR_RGB2GRAY), gray_size)
        [h_d, w_d] = (np.subtract(gray_size, self.crop_size) / 2).astype(np.int)
        cropped = gray[h_d + 5:gray_size[0]-h_d+5, w_d:gray_size[1]-w_d]
        cropped /= 255
        return cropped

    def compute_policy_loss(self, S, A, sample_logp, target):
        e = 0.2
        new_logp = self.agent.policy_logp(S, A)
        ratio = (new_logp - torch.as_tensor(sample_logp, dtype=torch.float32)).exp()
        loss = torch.as_tensor(ratio, dtype=torch.float32) * torch.as_tensor(target, dtype=torch.float32)
        loss_clip = torch.clamp(ratio, 1 - e, 1 + e) * torch.as_tensor(target, dtype=torch.float32)
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

        e = 1
        total_step = 0

        for epoch_i in range(epoch):

            batch_S = []  # for observations
            batch_A = []  # for actions
            batch_P_A = []  # for selection policy prob
            batch_adv = []  # for advantage
            batch_ret = []  # for return

            batch_rets = []  # for measuring episode returns
            batch_lens = []  # for measuring episode lengths

            first_episode_rendered = False

            with torch.no_grad():

                while len(batch_S) < batch_size:
                    print(len(batch_S))
                    # ts = time.time()

                    S = []
                    A = []
                    R = []
                    Sn = []
                    P_A = []
                    Done = []

                    true_s = [np.zeros((84,84), dtype=np.float32)] * 4
                    ep_len = 0
                    def get_action(s):
                        true_s.append(self.preprocess(s))
                        if np.random.rand() <= e:
                            action, logp = self.env.action_space.sample(), 1/self.env.action_space.n
                        else:
                            thi = np.stack(true_s[-4:], axis=-1).transpose(2,0,1)
                            action, logp = self.agent.policy_select(f32tensor([thi]))
                        return action, logp
                    for s, a, r, sn, logp, done in run_episole(self.env, get_action,
                                                               max_ep_len, 4,
                                                               render and not first_episode_rendered):
                        ep_len += 1

                        # clip reward
                        if r > 0:
                            r = 1
                        elif r < 0:
                            r = -1

                        # ignore the done cause by hitting the env time horizon
                        done = False if ep_len == max_ep_len else done

                        # combine last 4 frame
                        true_s = true_s[-4:]
                        thi = np.stack(true_s, axis=-1).transpose(2, 0, 1)
                        thi_next = np.stack(true_s[-3:] + [self.preprocess(sn)], axis=-1).transpose(2, 0, 1)

                        total_step += 1
                        e = np.max([0.1, 1 - (1 - 0.1) / 1e6 * total_step])

                        S.append(thi)
                        A.append(a)
                        R.append(r)
                        Sn.append(thi_next)
                        P_A.append(logp)
                        Done.append(done)

                    # print("time", time.time() - ts)

                    ret = cumulate_return(R, gamma)
                    v_s = self.agent.state_value(S).numpy()
                    v_sn = self.agent.state_value(Sn).numpy()
                    delta = np.array(R) + gamma * v_sn - v_s
                    adv = cumulate_return(delta.tolist(), gamma * lambd)

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

            print('epoch: %3d \t policy_loss: %.3f \t state_value_loss: %.3f \t return: %.3f \t ep_len: %.3f' %
                  (epoch_i, batch_p_loss, batch_v_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == "__main__":
    task = "Breakout-v0"
    env = gym.make(task)
    ag = myAgent(env)
    method = Method(env, ag)
    method.train()  # render=True)