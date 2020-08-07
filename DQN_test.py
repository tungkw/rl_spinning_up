from torch.optim import Adam
from torch.nn.functional import mse_loss
from utils.utils import *
from utils.logger import Logger

device = 'cpu'

class ReplayBuffer():

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((int(size), int(obs_dim)), dtype=np.float32)
        self.act_buf = np.zeros((int(size)), dtype=np.float32)
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

class myAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.feature_dims = self.env.observation_space.shape[0]
        self.action_dims = self.env.action_space.n
        self.action_value_model = mlp([self.feature_dims, 20, self.action_dims])#, activation=nn.ReLU)

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
        # optimizer
        self.q_optimizer = Adam(self.agent.action_value_model.parameters(), lr=self.params['q_lr'])
        # replay buffer
        self.dataset = ReplayBuffer(self.agent.feature_dims, self.agent.action_dims, self.params['replay_size'])
        # log
        self.logger = Logger()
        print(
            "action-value model size {}".format(
                sum([np.prod(p.shape) for p in self.agent.action_value_model.parameters()])
            )
        )

    def get_test_sa(self):
        test_S = []
        test_A = []
        def get_action(s):
            return self.env.action_space.sample(), 0.0
        for s, a, r, sn, _, _ in run_episole(self.test_env, get_action, self.params['max_ep_len']):
            test_S.append(s)
            test_A.append(a)
        return test_S, test_A

    def compute_action_value_loss(self, S, A, target):
        Q = self.agent.action_value(S, A)
        self.logger.log("q", Q.cpu().detach().numpy())
        self.logger.log("y", target.cpu().detach().numpy())
        return mse_loss(Q, target)

    def test(self):
        print('testing episodes')
        def get_action(s):
            action = self.agent.get_action(f32tensor([s]).to(device))
            return action[0], 0.0
        for i in range(self.params['num_test_episodes']):
            R=[]
            for s, a, r, sn, p_a, done in run_episole(self.test_env, get_action, self.params['max_ep_len']):
                R.append(r)
            self.logger.log("rets", np.sum(R).item())
            self.logger.log("lens", len(R))


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

        e = 0.5
        def get_action(s):
            if np.random.rand() <= e:
                return self.env.action_space.sample(), 0.
            else:
                action = self.agent.get_action(f32tensor([s]).to(device))
                return action[0], 0.0

        total_step = 0
        for epoch_i in range(self.params['epochs']):

            epoch_step = 0
            while epoch_step < self.params['steps_per_epoch']:
                ep_len = 0
                for s, a, r, sn, _, done in run_episole(self.env, get_action, self.params['max_ep_len']):
                    ep_len += 1
                    # ignore the done cause by hitting the env time horizon
                    done = False if ep_len == self.params['max_ep_len'] else done

                    self.dataset.store(s, a, r, sn, done)

                    epoch_step += 1
                    total_step += 1
                    e = np.max([0.01, 0.5-(0.5-0.01) / 100000 * total_step])

                    if(total_step > self.params['update_after'] and total_step % self.params['update_every'] == 0):
                        for i in range(self.params['update_every']):
                            batch_data = self.dataset.sample_batch(self.params['batch_size'])
                            self.update(batch_data)

            self.test()
            msg = "epoch: \t {:3d} \n" \
                  "greedy e: \t {:.3f}".format(
                epoch_i,
                e
            )
            ops = {
                "loss_q": ["mean"],
                "rets": ["mean"],
                "lens": ['mean'],
                'q': ['mean', 'max', 'min', 'std'],
                'y': ['mean', 'max', 'min', 'std'],
            }
            map = {'mean': np.mean, 'max': np.max, 'min': np.min, 'std': np.std}
            self.logger.show(msg, ops, map)

if __name__ == '__main__':
    task = 'CartPole-v0'
    env = gym.make(task)
    test_env = gym.make(task)
    ag = myAgent(env)
    params = dict(
        q_lr=1e-3,
        batch_size=32,
        epochs=50,
        steps_per_epoch=4000,
        replay_size=int(1e4),
        update_after=100,
        update_every=1,
        gamma=0.99,
        num_test_episodes=10,
        max_ep_len=1000
    )
    method = Method(env, test_env, ag, params)#, render=True)
    method.train()
