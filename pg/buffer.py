import torch
import numpy as np
from utils.utils import cumulate_return


class Buffer:
    def __init__(self, obs_dims, action_dims, max_size):
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.max_size = max_size
        self.obs_buf = np.zeros((max_size, obs_dims), dtype=np.float32)
        self.a_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.r_buf = np.zeros((max_size, 1), dtype=np.float32)
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
        self.r_buf[self.ep_start:self.ptr] = cumulate_return(self.r_buf[self.ep_start:self.ptr], 0.99)
        R = self.r_buf[self.ep_start].item()
        len = self.ptr - self.ep_start
        # for i in reversed(range(self.ep_start, self.ptr-1)):
        #     self.r_buf[i] += self.r_buf[i+1]
        self.ep_start = self.ptr
        return R, len

    def sample(self, batch_size):
        batch = dict(obs=self.obs_buf,
                     a=self.a_buf,
                     r=self.r_buf,
                     )
        self.ptr = 0
        self.size = 0
        self.ep_start = 0
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
