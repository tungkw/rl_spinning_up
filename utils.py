import numpy as np
import torch
import torch.nn as nn

def mlp(sizes, activation=nn.Tanh, output_activation=None):
    layers = []
    for j in range(len(sizes)-1):
        layers += [nn.Linear(sizes[j], sizes[j+1])]
        if j < len(sizes)-2:
            layers += [activation()]
    if output_activation is not None:
        layers += [output_activation()]
    return nn.Sequential(*layers)

def cumulate_return(R, discount):
    G = 0
    ret = []
    for i in reversed(range(len(R))):
        G = R[i] + discount * G
        ret = [G] + ret
    return ret

def run_episole(env, sample_func, max_ep_len, render = False):
    t = 0
    s = env.reset()
    a, logp = sample_func(s)
    done = False
    while not done and t < max_ep_len:
        if render:
            env.render()
        sn, r, done, _ = env.step(a)
        yield s, a, r, sn, logp, done
        s = sn
        a, logp = sample_func(s)
        t += 1


if __name__ == '__main__':
    import gym
    env = gym.make("HalfCheetah-v2")
    def sample_func(s):
        a = env.action_space.sample()
        p_a = 0
        return a,p_a
    for data in run_episole(env, sample_func, 10):
        print(data)