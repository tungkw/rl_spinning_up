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

def run_episole(env, agent, max_ep_len, render = False):
    t = 0
    S = [env.reset()]
    a, p_a = agent.policy_select(S[t])
    A = [a]
    P_A = [p_a]
    R = [0]
    while True:
        if render:
            env.render()
        S_next, r, done, _ = env.step(A[t])
        S.append(S_next)
        a, p_a = agent.policy_select(S_next)
        A.append(a)
        P_A.append(p_a)
        R.append(r)
        t+=1
        if done or t > max_ep_len:
            break
    return S, A, R, P_A, done

