import numpy as np

class Method:
    def __init__(self, agent):
        self.agent = agent
    
    def learn(self, epoch=10, step=1, threshold=0.1, show=False):
        for episode in range(epoch):
            step_ = step
            t = 0
            tau = 0
            T = float('inf')
            S = [self.agent.new_episode()]
            A = [self.agent.policy_select(S[0])]
            R = [0.0]
            while True:
                print(t, end='\r')
                if t < T:
                    sn, r = self.agent.act(S[t], A[t])
                    an = self.agent.policy_select(sn)
                    S.append(sn)
                    A.append(an)
                    R.append(r)
                    if self.agent.stop_state(sn):
                        T = t+1
                        step_ = min(step_, T)
                tau = t - step_ + 1
                if tau >= 0:
                    G = 0.0
                    for k in reversed(range(tau+1, min(tau+step_, T)+1)):
                        if k == T:
                            G = R[k]-self.agent.R_mean
                        elif k == tau + step_:
                            action_value = self.agent.action_value(S[k], A[k])
                            # action_value = self.agent.action_value(S[k], self.agent.p[S[k]])
                            # action_value = np.sum([self.agent.policy(S[k], a) * self.agent.action_value(S[k], a) for a in self.agent.get_actions(S[k])])
                            G = R[k]-self.agent.R_mean + self.agent.discount * action_value
                        else:
                            # mean_value = np.sum([self.agent.policy(S[k], a) * self.agent.action_value(S[k], a) for a in self.agent.get_actions(S[k])])                  
                            # diff = G - self.agent.action_value(S[k],A[k])
                            # G = R[k] + self.agent.discount * self.agent.policy(S[k],A[k]) * diff + self.agent.discount * mean_value
                            G = R[k]-self.agent.R_mean + self.agent.discount * G
                    self.agent.update(tau, S[tau], A[tau], G)
                    self.agent.print_t(tau, S[tau], A[tau], R[tau+1], S[tau+1], A[tau+1], G)
                if tau == T-1:
                    break
                t += 1
            self.agent.print_e(episode, S, A, R)