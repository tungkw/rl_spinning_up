import numpy as np

class Agent:
    def __init__(self, state_size=None, action_size=None, lambd_value=0.0, lambd_policy=0.0, discount=0.9):
        self.discount = discount
        
        # difference
        self.R_mean = 0.0

        # trace
        self.lambd_value = lambd_value
        self.lambd_policy = lambd_policy

        if state_size is not None:
            self.state_size = state_size
            self.v = np.zeros((self.state_size),dtype=float)
        if action_size is not None:
            self.action_size = action_size
            if state_size is not None:
                self.p = np.zeros((self.state_size),dtype=int)
                self.q = np.zeros((self.state_size, self.action_size),dtype=float)

    def state_value(self, state):
        raise NotImplementedError("state_value(): not implemented!")

    def action_value(self, state, action):
        raise NotImplementedError("action_value(): not implemented!")

    def policy(self, state, action):
        raise NotImplementedError("policy(): not implemented!")
    
    def policy_select(self, state):
        raise NotImplementedError("policy_select(): not implemented!")

    def policy_off(self, state, action):
        raise NotImplementedError("policy_off(): not implemented!")

    def policy_off_select(self, state):
        raise NotImplementedError("policy_off_select(): not implemented!")
    
    def reward(self, state, action):
        raise NotImplementedError("reward(): not implemented!")

    def get_actions(self, state):
        raise NotImplementedError("get_actions(): not implemented!")
    
    def new_episode(self):
        raise NotImplementedError("new_episode(): not implemented!")

    def print_evaluation(self):
        print("'print_evaluation()' not implemented")
    
    def print_improvement(self):
        print("'print_improvement()' not implemented")
