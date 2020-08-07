import torch
from torch.distributions.categorical import Categorical
from utils.utils import mlp

class PolicyModel(torch.nn.Module):
    def __init__(self, layer_dims):
        super(PolicyModel, self).__init__()
        self.fc1 = mlp(layer_dims)

    def forward(self, obs):
        return self.fc1(obs)

class Agent(torch.nn.Module):
    def __init__(self, obs_dims, action_dims, hidden_dims):
        super(Agent, self).__init__()
        self.policy_model = PolicyModel([obs_dims] + hidden_dims + [action_dims])

    def policy_select(self, obs):
        with torch.no_grad():
            logits = self.policy_model(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            p_a = dist.log_prob(action)
            return action.item(), p_a.item()

    def policy_loss(self, obs, a, ret):
        logp = Categorical(logits=self.policy_model(obs)).log_prob(a)
        return -(logp * ret).mean()