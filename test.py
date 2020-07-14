import torch
import torch.nn as nn
import utils

if __name__ == "__main__":
    model = utils.mlp([3,3,3])
    result = model(torch.as_tensor([[1,2,1]], dtype=torch.float32))
    result.mean().backward()

    state_dict = model.state_dict()
    for name, parameters in state_dict.items():
        print(name)
        print(parameters)
        print(parameters.grad)