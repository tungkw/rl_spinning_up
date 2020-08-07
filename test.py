import torch
import torch.nn as nn


class testModle(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(2,1)#, bias=False)
        # self.l2 = nn.Linear(32,32)
        # self.l3 = nn.Linear(32,1)
        # self.w = torch.nn.Parameter(torch.as_tensor([[0.9,0.8]], dtype=torch.float32))

    def forward(self, x):
        u = x.clone()
        u[:,0] = torch.log(u[:,0])
        u[:,1] = torch.log(u[:,1])
        u = self.l(u)
        u = torch.exp(torch.exp(u))

        return u

if __name__ == "__main__":
    model = testModle()
    for p in model.parameters():
        print(p.data.requires_grad)
        print(p.requires_grad)
    x = torch.randint(2,5,(1000,2), dtype=torch.float32)
    y = torch.prod(x, dim=1)
    print("y", y)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(100):
        optimizer.zero_grad()
        y_ = model(x)
        loss = torch.nn.functional.mse_loss(y_, y)
        loss.backward()
        optimizer.step()
        print(loss.item())
        print([p for p in model.parameters()])

    test_x = torch.randint(2,5,(10,2), dtype=torch.float32)
    test_y = torch.pow(test_x[:, [0]], test_x[:, [1]])
    y_ = model(test_x)
    loss = torch.nn.functional.mse_loss(y_, test_y)
    print(test_x)
    print(y_)
    print(test_y)
    print(loss.item())