import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class EntroyLoss(nn.Module):
    def __init__(self):
        super(EntroyLoss, self).__init__()
    def forward(self, input, minV, maxV):

        p = torch.histc(input, bins = (maxV - minV + 1), min = minV, max = maxV)

        p = p / p.shape[0] # 转换为各个数值的概率值

        entropy = -p.mul(p.log2()).sum()
        return entropy


class MyMseLoss(nn.Module):
    def __init__(self):
        super(MyMseLoss, self).__init__()

    def forward(self, x, y):
        return torch.pow(x-y,2).mean()

    def backward(self):
        return None



def main():
    x_data = torch.Tensor([[1.0], [2.0], [3.0]])
    y_data = torch.Tensor([[2.0], [4.0], [6.0]])
    x_data.requires_grad_(requires_grad = True)
    y_data.requires_grad_(requires_grad = True)


    # our model
    model = Model()

    criterion = EntroyLoss(x_data.shape)  # Defined loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Defined optimizer

    # Training: forward, loss, backward, step
    # Training loop
    for epoch in range(50):
        # Forward pass
        y_pred = model(x_data)

        # Compute loss
        loss = criterion(y_pred, -50, 50)
        print(epoch, loss)

        # Zero gradients
        optimizer.zero_grad()
        # perform backward pass
        loss.backward()
        # update weights
        optimizer.step()

    # After training
    hour_var = torch.Tensor([[4.0]])
    print("predict (after training)", 4, model.forward(hour_var).data[0][0])


if __name__ == '__main__': # 如果运行本py文件 就运行main函数
    main()

