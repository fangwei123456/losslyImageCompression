import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

'''
minV = 0 maxV = 2

input = [ 0 1 2 1]

v =    0   1   2

p = [ 1/4 2/4 1/4 ]

信息熵为
EL = -p.mul(p.log2()).sum()

反向传播部分

记p的导数为c
由于
y=x*log2(x)
dy/dx = log2(x) + 1/ln2
因此
c = -( torch.ones_like(p) / torch.log(2) + p.log2() ) = [ 0.5573 -0.4427 0.5573 ]
c的shape与p相同
grad_input[i] = c[ input[i] ]
grad_input = [ 0.5573 -0.4427 0.5573 -0.4427]


'''

class EntroyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, minV, maxV):

        # 将区间分为[minV, maxV] 步长为1 统计落入各个值的数量
        # input应该为范围在[minV, maxV]之间的整数
        p = torch.histc(input, bins = (maxV - minV + 1), min = minV, max = maxV)

        p = p / p.shape[0] # 转换为各个数值的概率值

        ctx.save_for_backward(input, p)

        entropy = -p.mul(p.log2()).sum()
        return entropy

    @staticmethod
    def backward(ctx, grad_output):

        input, p = ctx.saved_variables[0]
        c = -( torch.ones_like(p) / torch.log(2) + p.log2() ) # d(EL)/d(p)
        grad_input = torch.zeros_like(input)

        for i in range(input.shape[0]):
            grad_input[i] = c[ input[i] ]

        return grad_input, None, None


class MyMseLoss(nn.Module):
    def __init__(self):
        super(MyMseLoss, self).__init__()

    def forward(self, x, y):
        return torch.pow(x-y,2).mean()

    def backward(self):
        return 1

class MyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, y_pred):
        ctx.save_for_backward(y, y_pred)
        return (y_pred - y).pow(2).sum()

    @staticmethod
    def backward(ctx, grad_output):
        yy, yy_pred = ctx.saved_tensors
        grad_input = 2.0*(yy_pred - yy)
        return -grad_input, None


def main():
    x_data = torch.Tensor([[1.0], [2.0], [3.0]])
    y_data = torch.Tensor([[2.0], [4.0], [6.0]])
    x_data.requires_grad_(requires_grad = True)
    y_data.requires_grad_(requires_grad = True)


    # our model
    model = Model()

    criterion = MyLoss.apply  # Defined loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Defined optimizer

    # Training: forward, loss, backward, step
    # Training loop
    for epoch in range(50):
        # Forward pass
        y_pred = model(x_data)

        # Compute loss
        loss = criterion(y_pred, y_data)
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

