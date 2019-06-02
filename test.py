import torch

from torch.utils.cpp_extension import load
entropy_loss_cuda = load(
    'entropy_loss_cuda', ['./pytorch_entropy_loss/entropy_loss_cuda.cpp', './pytorch_entropy_loss/entropy_loss_cuda_kernel.cu'], verbose=True)
help(entropy_loss_cuda)

from pytorch_entropy_loss.entropy_loss import EL

x = torch.randint(0,8,[2,4,4,4]).cuda() # 范围0-7
print(x)
print(x[0].view(-1).histc(bins=8,min=0,max=7))
print(x[1].view(-1).histc(bins=8,min=0,max=7))

xf = x.cuda().float().requires_grad_(True)


el = EL.apply

y = el(xf,0,7)
print(y)

y.backward()

print(xf.grad)



