import torch
from torch.autograd import Variable

x=Variable(torch.tensor(1.), requires_grad=True)
y=Variable(torch.tensor(2.), requires_grad=True)
f1 = 6*(x+y)**2
f2 = 6*(x)**2
print(x.grad)

print(torch.autograd.grad(f1, x))
print(torch.autograd.grad(f2,x))
