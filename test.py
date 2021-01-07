import torch
from torch.autograd import Variable
import torch.nn as nn

class net(nn.Module):
    y = 0
    def __init__(self):
        super(net, self).__init__()
        self.l = nn.Linear(2,2)
        self.noisy = nn.ModuleList([
            nn.Linear(2,2)
        ])

    def forward(self, x, y):
        net.y = y
        h = self.l(x)
        h = self.noisy[0](h)
        h = self.noisy[1](h)
        return h

def hook(module, input, output):
    if net.y==1:
        print('{}:{}'.format(module,output))
    else:
        print('y=',net.y)


model = net()
model.noisy.append(nn.Linear(2,2))
for i, (name, module) in enumerate(model.named_modules()):
    if isinstance(module, nn.Linear):
        module.register_forward_hook(hook)
        print(name)

x1 = torch.tensor([[2,2]]).float()
x2 = torch.tensor([[3,3]]).float()
out1 = model(x1, 1)
out2 = model(x2, 0)
print('out1, out2={},{}'.format(out1,out2))

class First(object):
    def __init__(self):
        print("first")

class Second(object):
    def __init__(self):
        print("second")

class Third(First, Second):
    def __init__(self):
        super().__init__()
        Second.__init__(self)
        print("that's it")

t = Third()

class FooMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self.foo = 'foo'
        self.array = [1]
        print(self.array)
        self.bar = Bar(self.array)
        print(self.array)

class Bar(object):
    def __init__(self, array):
        self.array = array
        self.array.append(2)

f = FooMixin()

