import numpy as np, torch

mylist = []
a = torch.tensor([1,2])
b = torch.tensor([3,4])

mylist.append(a)
mylist.append(b)

print(torch.stack(mylist))
