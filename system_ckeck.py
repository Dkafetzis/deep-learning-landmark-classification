import torch
x = torch.rand(5, 3)
print(x)

torch.cuda.is_available()

if torch.cuda.is_available():
    print(torch.cuda.get_device_name())