import torch
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
y = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
torch.matmul(x, y)
print("GPU 正常工作 ✅")