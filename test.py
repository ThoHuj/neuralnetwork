import torch

# Check how many GPUs PyTorch sees
print(f"GPUs found: {torch.cuda.device_count()}")

# Print the name of each GPU
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
