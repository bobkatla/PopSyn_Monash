import torch
import os
import psutil
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()
print(f"Running on device: {device}")
print(f"Number of GPUs available: {gpu_count}")

# Check the number of CPUs available
cpu_count = os.cpu_count()
cpu_usage = psutil.cpu_percent(interval=1)
print(f"Number of CPUs: {cpu_count}")
print(f"CPU Usage: {cpu_usage}%")

# Perform computation as before
size = 10000
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Measure time and perform matrix multiplication
start_time = time.time()
result = torch.matmul(a, b)
end_time = time.time()

print(f"Computation completed in {end_time - start_time} seconds.")
