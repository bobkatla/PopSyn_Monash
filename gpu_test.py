import torch
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Create two large random matrices on the GPU
size = 10000
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Start the timer
start_time = time.time()

# Perform matrix multiplication on GPU
result = torch.matmul(a, b)

# End the timer
end_time = time.time()

print(f"Computation completed in {end_time - start_time} seconds.")
