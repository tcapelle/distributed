import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

def example(rank, world_size):
    print(f"Initializing process group for rank {rank}...")
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    print(f"Process {rank}: Model and optimizer setup.")
    # Create model and move it to the CPU explicitly
    model = nn.Linear(10, 10)
    model.to(torch.device("cpu"))
    # Wrap model with DDP using CPU
    ddp_model = DDP(model, device_ids=None)  # No device_ids needed for CPU

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    print(f"Process {rank}: Beginning training step.")
    # Example input and target, explicitly on CPU
    inputs = torch.randn(20, 10)
    targets = torch.randn(20, 10)

    # Forward pass
    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, targets)
    print(f"Process {rank}: Loss computed = {loss.item()}.")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Process {rank}: Training step completed.")

    # Cleanup
    dist.destroy_process_group()
    print(f"Process {rank}: Cleanup done.")

def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    world_size = 4  # Number of processes to simulate
    print("Starting multiprocessing...")
    mp.spawn(example,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
