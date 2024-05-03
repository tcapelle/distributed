"""
This is intended to simulate running on a multi-node multi-GPU setup

Run with one node:
$ torchrun --nproc-per-node 2 --master_port=1234 --master_addr=localhost distributed.py

Run with 2 nodes:
- Run on one terminal:
$ torchrun --nproc-per-node 2  --nnodes 2 --master_port=1234 --master_addr=localhost --node_rank 0 distributed.py
- Run on another terminal:
$ torchrun --nproc-per-node 2  --nnodes 2 --master_port=1234 --master_addr=localhost --node_rank 1 distributed.py

You can customize the logging experience by calling wandb.init multiple times. We propose 3 strategies:
- "main": Log only on the main process (rank=0)
- "node": Log on the main process and all local processes (rank=0, local_rank=0)
- "all": Log on all processes (rank=0, local_rank=0)

This will create 2 W&B runs, one for each process.
$ torchrun --nproc-per-node 2  distributed.py --log_strategy all

Note: "main" and "node" behave the same if there is only one node.
"""



import os, logging
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import wandb
import simple_parsing

def rprint(message: str) -> None:
    rank = os.getenv("RANK", None)
    print(f"[rank{rank}] {message}")

def get_world_size_and_rank() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank(), int(os.getenv("LOCAL_RANK"))
    else:
        return 1, 0, 0

def setup_wandb(config, log_strategy, group_name=None):
    "Setup wandb and identify the rank of the current process"
    wandb.setup()
    world, rank, local_rank = get_world_size_and_rank()
    print(f"world: {world}, rank: {rank}, local_rank: {local_rank}")

    if (rank == 0 and log_strategy == "main"):
        # only log on rank0 process
        wandb.init(project="distributed-wandb", 
                   group=group_name,
                   config=config)
    elif (local_rank == 0 and log_strategy == "node"):
        # log on local_rank==0 on each node
        wandb.init(project="distributed-wandb", 
                   name=f"rank-{rank}",
                   group=group_name,
                   config=config)
    elif log_strategy == "all":
        # log on all processes and group them by rank
        wandb.init(project="distributed-wandb", 
                   name=f"rank-{rank}",
                   group=group_name, 
                   config=config)
    if wandb.run:
        # we can update the config to identify the node
        wandb.config.update({"rank": rank,
                             "local_rank": local_rank})

def train():
    rprint("Model and optimizer setup.")
    # Create model and move it to the CPU explicitly
    model = nn.Linear(10, 10)
    model.to(torch.device("cpu"))
    # Wrap model with DDP using CPU
    ddp_model = DDP(model, device_ids=None)  # No device_ids needed for CPU

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=config["learning_rate"])

    rprint("Beginning training.")
    for i in range(config["epochs"]):
        # Example input and target, explicitly on CPU
        inputs = torch.randn(config["batch_size"], 10)
        targets = torch.randn(config["batch_size"], 10)

        # Forward pass
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, targets)
        rprint(f"[Epoch {i}] Loss computed = {loss.item()}.")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    rprint("Training step completed.")

    # Cleanup
    dist.destroy_process_group()
    rprint("Cleanup done.")
    

if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_argument("--log_strategy", type=str, default="main", choices=["main", "node", "all"])
    parser.add_argument("--group_name", type=str, default=None)
    args = parser.parse_args()
    
    config = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
    }
    rprint(f"Initializing process group")
    dist.init_process_group("gloo")

    setup_wandb(config, args.log_strategy, args.group_name)
    train()
