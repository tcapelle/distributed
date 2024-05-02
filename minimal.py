import os, logging
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import wandb
import simple_parsing

N = int(1e10)

def pprint(message):
    rank = os.getenv("RANK", None)
    print(f"[rank{rank}] {message}")

def setup_wandb(config, log_strategy):
    # Setup wandb
    wandb.setup()
    
    rank = int(os.getenv("RANK"))
    local_rank = int(os.getenv("LOCAL_RANK"))

    print(f"rank: {rank}, local_rank: {local_rank}")

    if (rank == 0 and log_strategy == "main"):
        # only log on rank0 process
        wandb.init(project="minimal-wandb", config=config)
    elif (local_rank == 0 and log_strategy == "node"):
        # log on local_rank==0 on each node
        wandb.init(project="minimal-wandb", 
                   name=f"node-{rank}",
                   group=f"grouped-exp-{random.randint(0, N)}",
                   config=config)
    elif log_strategy == "all":
        # log on all processes and group them by rank
        wandb.init(project="minimal-wandb", 
                   name=f"rank-{rank}",
                   group=f"grouped-exp-{random.randint(0, N)}", 
                   config=config)

def train(log_strategy):

    pprint(f"Initializing process group")
    # Initialize the process group
    dist.init_process_group("gloo")


    config = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
    }

    setup_wandb(config, log_strategy)

    pprint("Model and optimizer setup.")
    # Create model and move it to the CPU explicitly
    model = nn.Linear(10, 10)
    model.to(torch.device("cpu"))
    # Wrap model with DDP using CPU
    ddp_model = DDP(model, device_ids=None)  # No device_ids needed for CPU

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=config["learning_rate"])

    pprint("Beginning training.")
    for i in range(config["epochs"]):
        # Example input and target, explicitly on CPU
        inputs = torch.randn(config["batch_size"], 10)
        targets = torch.randn(config["batch_size"], 10)

        # Forward pass
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, targets)
        pprint(f"[Epoch {i}] Loss computed = {loss.item()}.")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pprint("Training step completed.")

    # Cleanup
    dist.destroy_process_group()
    pprint("Cleanup done.")
    

if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_argument("--log_strategy", type=str, default="main", choices=["main", "node", "all"])
    args = parser.parse_args()
    train(args.log_strategy)
