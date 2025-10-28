import arch_config  # noqa: F401 - Configure Blackwell optimizations
# before_dataparallel.py
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Dummy model and dataset
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

def main():
    # Setup model and data
    input_size = 1024
    hidden_size = 256
    model = SimpleNet(input_size, hidden_size)
    
    # Check if we have multiple GPUs
    if torch.cuda.device_count() < 2:
        print(f"This example requires at least 2 GPUs, but only {torch.cuda.device_count()} available.", flush=True)
        print("Running single-GPU training instead.", flush=True)
        
        # Single GPU training
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        data = torch.randn(512, input_size).to(device)
        target = torch.randn(512, 1).to(device)
        
        # Timing a single training step
        torch.cuda.synchronize()
        start = time.time()
        
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"Single-GPU training step took {elapsed*1000:.2f} ms", flush=True)
        return
    
    model.cuda()  # move model to GPU 0, it will also replicate to GPU 1
    model = nn.DataParallel(model)  # utilize 2 GPUs (0 and 1 by default)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    data = torch.randn(512, input_size).cuda()  # batch of 512 on GPU0
    target = torch.randn(512, 1).cuda()  # target on GPU0
    
    # Timing a single training step
    torch.cuda.synchronize()
    start = time.time()
    
    output = model(data)  # forward (DP splits data internally)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()  # backward (DP gathers grads to GPU0)
    optimizer.step()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"DataParallel step took {elapsed*1000:.2f} ms", flush=True)

if __name__ == "__main__":
    main()
