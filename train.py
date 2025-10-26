import torch
import torch.optim as optim
from model import Net
from data_loader import get_data_loaders
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

CHECKPOINT_DIR = "checkpoints"

def train(model, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(f'Train Epoch: {epoch} Loss: {loss.item():.6f}')

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return test_loss, correct / len(test_loader.dataset)

def save_checkpoint(state, filename="checkpoint.pt"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filename="checkpoint.pt"):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.isfile(filepath):
        print(f"Loading checkpoint {filepath}")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at {filepath}")
        return 0

def evaluate_models_for_plot():
    model_sizes = [1200, 12000, 60000]  # Approximate training data sizes
    model_paths = [
        "models/mnist_cnn_subset_1200.pt",
        "models/mnist_cnn_subset_12000.pt",
        "models/mnist_cnn_full_dataset.pt"
    ]
    
    _, full_test_loader = get_data_loaders() # Use full test set for evaluation

    losses = []
    accuracies = []

    for path in model_paths:
        model = Net()
        # Ensure the model is on the correct device if using GPU
        # model = model.to(device)
        model.load_state_dict(torch.load(path))
        loss, acc = evaluate(model, full_test_loader)
        losses.append(loss)
        accuracies.append(acc)
    
    return model_sizes, losses, accuracies

def main(run_id="default_run", use_full_dataset=True, epochs=14):
    learning_rate = 1.0
    gamma = 0.7

    train_loader, test_loader = get_data_loaders()

    if not use_full_dataset:
        # Create a subset of the datasets for faster training
        train_dataset = torch.utils.data.Subset(train_loader.dataset, range(12000))
        test_dataset = torch.utils.data.Subset(test_loader.dataset, range(2000))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    start_epoch = load_checkpoint(model, optimizer, filename=f"checkpoint_{run_id}.pt")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(start_epoch + 1, epochs + 1):
        train(model, train_loader, optimizer, epoch)
        evaluate(model, test_loader)
        scheduler.step()
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=f"checkpoint_{run_id}.pt")

    torch.save(model.state_dict(), os.path.join("models", f"mnist_cnn_{run_id}.pt"))

if __name__ == '__main__':
    # To generate the plot, uncomment the lines below and comment out main() call
    # model_sizes, losses, accuracies = evaluate_models_for_plot()

    # plt.figure(figsize=(10, 6))
    # plt.plot(model_sizes, losses, marker='o', linestyle='-', color='b')
    # plt.title('Model Test Loss vs. Training Data Size')
    # plt.xlabel('Training Data Size')
    # plt.ylabel('Test Loss (Negative Log Likelihood)')
    # plt.xscale('log') # Use log scale for x-axis if data sizes vary widely
    # plt.grid(True)
    # plt.savefig('training_performance.png')
    # plt.close()

    # print("Generated training_performance.png")

    # Example usage for training with checkpointing
    # To train on full dataset:
    main(run_id="full_dataset", use_full_dataset=True, epochs=14)
    # To train on a subset (e.g., 12000 images):
    # main(run_id="subset_12000", use_full_dataset=False, epochs=10)
