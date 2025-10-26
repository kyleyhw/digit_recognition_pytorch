import torch
import torch.optim as optim
from model import Net
from data_loader import get_data_loaders
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

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
        model.load_state_dict(torch.load(path))
        loss, acc = evaluate(model, full_test_loader)
        losses.append(loss)
        accuracies.append(acc)
    
    return model_sizes, losses, accuracies

def main():
    epochs = 14
    learning_rate = 1.0
    gamma = 0.7

    train_loader, test_loader = get_data_loaders()

    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch)
        evaluate(model, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "mnist_cnn_full_dataset.pt")

if __name__ == '__main__':
    # main() # Comment out main training when generating plot
    model_sizes, losses, accuracies = evaluate_models_for_plot()

    plt.figure(figsize=(10, 6))
    plt.plot(model_sizes, losses, marker='o', linestyle='-', color='b')
    plt.title('Model Test Loss vs. Training Data Size')
    plt.xlabel('Training Data Size')
    plt.ylabel('Test Loss (Negative Log Likelihood)')
    plt.xscale('log') # Use log scale for x-axis if data sizes vary widely
    plt.grid(True)
    plt.savefig('training_performance.png')
    plt.close()

    print("Generated training_performance.png")
