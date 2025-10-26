import torch
import torch.optim as optim
from model import Net
from data_loader import get_data_loaders
from tqdm import tqdm

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
    main()
