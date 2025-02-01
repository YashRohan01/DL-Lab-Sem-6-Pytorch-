import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.train()  # Set the model to training mode
    Losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute predictions
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)
            Losses.append(loss.item())

            # Backward pass: compute gradients
            loss.backward()

            # Update the weights
            optimizer.step()

            # Track the loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print the statistics every epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(conf_matrix)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of learnable parameters in the model: {num_params}")
    return Losses


class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier,self).__init__()
        self.fc1 = nn.Linear(784,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
    def forward(self,x):
        x = x.view(-1,784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),           # Convert images to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std
])

data_directory = './data'  # Change this to the actual path

train_dataset = datasets.MNIST(root=data_directory, train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root=data_directory, train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

X_train = train_dataset.data.view(-1, 28*28).float()  # Flatten 28x28 images to 784 features
y_train = train_dataset.targets
X_test = test_dataset.data.view(-1, 28*28).float()  # Flatten 28x28 images to 784 features
y_test = test_dataset.targets

# Check shapes of X and y
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

model = MNIST_Classifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.002)

Losses = train(model, train_loader, criterion, optimizer, device, num_epochs=10)
plt.plot(Losses)
plt.show()
