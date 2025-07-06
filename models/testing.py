import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

def model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='../datasets', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0)

    classes = ['plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  
            x = self.pool(F.relu(self.conv2(x)))  
            x = x.view(-1, 32 * 8 * 8)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = Net().to(device)
    
    # Check if model file exists and load it
    model_path = "../model_cifar10.pth"
    if not os.path.exists(model_path):
        # Try alternative paths
        alt_paths = ["model_cifar10.pth", "../../model_cifar10.pth", 
                     os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cifar10.pth")]
        for path in alt_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            raise FileNotFoundError(f"Could not find model_cifar10.pth in any of the expected locations: {[model_path] + alt_paths}")
    
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Buat confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved as confusion_matrix.png")
    plt.close()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Akurasi pada data test: {accuracy:.2f}%')
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Fungsi untuk un-normalisasi gambar
    def imshow(img, filename='sample_predictions.png'):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().numpy()
        plt.figure(figsize=(12, 6))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved as {filename}")
        plt.close()

    # Tampilkan gambar + label
    imshow(torchvision.utils.make_grid(images[:8]))
    print('Label asli :', ' '.join(f'{classes[labels[j]]}' for j in range(8)))
    print('Prediksi   :', ' '.join(f'{classes[predicted[j]]}' for j in range(8)))
if __name__ == "__main__":
    model()