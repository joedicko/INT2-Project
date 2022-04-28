import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, ToPILImage


# Loads CIFAR10 into training_data
# Filepath = root, training indicator true, downloads if not available
# Option to transform to 3d tensors
training_data = datasets.CIFAR10(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)


# Loads CIFAR10 into testing_data
# Filepath = root, training indicator false, downloads if not available
# Option to transform to 3d tensors
testing_data = datasets.CIFAR10(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)


# Provides a dictionary for mapping label id to label text
labels_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}


# Function to display training data
def display_training(size):
    figure = plt.figure(figsize=(8,8))
    cols, rows = size, size
    for i in range(1, cols * rows + 1):
        random_index = np.random.randint(len(training_data))
        img, label = training_data[random_index]
        figure.add_subplot(rows, cols, i)
        plt.xticks([])
        plt.yticks([])
        print(img)
        plt.imshow(ToPILImage()(img), cmap="gray")
        plt.xlabel(labels_map[label])
    plt.show()
#display_training(4)


# Loads data sets into DataLoader for batch processing
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

# Iterating through DataLoader
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0]
label = train_labels[0]
plt.imshow(ToPILImage()(img), cmap="gray")
plt.xticks([])
plt.yticks([])
print(f"Label: {label}")
plt.show()

